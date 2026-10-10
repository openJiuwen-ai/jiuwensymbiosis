# Use the Trace Feedback Loop

> Category: How-to. The [Chinese source](../../zh/how-to/use-trace-feedback.md) is authoritative.

> This module upgrades `TraceRail` from a "replayable record" into a **two-layer feedback system**:
>
> - **Online**: after a failed step, `DiagnosisRail` feeds the current parameters + relevant history + system state back into the next LLM turn, so the model can self-correct within the same run.
> - **Offline**: `analyze_traces` clusters failed steps across many runs in batch, producing a failure report and SKILL.md patch proposals for human review.
>
> For the collection layer (`TraceRail` / trace JSON format / replay) see the [Tracing reference](../reference/tracing.md); for the design rationale see the [Trace Feedback Loop design](../../../design/trace-feedback-loop.md). This page only covers usage.

---

## 1. Overview of the two feedback layers

| Layer | Module | When it runs | Output | Audience |
|----|------|----------|------|------|
| **Online** | `DiagnosisRail` | After a single failure, before the next LLM turn | A synthetic user diagnosis message | The LLM (self-correction in the same run) |
| **Offline** | `jiuwensymbiosis.trace_feedback` + `scripts/analyze_traces.py` | Batch trace analysis | `failure_clusters.json` / `failure_report.md` / `skill_patch_proposals.md` | Engineers (human review, edit SKILL.md) |

Both layers share one trace substrate (the JSON `TraceRail` persists) and neither depends on the other: you can run online only, or run offline analysis over already-persisted traces only.

---

## 2. Online mode: `DiagnosisRail`

### 2.1 What it does

When a tool call fails, `DiagnosisRail` appends a synthetic user message **before the next LLM call**, in three parts:

1. **The current failed step**: tool name, parameters, rail/log events, error.
2. **Relevant history (the causal chain)**: the most recent N steps with the same tool name or the same class of rail event, flagging "is this failing repeatedly".
3. **System state**: RecoveryRail's home/release outcome + the current pose, flagging "the arm's state has changed".

Seeing this, the LLM can change parameters or strategy instead of retrying blindly. The diagnosis message has a soft token cap; when it is exceeded, content is dropped in the order "history → system state", and **the current step is always kept**.

### 2.2 Enable it in configuration (recommended)

Select stepagent and enable tracing and diagnosis in the task YAML:

```yaml
agent:
  modules:
    tracing:
      enabled: true
    diagnosis:
      enabled: true
      max_chars: 1500
      history_steps: 3
      history_kinds:
      - reject
      - recover
  execution:
    mode: stepagent
```

`modules.diagnosis.enabled` depends on `modules.tracing.enabled`; enabling diagnosis with tracing off raises an error. Diagnosis requires stepagent.

### 2.3 Enable it in code

```python
from jiuwensymbiosis.agent import RobotAgentConfig

config = RobotAgentConfig(
    modules={'tracing': {'enabled': True}, 'diagnosis': {'enabled': True}},
    execution={'mode': 'stepagent'},
)
```

### 2.4 Failure channels covered

| Type | Trigger | Typical scenario | Diagnosis source |
|------|------|----------|----------|
| **Type A (catch-path)** | The tool turns the exception into `ToolOutput(success=False, error=...)` | SKILL mode, `RobotControlTool` dispatch | `tool_result.error` + the current entry |
| **Type B (propagated)** | The exception escapes the tool / a before-hook | Non-SKILL, `@implements` exposed directly; `SafetyRail` raises `ValueError` | `ctx.exception` |

The same step is never injected twice (a per-step idempotency marker).

### 2.5 Fast-path behavior

The fast path (`run_fast_task`) has no per-step LLM and rejects `modules.diagnosis.enabled=true`. It supports tracing, and its trace JSON can be included in offline analysis.

### 2.6 Diagnosis-message shape

```text
### Diagnosis: the previous step failed
[diagnosis] step failed: goto_xyzr
  error: SafetyRail: z=-50 below z_floor=10
  params: {'x': 150, 'y': 0, 'z': -50, 'r': 0}
  rail: SafetyRail/reject {'tool_name': 'goto_xyzr', 'reason': 'z=-50 below z_floor=10'}

### Related history (possibly failing repeatedly)
  - #2 goto_xyzr({'x': 120, 'y': 0, 'z': -40, 'r': 0}) → FAIL: SafetyRail: z below floor

### System state
  pose: {'x': 120.0, 'y': 0.0, 'z': 80.0}

Correct the parameters or change strategy accordingly; do not retry with the same parameters.
```

---

## 3. Offline mode: `analyze_traces`

### 3.1 What it does

Load a batch of trace JSON, extract every failed step, cluster them by normalized signature, and produce three reports:

| File | Content | For whom |
|------|------|------|
| `failure_clusters.json` | Machine-readable clustering result (signature / count / examples / affected conversations) | Post-processing scripts |
| `failure_report.md` | Human-readable report: overview + per-cluster tool/rail/reason/param bucket/evidence steps | Engineers reviewing |
| `skill_patch_proposals.md` | SKILL.md patch proposals for human review: template diff + anchor + risks + validation suggestions | Engineers editing skills |

**No source file is changed automatically** — every suggestion is written to reports, and a human edits SKILL.md after review.

### 3.2 Quick start

```bash
# Analyze the complete Trace directory
python scripts/analyze_traces.py \
  --trace-dir ~/.jiuwensymbiosis/piper_workspace/traces \
  --out reports/trace_feedback/latest \
  --min-cluster-size 3

# Debug one Trace file
python scripts/analyze_traces.py --trace path/to/one_trace.json --out /tmp/out
```

Output lands in the `--out` directory (default `reports/trace_feedback/latest`).

### 3.3 CLI options

| Option | Default | Meaning |
|------|------|------|
| `--trace-dir <DIR>` | — | Directory of trace JSON (top-level `*.json`, not recursive) |
| `--trace <FILE>` | — | A single trace JSON (for debugging; mutually exclusive with `--trace-dir`) |
| `--out <DIR>` | `reports/trace_feedback/latest` | Output directory |
| `--min-cluster-size <N>` | `3` | Minimum cluster size; anything smaller is not reported |
| `--context-steps <N>` | `2` | Context steps kept around a failed step (evidence `before/after_context`) |

### 3.4 Exit codes

| Exit code | Meaning |
|--------|------|
| `0` | Completed normally. Includes "valid traces but no failed step" — an empty report is written, which is not an error |
| `1` | Input error: path does not exist / no trace source given / no valid trace after loading (including a directory whose JSON files are all corrupt) |

### 3.5 Clustering rules: how `FailureSignature` is built

Failures with the same signature cluster together. The signature is composed of these fields:

| Field | Source | Normalization |
|------|------|--------|
| `tool_name` | entry | Raw value |
| `rail_name` / `kind` | The first `success=False` rail event (only SafetyRail/reject counts as a root cause; RecoveryRail/recover is a remedy, never a root cause) | Raw value |
| `reason_norm` | `detail["reason"]` for SafetyRail, otherwise `entry.error` | `trim+lower` + numbers replaced by `<num>` (`z=-50 below z_floor=10` → `z=<num> below z_floor=<num>`) |
| `param_bucket` | The motion/vision fields of `input_params` | Order-of-magnitude buckets (see below) |

**param_bucket normalization** (only what can be decided from within the trace — env/config are never read):

- `x`/`y`/`z`/`r`: sign (`neg`/`pos`/`zero`) + magnitude (`abs<1` / `1-10` / `10-100` / `>=100`)
- `q`: length + whether any value is non-finite
- `object_name`/`target`/`prompt`: trim+lower; over 40 characters, the first 8 hex digits of `sha256` (stable across processes, unlike builtin `hash()`)
- A missing field never enters `param_bucket`; a field present but `None` is recorded as `<none>`, and non-finite numbers as `<nan>`/`<inf>`

Effect: SafetyRail rejections at `z=-50`, `z=-99` and `z=-20` with the same reason cluster into one group.

### 3.6 Patch-proposal template: `SkillPatchProposal`

Each cluster produces one proposal, with the template chosen by failure pattern:

| Pattern | Suggested direction |
|---------|----------|
| SafetyRail/reject, reason contains z/floor/below | Add a `z ≥ env.z_min_safe` pre-check to SKILL.md |
| Same, contains x/y/out of bounds | Add workspace XY bound constraints |
| Same, contains joint/limit/q | Add `q` length/range validation for `move_joint` |
| A vision tool (`analyze_scene`/`get_grasp_info_simple`/`pixel_to_base_xyz`) failed | Add a visual confirmation step, or disambiguate prompt/target |
| Fallback | Re-examine reason_norm; add a guard/retry/parameter constraint |

**Global post-process**: whenever a `RecoveryRail/recover` event appears among any cluster's examples, append "suggest adding to '## Failure handling': after an action fails, re-run `get_observation` to confirm the end effector is empty and the pose is right before continuing".

`target_skill` is always `<unresolved>` in the first version — traces do not record the active skill name, and a tool_name→skill mapping is unreliable (`goto_xyzr` is shared by many skills); a wrong match is worse than a blank. The skill is decided during human review.

### 3.7 Report shape

**`failure_report.md`**:

```markdown
# Trace Failure Report

- traces analyzed: 3
- failed steps clustered: 3
- clusters: 1

## Cluster 1 — goto_xyzr / SafetyRail / reject

- count: **3**
- affected conversations: ['c0', 'c1', 'c2']
- reason (normalised): `z=<num> below z_floor=<num>`
- param bucket: `x=pos/>=100, y=zero/abs<1, z=neg/10-100, r=zero/abs<1`

- **t0.json:step 1** — goto_xyzr
  - error: `SafetyRail: z=-50 below z_floor=10`
  - rail: SafetyRail/reject {'tool_name': 'goto_xyzr', 'reason': '...'}
```

**`skill_patch_proposals.md`**:

```markdown
## Proposal 1 — target: `<unresolved>`

- confidence: **medium**
- summary: 3 SafetyRail/reject failures (z=<num> below z_floor=<num>); decide the skill in review, then edit SKILL.md.

### Proposed diff (human review required)

```
In the relevant SKILL.md, under '## Parameter conventions' or '## Standard Workflow', add:
when calling `goto_xyzr`, `z` must be ≥ `env.z_min_safe`, otherwise SafetyRail rejects it.
Suggest a pre-check, or raising z and retrying after a failure.
```

- evidence signatures: `goto_xyzr/SafetyRail/reject`
- example: `t0.json:step 1` — goto_xyzr: SafetyRail: z=-50 below z_floor=10

### Risks
- The suggestion rests on clustered evidence and has not been validated on real hardware.
- target_skill is not resolved automatically; human review must confirm the target SKILL.md.
```

`confidence`: count≥5 high, 3-4 medium, 2 low.

---

## 4. Typical workflows

### 4.1 Development: use online and offline modes together

```yaml
agent:
  modules:
    tracing:
      enabled: true
    diagnosis:
      enabled: true
  execution:
    mode: stepagent
```

Run the task a few times (`--mock` or on hardware); traces land in `<workspace>/traces/`. Then:

```bash
python scripts/analyze_traces.py \
  --trace-dir ~/.jiuwensymbiosis/piper_workspace/traces \
  --min-cluster-size 2
```

Read `failure_report.md` for recurring failure patterns, take the suggestions from `skill_patch_proposals.md`, and edit SKILL.md after review.

### 4.2 Production: use online diagnosis only

```yaml
agent:
  modules:
    tracing:
      enabled: true
    diagnosis:
      enabled: true
  execution:
    mode: stepagent
```

The LLM automatically receives a diagnosis message on failure and self-corrects within the run. Traces still persist, for offline analysis later.

### 4.3 Post-incident review: run offline analysis only

You already have trace JSON (whether or not online mode was enabled — `modules.tracing.enabled=true` is enough), so simply:

```bash
python scripts/analyze_traces.py --trace-dir <trace directory>
```

---

## 5. Use it as a library (offline)

If you would rather not go through the CLI and want to call it from your own script:

```python
from pathlib import Path
from jiuwensymbiosis.trace_feedback import (
    load_trace_corpus,
    extract_failure_evidence,
    cluster_failures,
    propose_skill_patches,
)
from jiuwensymbiosis.trace_feedback.report import (
    render_failure_report,
    render_clusters_json,
    render_patch_proposals,
)

paths = sorted(Path("~/.jiuwensymbiosis/piper_workspace/traces").expanduser().glob("*.json"))
corpus = load_trace_corpus(paths)                       # load (bad JSON is skipped, never raised)
evidence = extract_failure_evidence(corpus, context_steps=2)  # extract failed steps
clusters = cluster_failures(evidence, min_size=2)       # cluster
proposals = propose_skill_patches(clusters)             # generate suggestions

print(render_failure_report(clusters, corpus=corpus))
print(render_patch_proposals(proposals))
```

---

## 6. Explicit boundaries

- Does **not** change the TraceRail / TraceEntry schema.
- Does **not** hard-code a repair strategy inside DiagnosisRail — it only enriches the evidence the LLM can see.
- Does **not** write back to production SKILL.md automatically — offline only writes reports.
- Does **not** read env/config for semantic bucketing (z floor / workspace bounds / joint limits) — the first offline version only does order-of-magnitude buckets decidable from within the trace.
- Does **not** parse SKILL.md frontmatter — `target_skill` stays `<unresolved>`; skill matching is left for later.
- Does **not** bring in an LLM/VLM — the first offline version is purely deterministic.
- Does **not** add a dependency — YAML parsing and Markdown string building both use the stdlib.

---

## 7. Related files

| File | Role |
|------|------|
| `jiuwensymbiosis/rails/diagnosis.py` | Online DiagnosisRail |
| `jiuwensymbiosis/trace_feedback/analysis.py` | Offline load/extract/signature/cluster |
| `jiuwensymbiosis/trace_feedback/report.py` | Offline json/markdown rendering |
| `jiuwensymbiosis/trace_feedback/patches.py` | Offline SkillPatchProposal |
| `scripts/analyze_traces.py` | Offline CLI |
| [Tracing reference](../reference/tracing.md) | Collection-layer (TraceRail) manual |
| [Trace Feedback Loop design](../../../design/trace-feedback-loop.md) | Design rationale |
