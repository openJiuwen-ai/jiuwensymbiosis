# Use the Trace Feedback Loop

> Category: How-to. The [Chinese source](../../zh/how-to/use-trace-feedback.md) is authoritative.

The Trace Feedback Loop turns recorded failures into online guidance for the next model turn and offline evidence for
human-reviewed skill improvements. Neither path automatically changes a SKILL.md file or dispatches a robot action.

## 1. Overview of the two feedback layers

| Layer | When it runs | Output | Primary purpose |
| --- | --- | --- | --- |
| Online `DiagnosisRail` | During an Agent run | A compact diagnosis in the next model turn | Avoid repeating the same failed action |
| Offline `analyze_traces` | After one or more runs | Failure clusters, reports, and patch proposals | Find recurring failure patterns |

Both layers consume Trace evidence. Online diagnosis is bounded context for immediate recovery; offline analysis is a
review workflow and never mutates a skill automatically.

## 2. Online mode: `DiagnosisRail`

### 2.1 What it does

After a failed tool result or exception, `DiagnosisRail` stages the current error, relevant recent history, and system
state. It injects that message before the next model call, after the tool result is already present in context, so
OpenAI-style tool-message ordering remains valid.

### 2.2 Enable it in configuration (recommended)

Diagnosis depends on tracing:

```yaml
agent:
  enable_tracing: true
  enable_diagnosis: true
  diagnosis_max_chars: 1500
  diagnosis_history_steps: 3
  diagnosis_history_kinds: ["reject", "recover"]
```

If diagnosis is enabled while tracing is off, the builder logs a warning and does not attach `DiagnosisRail`.

### 2.3 Enable it in code

```python
from jiuwensymbiosis.agent import RobotAgentConfig

config = RobotAgentConfig(
    enable_tracing=True,
    enable_diagnosis=True,
    diagnosis_max_chars=1500,
    diagnosis_history_steps=3,
)
```

Pass this configuration to `build_robot_agent()` in the same way as other Agent options.

### 2.4 Failure channels covered

The Rail covers both a tool returning a structured failed result and a tool raising an exception. Safety rejection,
recovery outcome, and matching recent calls are retained when Trace events make them available.

### 2.5 Fast-path behavior

When execution uses a fast path without model context, message injection is skipped. Trace recording still works, so
the failure remains available to offline analysis.

### 2.6 Diagnosis-message shape

```text
### Diagnosis: the previous step failed
current tool, error, and parameters

### Related history
recent calls to the same tool or steps with matching Rail event kinds

### System state
recovery result and current pose when available
```

When the message exceeds `diagnosis_max_chars`, related history is removed first. The current failure is retained as
long as possible. The message asks the model to change parameters or strategy instead of repeating the same call.

## 3. Offline mode: `analyze_traces`

### 3.1 What it does

The analyzer loads persisted Trace JSON, extracts failed steps with nearby evidence, groups stable failure signatures,
and writes machine-readable clusters plus Markdown reports and skill-patch proposals for human review.

### 3.2 Quick start

Analyze all JSON files in a trace directory:

```bash
# Analyze the complete Trace directory
python scripts/analyze_traces.py \
  --trace-dir ~/.jiuwensymbiosis/piper_workspace/traces \
  --out reports/trace_feedback/latest \
  --min-cluster-size 3
```

Analyze one file while debugging:

```bash
# Debug one Trace file
python scripts/analyze_traces.py \
  --trace path/to/run.json \
  --out reports/trace_feedback/single \
  --min-cluster-size 1
```

### 3.3 CLI options

| Option | Meaning |
| --- | --- |
| `--trace-dir DIR` | Read all trace JSON files directly under a directory |
| `--trace FILE` | Analyze one trace file |
| `--out DIR` | Output directory; default `reports/trace_feedback/latest` |
| `--min-cluster-size N` | Suppress failure groups smaller than N; default 3 |
| `--context-steps N` | Keep up to N neighboring steps on each side; default 2 |

### 3.4 Exit codes

The command returns zero after a successful analysis. It returns nonzero when input paths are missing or no trace is
found. A malformed trace is skipped with a warning instead of aborting the rest of the corpus.

### 3.5 Clustering rules: how `FailureSignature` is built

Every failed step becomes `FailureEvidence`, including parameters, error, Rail/log events, frame path, and nearby
steps. Its stable signature uses:

- effective tool name;
- failing SafetyRail name and event kind, when applicable;
- a normalized reason with concrete numbers replaced;
- parameter buckets for motion coordinates, joints, and target text.

Recovery is a remedy rather than the original failure cause. For example, a tool exception followed by a failed home
attempt remains a tool-failure cluster, with recovery retained as supporting evidence. Long prompts use deterministic
SHA-256 summaries so clustering is stable across processes.

### 3.6 Patch-proposal template: `SkillPatchProposal`

A proposal contains a summary, representative evidence, suggested rule text, confidence derived from occurrence count,
risks, validation steps, and `target_skill="<unresolved>"` until a human selects the correct skill.

Do not apply a proposal mechanically. Confirm a shared physical cause, check that the rule generalizes across supported
robots, and reproduce the failure with Mock or hardware integration tests.

### 3.7 Report shape

The output directory contains machine-readable clusters and Markdown reports. A typical excerpt is:

```markdown
# Trace Failure Report

## Cluster 1 — goto_xyzr / SafetyRail / reject

- normalized reason: z=<num> below z_floor=<num>
- count: 4
- affected conversations: 3

## Proposal 1 — target: `<unresolved>`

### Proposed diff (human review required)

Check `env.z_min_safe` before issuing cartesian motion.

### Risks

Validate TIP/FLANGE semantics for every supported body.
```

Only groups meeting `--min-cluster-size` appear in the report.

## 4. Typical workflows

### 4.1 Development: use online and offline modes together

Enable tracing and diagnosis. Let the model recover within a run, then analyze a batch of traces before changing skills
or defaults.

```yaml
# Task YAML
agent:
  enable_tracing: true
  enable_diagnosis: true
```

### 4.2 Production: use online diagnosis only

Enable online diagnosis only when its additional model-context content is acceptable. Keep frame saving and log capture
bounded, and do not enable automatic skill mutation.

### 4.3 Post-incident review: run offline analysis only

Analyze an immutable copy of the Trace corpus. Preserve original JSON and frame paths as evidence, and record the
hardware validation performed for every accepted rule change.

## 5. Use it as a library (offline)

```python
from pathlib import Path

from jiuwensymbiosis.trace_feedback import (
    cluster_failures,
    extract_failure_evidence,
    load_trace_corpus,
    propose_skill_patches,
)

paths = sorted(Path("traces").glob("*.json"))
corpus = load_trace_corpus(paths)
evidence = extract_failure_evidence(corpus, context_steps=2)
clusters = cluster_failures(evidence, min_size=3)
proposals = propose_skill_patches(clusters)
```

Report functions accept these already-built structures and do not reread or reanalyze source files.

## 6. Explicit boundaries

- Online diagnosis does not bypass SafetyRail or retry a command automatically.
- Offline analysis does not prove that correlated failures share one physical cause.
- Patch proposals are review artifacts, not executable patches.
- Trace limits and redaction remain the operator's responsibility in production.

## 7. Related files

- [Tracing reference](../reference/tracing.md): persisted schema, configuration, and replay.
- [Trace Feedback Loop design](../../../design/trace-feedback-loop.md): safety and layering decisions.
- [`scripts/analyze_traces.py`](../../../scripts/analyze_traces.py): command-line entry point.
- [`jiuwensymbiosis/trace_feedback/`](../../../jiuwensymbiosis/trace_feedback/): offline analysis library.
