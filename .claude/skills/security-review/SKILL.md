---
name: security-review
description: Review JiuwenSymbiosis changes for physical safety and execution trust boundaries, including motion dispatch, recovery, generated Python, subprocesses, credentials, and persisted data. Use for an explicit security or safety review of a defined change.
disable-model-invocation: true
---

# Security and Safety Review

This skill is manually invoked. Routine development still follows
[security rules](../../rules/security.md). Review the affected behavior and
its callers; select applicable categories rather than filling every category
with a default PASS.

A review produces findings and verification evidence. It does not authorize
hardware motion, torque release, external publication, or changes to credentials.
Complete the local review with available evidence and identify any live
validation still needed.

## Establish scope and evidence

Record the reviewed diff/range, affected entry points, relevant configuration,
and whether each conclusion comes from code, an executed test, or an assumption.
Read `AGENTS.md` and the implementations on the actual changed paths.

Trace inputs from their source through validation to their side effect. For
each relevant path record the policy owner, recovery owner, and remaining
trust assumptions. If configuration disables a rail or an envelope is absent,
report that condition explicitly.

## Motion entry points

| Path | Review focus |
|---|---|
| Agent tools and `robot_control` | Rail attachment in `jiuwensymbiosis/agent/builder.py`, wrapped argument handling in `jiuwensymbiosis/rails/safety.py`, and rejection before dispatch |
| Fastagent | Executor supplied to `run_sequence`; `jiuwensymbiosis/agent/fast/ability_exec.py` routes through callbacks, while `direct_executor` in `jiuwensymbiosis/agent/fast/runner.py` does not |
| Servo | Per-target validation and cancellation in `jiuwensymbiosis/agent/fast/realtime/binding.py`, plus the affected driver's hardware limits |
| Generated Python | `jiuwensymbiosis/tools/inproc_code.py`, globals supplied by the session, and motion reachable inside the code |
| GUI, calibration, bring-up and manual scripts | Their actual validation, authorization context, cleanup, and driver checks; do not assume they traverse agent callbacks |

A direct `api.goto_xyzr(...)` call is not evidence of SafetyRail enforcement.
`@implements` attaches action metadata, not a safety wrapper. Even a tool
object can be called outside the agent lifecycle; follow the complete path.

For affected motion, check applicable capability gates, finite values,
units/frames, bounds and per-command limits. A missing Env envelope means
the corresponding range is unchecked. Inspect configuration and actual
driver enforcement instead of substituting invented limits.

## Failure, recovery, and calibration

- Trace raised exceptions and business-level failure results separately.
  Preserve failure codes and ensure recovery is neither skipped nor repeated
  after an earlier layer handled it.
- Check `jiuwensymbiosis/rails/recovery.py` and the affected body:
  held payload, safe retreat, end-effector release, failure during home,
  partial execution, and cancellation must retain their intended semantics.
  Do not simplify recovery to unconditional home followed by release.
- For hand guiding, verify target resynchronization before torque restoration
  and preservation of `include_end_effector` semantics. Recovery failures
  must reach the operator through the existing error path.
- For calibration changes, verify acceptance gates, adapter reload validation,
  and that REVIEW/candidate reports remain unavailable to runtime loading.
  Check failure paths for accidental formal publication or overwriting a
  valid calibration.

## Executable inputs, processes, and networking

The in-process executor is intentionally unsandboxed and receives live objects.
Review which inputs can influence executable code and which objects are exposed.
Tool output, retrieved text, and perception labels can carry untrusted content;
do not assume prompts alone enforce permissions or motion restrictions.
Distinguish an observed bypass from an untested trust assumption.

Inspect subprocess argument construction, executable provenance, inherited
environment, startup failure and shutdown. An argument list prevents shell
interpretation; it does not make an arbitrary executable or argument safe.
Sidecar lifecycle should remain owned by the session.

For network changes, examine bind address, caller reachability, transport,
and credential handling. Account for the actual deployment; do not conceal
plaintext HTTP by relocating a literal to satisfy a scanner.

## Credentials, dependencies, and persisted data

Follow the security rules for credentials and dependency audit requirements.
When dependencies change, record the environment, audit command, result, and
unavailable checks; do not report a clean scan that was not executed.

Inspect logs, errors, tool outputs, observation extras, frame paths and saved
frames for data newly persisted or exposed. Summarization and central logging
do not provide automatic redaction.

## Verification and findings

Select tests from [the change-validation map](../../references/change-validation.md).
Policy unit tests, executor tests, and static path inspection support different
claims. Identify missing wiring coverage instead of treating their names as
proof of end-to-end enforcement.

Report each finding with severity, affected path/configuration, evidence,
failure consequence, and the smallest useful fix or verification step.
For reviewed categories use `verified`, `issue found`, `not checked`, or
`not applicable`, with evidence or a reason. Separate code inspection from
test results and name any remaining hardware-dependent uncertainty.

Changes to rails, hardware drivers, and serving behavior retain the requirement
for a dedicated second reviewer before merge. Finish this report first and
identify that review need in the handoff; do not contact another person or
claim their approval as part of this skill.
