---
description: LLM coding behavior guidelines — think before coding, prefer simplicity, make surgical changes, drive by verifiable goals.
language: chinese
paths: []
alwaysApply: true
---

# Coding Behavior Principles

Behavioral guidelines to reduce common LLM coding mistakes. They govern how
the agent thinks and acts, not what the code does. Merge with project-specific
instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial
tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State assumptions explicitly. If uncertain, ask rather than guess.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

In jiuwensymbiosis context: the framework has layered abstractions
(capability gating, mixin MRO, Card/Config split, safety rails). Do not
silently assume how they interact — ask when behavior is ambiguous.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

In jiuwensymbiosis context: when adding a new tool, rail, or adapter method,
prefer a minimal implementation first. New hardware only needs a YAML config
+ 6 adapter files — do not speculative-generalize the mixin hierarchy before
a second form factor actually arrives. "Would a senior engineer say this is
overcomplicated?" — if yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

In jiuwensymbiosis context: capability mixins and their `@robot_tool` methods
are tightly coupled across MRO — changes to one mixin may affect tool
emission for every adapter that mixes it in. Stay focused; every changed line
should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

| Instead of... | Transform to... |
|--------------|----------------|
| "Add validation" | "Write tests for invalid inputs, then make them pass" |
| "Fix the bug" | "Write a test that reproduces it, then make it pass" |
| "Refactor X" | "Ensure tests pass before and after" |

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

In jiuwensymbiosis context: use `pytest tests/unit_tests/` for fast deterministic
verification (no hardware/GPU). Use `python scripts/validate_adapter.py` and
`python scripts/smoke_test_adapter.py` to verify adapter compatibility. Use
`python examples/piper_pick_demo.py --mock` for an end-to-end smoke run
without real hardware or LLM. Return the verification result to the user.

---

**These principles are working if:** fewer unnecessary changes in diffs,
fewer rewrites due to overcomplication, and clarifying questions come before
implementation rather than after mistakes.
