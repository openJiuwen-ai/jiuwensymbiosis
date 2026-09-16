# Development Principles

- Start from the requested outcome and the relevant code, public exports,
  tests, and project constraints in `AGENTS.md`. State material assumptions.
  Ask when missing information changes the intended behavior or safe scope;
  resolve routine implementation choices from repository evidence.
- Match the effort to the change. Local fixes need a focused diff and relevant
  verification. Changes to public contracts, ownership, or dependency direction
  need an explicit design rationale; use
  [module-design](../skills/module-design/SKILL.md) when that reasoning is substantial.
- Reuse an existing owner before adding another abstraction. Explain what
  complexity a new module hides and which callers benefit. Keep hardware
  variation behind the project's existing capability and adapter contracts.
- Keep changes within scope. Remove code made obsolete by this change, preserve
  unrelated work, and report unrelated findings without expanding the task.
- Define observable success before implementation. Reproduce behavior bugs
  with a focused regression test where practical; verify refactors against
  the existing contract. Do not add tests that only mirror implementation.
- Choose checks using the
  [change-validation map](../references/change-validation.md). Report the actual
  command, outcome, skips, and remaining uncertainty. A static reading is not a
  test run, and an empty test selection is not a pass.
