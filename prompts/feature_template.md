# Feature Request Template

When working on a new feature, follow this architecture:

1. **Verify Skills**: Check `skills/` to see if a workflow exists.
2. **Utilize Scripts**: Any new deterministic logic (validation, calculation, formatting) MUST be written as a Python/Bash script in `scripts/` and tested in `tests/scripts/`.
3. **Execute Hooks**: Ensure your local environment has `git config core.hooksPath .hooks`.
4. **Implement**: 
   - Follow instructions in `CLAUDE.md`.
   - Never write LLM prompts to handle math or logical validation. 
   - Write code. Let code do the logic.

## Goal
[Describe the goal of the feature]

## Acceptance Criteria
- [ ] Tests provided in `tests/`
- [ ] All `make lint` checks pass
- [ ] Logic shifted to deterministic code
- [ ] No secrets exposed (`scripts/security_scan.py` passes)
