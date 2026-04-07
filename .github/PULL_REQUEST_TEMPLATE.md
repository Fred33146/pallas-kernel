## Description

What does this PR do and why?

## Related Issue

Closes #

## Change Type

- [ ] `feat` — New feature
- [ ] `fix` — Bug fix
- [ ] `refactor` — Code refactoring
- [ ] `docs` — Documentation
- [ ] `ci` — CI/CD changes
- [ ] `test` — Tests
- [ ] `perf` — Performance improvement

## Checklist

- [ ] Code passes `uv run ruff check src/ tests/` and `uv run ruff format src/ tests/`
- [ ] New/modified public APIs have complete docstrings (tensor shapes, dimension semantics, business logic)
- [ ] Public functions have input assertions (`assert` or `assert_shape_or_none`)
- [ ] Tests added at the appropriate layer (`tests/ops/`, `tests/modules/`, `tests/layers/`, or `tests/ref/`)
- [ ] If `tops/cpu/` is modified, core developers have been notified and PR is labeled `cpu-ref`

## Test Results

```
Paste relevant test output here.
```
