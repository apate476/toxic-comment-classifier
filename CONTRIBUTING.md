# Contributing to toxic_comment_classifier

## Team Roles

| Member      | Role                        |
| ----------- | --------------------------- |
| Arya Patel  | Project Lead, Documentation |
| Taha Patil  | Documentation, Modeling     |
| Bilal Qader | Model Training              |
| Asad Khan   | Data Handling               |

## Code Review Standards

Before approving a pull request, reviewers must verify:

- Code runs without errors
- All tests pass (`pytest tests/`)
- Ruff and mypy pass cleanly (`ruff check .` and `mypy src/`)
- Functions and classes have docstrings
- No data files or secrets are committed
- PR targets `dev`, not `main`

## Merge Conflict Resolution

- Conflicts must be resolved before requesting review
- Always pull the latest `dev` before starting a new branch
- When a conflict arises, coordinate with the team member who made the conflicting change
- After resolving, run tests to confirm nothing broke
