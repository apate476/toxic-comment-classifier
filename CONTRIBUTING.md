# Contributing to toxic_comment_classifier

## Team Roles
| Member      | Role                        |
| ----------- | --------------------------- |
| Arya Patel  | Project Lead, CI/Testing    |
| Taha Patil  | GCP Training                |
| Bilal Qader | GCP Deployment, FastAPI     |
| Asad Khan   | Docker, CML                 |

## Branching Strategy
- All work is done on feature branches off `dev`
- Branch naming: `feature/<description>` or `fix/<description>`
- PRs must target `dev`, never `main`
- `main` is merged into only at phase completion

## CI/CD Requirements
All PRs must pass these checks before merging:
- **CI** (`ci.yml`): ruff, mypy, and pytest across Python 3.10, 3.11, 3.12
- **Code Quality** (`codecheck.yaml`): ruff and mypy
- **Docker Build** (`docker.yaml`): Docker image builds and passes smoke test

## Testing Requirements for PRs
- All existing tests must pass: `pytest tests/`
- New features must include tests
- Coverage must not drop below 80%: `pytest tests/ --cov=toxic_comment_classifier`
- Run pre-commit before pushing: `pre-commit run --all-files`

## Pre-commit Hooks
Install once after cloning:
```bash
pre-commit install
```

Hooks run automatically on `git commit`. To run manually:
```bash
pre-commit run --all-files
```

To skip in an emergency:
```bash
git commit -m "message" --no-verify
```

## Code Review Standards
Before approving a pull request, reviewers must verify:
- Code runs without errors
- All CI checks pass
- All tests pass (`pytest tests/`)
- Ruff and mypy pass cleanly (`ruff check .` and `mypy src/`)
- Functions and classes have docstrings
- No data files or secrets are committed
- PR targets `dev`, not `main`

## Deployment Process
- Docker image is built and tested automatically on every push via GitHub Actions
- Merges to `main` trigger the full CI pipeline
- GCP deployments are done manually following `DEPLOYMENT.md`

## Merge Conflict Resolution
- Conflicts must be resolved before requesting review
- Always pull the latest `dev` before starting a new branch
- When a conflict arises, coordinate with the team member who made the conflicting change
- After resolving, run tests to confirm nothing broke
