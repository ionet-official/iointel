# Linting and Formatting Guide for IOIntel

This guide provides step-by-step instructions for cleaning up lint and formatting issues in the codebase using Ruff and Git. Use these commands whenever you need to ensure your code passes CI and pre-commit checks.

---

## 1. Install Ruff and Pre-commit (if not already installed)

```sh
pip install ruff pre-commit
```

---

## 2. Run Ruff Lint and Auto-fix

Check for lint errors and auto-fix what you can:

```sh
ruff check . --fix
```

---

## 3. Run Ruff Format

Format the codebase according to Ruff's style rules:

```sh
ruff format .
```

---

## 4. Run All Pre-commit Hooks (Optional)

If you use pre-commit, you can run all hooks (including Ruff and others):

```sh
pre-commit run --all-files
```

---

## 5. Stage Only Tracked Files (Avoid Adding Untracked Files)

To stage only changes to files already tracked by git (and avoid adding new/untracked files):

```sh
git add -u
```

---

## 6. Commit and Push

```sh
git commit -m "style: apply ruff lint and formatting"
git push
```

---

## 7. Troubleshooting

- **Untracked files:**
  - Use `git status` to see which files are untracked.
  - Use `git add -u` to avoid staging them.
- **GHA/pre-commit fails due to formatting:**
  - Run `ruff format .` and commit the changes.
- **See what Ruff would change:**
  - Run `ruff check .` to see lint errors.
  - Run `ruff format . --diff` to see formatting changes without applying them.

---

## 8. Useful References
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pre-commit Documentation](https://pre-commit.com/)

---

**Tip:** You can always copy-paste these commands into your terminal to quickly clean up your codebase before pushing or opening a PR. 