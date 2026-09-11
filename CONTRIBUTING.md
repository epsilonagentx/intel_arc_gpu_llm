  # Contributing

  Thanks for your interest! This is a small, opinionated stack for running a local
  LLM on an Intel Arc Pro B60 (vLLM on the `xe` GPU driver). Bug reports, doc fixes,
  and config improvements are all welcome.

  ## License

  By contributing, you agree that your contributions are licensed under this
  project's **GNU GPL v3** — see [LICENSE](LICENSE).

  ## Getting started

  - For anything non-trivial (bugs, new features), **open an issue first** so the
    approach is agreed before you invest time.
  - For small fixes (typos, doc tweaks), a direct pull request is fine.

  ## Branching & pull requests

  - `main` and `developer` are **protected** — no direct pushes.
  - Branch off **`developer`**:
    - `feature/<short-description>` — new work
    - `bug/<short-description>` — fixes
  - Open your PR against **`developer`**, not `main`. Releases are cut from
    `developer` → `main`.
  - Keep PRs focused: one logical change per PR.

  ## Commits

  - **Sign your commits** so they show as *Verified* on GitHub (GPG or SSH). See
    [GitHub's signing docs](https://docs.github.com/en/authentication/managing-commit-signature-verification).
  - Clear messages: a concise imperative summary line, a blank line, then a short
    body explaining the *why* when it isn't obvious.
  - **Never commit** secrets, credentials, private IPs/hostnames, or machine-specific
    paths. Use placeholders (`/home/<user>/...`, `192.168.x.x`).

  ## Style

  - Match the style of the file you're editing.
  - Docs are split by audience — put content where it belongs:
    `README.md` (operator how-to), `DEVELOPER.md` (the *why*),
    `INTEL_ARC_B60.md` (current state).
  - Shell scripts: keep them dependency-light.

  ## Before opening a PR

  - [ ] Branched off `developer`, targeting `developer`
  - [ ] Commits are signed (Verified)
  - [ ] No secrets, private IPs, or personal paths
  - [ ] Docs updated if behavior or config changed
