# infra

Local development, CI and deploy configuration.

- `docker/` — shared base images.
- `compose/` — profile fragments; the root `docker-compose.yml` is the entrypoint. Profiles let a
  developer run a subset (`--profile m2`) instead of the whole agent, so a naive build does not
  rebuild everything.
- `ci/` — shared checks that CI and pre-commit both call. The **workflows themselves live in
  `.github/workflows/`**, because that is the only path GitHub Actions reads; keeping a second copy
  here would guarantee the two drift apart. What stays in this folder is
  `check_core_purity.sh`, which is invoked from the `deterministic-core` workflow, from
  `task lint` and from the pre-commit hook — one script, three callers, one definition of the
  learned/deterministic boundary.
- `k8s/` — optional; not required for the simulation-only scope.
- `env/` — `.env` templates. Real `.env` files are git-ignored.
