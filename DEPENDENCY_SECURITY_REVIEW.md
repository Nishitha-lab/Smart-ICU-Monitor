# Dependency Security Review

## Scope and result

This review covers the 14 development-dependency findings reported by the
earlier npm audit, using the current frontend lockfile. Compatible updates
were applied and the lockfile was installed with `npm ci`.

- Current full audit: **10 moderate package records, 0 high, 0 critical**.
- Production-only audit (`npm audit --omit=dev`): **0 vulnerabilities**.
- The 10 current records all trace to one advisory affecting `minimatch`, not
  10 distinct underlying vulnerabilities.
- No forced dependency downgrade was applied.

## Underlying advisory

| Advisory | Affected package/version | Installed | Patched | Reachability and impact |
|---|---|---:|---|---|
| [GHSA-g3hv-75wg-m249](https://github.com/advisories/GHSA-g3hv-75wg-m249) | `minimatch` 7.0.0 through 11.0.0 | 10.2.6 | 11.0.1 or later | Transitive development tooling only. The vulnerable glob-pattern processing is used by lint/type-analysis dependencies; it is not imported by the application runtime. Exploitation would require a hostile or pathological pattern/configuration passed to those tools. |

## Package records in the audit

| Package | Installed | Direct/transitive | Path/role |
|---|---:|---|---|
| `@eslint/config-array` | 0.23.5 | Transitive | ESLint configuration and glob matching |
| `@eslint/js` | 10.0.1 | Direct dev dependency | ESLint rules; audit path includes ESLint |
| `@typescript-eslint/eslint-plugin` | 8.72.1 | Transitive | TypeScript lint rules |
| `@typescript-eslint/parser` | 8.72.1 | Transitive | TypeScript lint parsing |
| `@typescript-eslint/type-utils` | 8.72.1 | Transitive | TypeScript lint utility chain |
| `@typescript-eslint/typescript-estree` | 8.72.1 | Transitive | TypeScript syntax parsing and glob dependency |
| `@typescript-eslint/utils` | 8.72.1 | Transitive | TypeScript lint utility chain |
| `eslint` | 10.14.0 | Direct dev dependency | Lint tooling and configuration matching |
| `minimatch` | 10.2.6 | Transitive | Underlying vulnerable glob matcher |
| `typescript-eslint` | 8.72.1 | Direct dev dependency | TypeScript ESLint tooling |

These packages are development dependencies. They do not ship in the runtime
frontend bundle and are not used by the FastAPI service.

## Remediation decision

`npm audit fix` applied compatible lockfile updates and removed the earlier
high-severity findings. The remaining remediation suggested by npm requires a
breaking downgrade (notably ESLint 10 to ESLint 9) or a dependency-graph change
outside the current supported ranges. That change could affect lint behavior
and compatibility, so it was not forced. No unrelated package manifest change
was made for the advisory.

The installed versions were synchronized to the lockfile with `npm ci`.
Frontend tests, lint, and production build all pass after installation.

## Residual risk

The moderate advisory remains relevant when trusted developers run lint or
type-analysis tools on untrusted glob configuration. Keep lint configuration
and patterns trusted, avoid running the tooling on hostile repositories, and
reassess when compatible patched dependency ranges become available. This
review does not constitute a complete audit of Python dependencies or an
application penetration test.
