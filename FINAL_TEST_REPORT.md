# Final Test Report

## Automated checks

| Check | Result | Exact outcome |
|---|---|---|
| Backend unittest suite | PASS | 10 tests, 0 failures |
| Experimental robustness unittest suite | PASS | 5 tests, 0 failures |
| Frontend Vitest suite | PASS | 7 tests, 0 failures |
| Frontend ESLint | PASS | `npm run lint`, exit 0 |
| Frontend production build/type-check | PASS | `npm run build`, TypeScript and Vite 8.3.6 build succeeded |
| Production dependency audit | PASS | `npm audit --omit=dev`: 0 vulnerabilities |
| Full dependency audit | PASS WITH FINDING | 10 moderate development-tool package records from one `minimatch` advisory; 0 high/critical |

**Total automated tests:** 22 passed, 0 failed.

## Coverage exercised

- Latest-row patient priority selection and invalid priority handling.
- Unloaded and empty dataset handling, zero summary, and unknown stay IDs.
- Serial telemetry startup, malformed frames, stale frames, and fresh valid
  frames.
- Frontend API failure state, selected-priority requests, request races,
  closing behavior, historical target labels, telemetry unavailable state,
  and dialog keyboard/focus behavior.
- SpO2 feature gap handling, validation-only threshold selection, alert
  priority retention, cooldown checks for new events/escalations, and episode
  gap behavior.

Tests use technical fixtures for software behavior; they are not clinical
observations or clinical validation.

## Runtime and data consistency checks

- FastAPI loaded 139,432 rows across 2,419 ICU stays.
- Health, summary, patient list, history, latest-state, and hardware telemetry
  routes returned the expected responses.
- Unknown patient stay returned HTTP 404; invalid priority returned HTTP 400.
- The browser loaded an actual critical queue row and rendered its historical
  details, trend charts, and disconnected ESP32 state.
- Responsive layout checks at 320, 390, 768, 1024, and 1440 px found no
  horizontal overflow.
- Independent reconstruction of saved final alert priority and explanation
  fields had zero mismatches across the complete 139,432-row file.
- No row with a critical clinical-rule severity had a lower saved final
  priority.

## Prior milestone reports

The repository's earlier experimental reports record a leakage-safe split
validator with 5 passing checks, paired model validation with 13 passing
checks, and 5 passing robustness tests. The three automated unit-test suites
above were rerun for this completion task; model training and the complete
paired cross-validation were not rerun.

## Limitations

No physical ESP32 was connected. No live upstream-outage browser test was
forced; the frontend API-failure behavior is covered by the component suite.
All runtime results are from local retrospective eICU demo data and do not
establish clinical utility, safety, or production readiness.
