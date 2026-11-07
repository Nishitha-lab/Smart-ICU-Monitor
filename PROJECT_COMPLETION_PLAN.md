# Smart ICU Project Completion Plan

## Purpose and guardrails

This plan records the Phase 1 repository and runtime audit and the implementation
approved by the user. No application, model, alert, hardware, or data files
were changed during the audit itself. The approved fixes are now complete.

The project is a research prototype using historical eICU demo data and a
separate, optional ESP32 telemetry feed. It is not a medical device and is not
validated for patient-care use. In particular, the modeled target is an
expired ICU discharge within the next 24 hours, not general clinical
deterioration. The retrospective experiments do not establish clinical
utility, safety, or production readiness. Experimental SpO2 work remains
isolated and does not authorize production preprocessing, model, alert, or
application changes.

## Phase 1 audit summary

- The working tree was clean on branch `master`, synchronized with
  `origin/master` at `a71dc99`.
- FastAPI started and loaded `data/processed/smart_icu_final_alerts.csv`
  (139,432 patient-hour rows and 2,419 ICU stays). Health, summary, patient
  list, patient history, and hardware telemetry endpoints responded. Invalid
  stay IDs, priority values, and limits returned appropriate client errors.
- The React dashboard loaded from that API, and a patient detail view rendered.
  A browser reload had no console errors. The hardware endpoint reported
  disconnected because the configured serial port was unavailable.
- At the audit baseline, `npm run build` passed. `npm run lint` failed on a React hook rule in
  `frontend/src/App.tsx:500` and reported a dependency warning at line 501.
  There is no frontend test script; `npm test -- --run` cannot run. Backend
  unittest discovery found no backend tests.
- At the audit baseline, the API's critical-priority patient filter selected a stay's latest
  *matching-priority* row rather than first selecting the stay's latest row.
  An independent summary of the processed CSV found 768 stays with at least
  one historical critical row, but only 110 with a critical latest row; 658
  have an older critical row and a newer noncritical row. Thus a filtered
  patient list can present an out-of-date alert state.
- At the audit baseline, the UI described the data as “REAL-TIME” and labeled the historical patient's
  final row “Current hour.” It does not disclose beside the risk value that
  the target is an expired ICU discharge within 24 hours.
- At the audit baseline, the patient detail view showed zero/normal IV values while hardware was
  disconnected. The UI defaults missing fluid to zero, assumes an initial
  weight of 1000 g and a flow of zero, and maps missing device status to
  normal. The serial reader can also report connected after opening a port
  before receiving a valid telemetry frame.
- At the audit baseline, in `frontend/src/App.css`, the `max-width: 760px` media query opened at line
  829 is not closed until the end of the file (line 1298). Later base styles
  for recent context, signal convergence, history, and IV monitoring are
  therefore scoped to narrow viewports instead of all viewports; some later
  rules also override earlier mobile rules.
- The backend currently serves precomputed CSV predictions; it does not load
  models for API inference. `ml/predict.py` expects production artifacts under
  `ml/models/`, but those artifacts are absent from this checkout. The
  experimental bundles are not production inference artifacts.
- A local `Smart-ICU.zip` archive is about 602 MB and ignored by Git. Raw and
  processed datasets are local prerequisites rather than tracked source.
  These files were not read row-by-row, modified, or staged.

## Approved implementation results

- The patient list now selects each stay's latest row before applying a
  priority filter. An independent comparison against the local processed CSV
  confirmed the API's 110 critical stays exactly match the latest-row
  population.
- The UI labels eICU records as historical and identifies the model target as
  expired ICU discharge within 24 hours. It no longer presents the historical
  dataset as real-time or uses “Current hour.”
- Missing, invalid, disconnected, and stale hardware readings are shown as
  unavailable. The UI no longer invents a zero reading, normal status, initial
  weight, remaining volume, or flow rate. The serial reader waits for a valid
  frame before reporting a connected telemetry stream.
- Shared stylesheet rules now apply at all viewport widths. Browser checks at
  320, 390, 768, 1024, and 1440 px found no horizontal overflow; fluid-monitor
  and recent-context layouts use the expected responsive columns.
- Dashboard requests cancel on filter changes, patient detail requests are
  cancelled when selection changes or closes, and the detail overlay now has
  dialog semantics, Escape handling, focus trapping, and focus restoration.
- Added backend `unittest` cases and frontend Vitest/Testing Library coverage
  for latest-priority selection, API errors, serial waiting/stale/fresh states,
  malformed telemetry frames, historical target labels, telemetry validity,
  priority requests, request races, and dialog keyboard behavior.
- Updated the root and frontend READMEs with Windows setup, test commands,
  dataset prerequisites, the precomputed-inference boundary, and research
  limitations. The API root version now matches the FastAPI metadata version.
- No production preprocessing, trained models, alert rules, raw datasets, or
  experimental results were changed. Live model inference, hardware
  calibration, clinical validation, and deployment remain out of scope.

## Prioritized work

Priority definitions: **P0** is a release-blocking safety or data-integrity
issue; **P1** is an urgent correctness or misleading-output issue; **P2** is
important reliability, test, or usability work; **P3** is lower-risk
documentation and polish. No P0 issue was identified in the local startup and
dashboard exercise. That does not constitute a safety review or production
readiness assessment.

### P1 — Correct the priority-filtered patient list

- **Location:** `backend/main.py`, `/api/patients` filtering and latest-row
  selection (around lines 285–347).
- **Root cause:** Priority filtering happens before grouping to the latest
  hour, so a stay with an older critical row can appear as currently critical.
- **Fix:** Select the latest row per stay first, then apply the requested
  priority filter and ordering. Preserve the existing unfiltered endpoint
  semantics and response shape.
- **Dependencies:** None.
- **Effort:** 2–4 hours.
- **Verification:** Add endpoint tests with a stay whose old and latest rows
  have different priorities; verify all priority filters against a small
  fixture and independently compare filtered API output to the latest-row
  population from the processed CSV.
- **Status:** Completed; backend regression tests pass and all 110 API results
  match an independent latest-row reconstruction.

### P1 — Clearly distinguish historical data and name the prediction target

- **Location:** `frontend/src/App.tsx` (status, overview, and patient detail
  labels around lines 554, 580, and 891); `ml/prepare_deterioration_labels.py`
  (target definition).
- **Root cause:** Historical hourly records are labeled “REAL-TIME” and
  “Current hour,” while “24h deterioration risk” does not identify the actual
  expired-ICU-discharge target.
- **Fix:** Label historical records and their timestamps as historical; reserve
  “live” wording for fresh hardware telemetry. Describe the model output as
  the retrospective probability of expired ICU discharge within 24 hours.
  Keep the research and non-clinical-use limitations visible.
- **Dependencies:** None.
- **Effort:** 4–6 hours.
- **Verification:** Browser-check dashboard and details for unambiguous
  historical/live separation and the exact target wording; verify no text
  implies general deterioration prediction, clinical validation, or
  patient-care readiness.
- **Status:** Completed; verified in browser and frontend tests.

### P1 — Represent absent or invalid hardware readings as unavailable

- **Location:** `frontend/src/App.tsx` (telemetry mapping around lines 507–520
  and IV display); `backend/services/serial_service.py` (connection and
  telemetry validity handling).
- **Root cause:** Missing fluid becomes `0`, initial weight is fixed at
  `1000 g`, flow is fixed at `0`, and absent device status becomes `normal`.
  Opening a serial port can mark it connected before a valid frame arrives.
- **Fix:** Keep unavailable values unavailable; do not infer bag level, flow,
  or normal status without corresponding valid measurements and calibration.
  Distinguish an open serial port from a fresh, valid telemetry feed, and
  present stale/invalid values as unavailable.
- **Dependencies:** Confirm which quantities the device actually measures
  and the accepted freshness/validity rules; hardware was not connected during
  this audit.
- **Effort:** 1–2 days, including protocol confirmation.
- **Verification:** Test disconnected, port-open/no-frame, malformed,
  stale, invalid, and fresh-valid telemetry. Confirm the UI never displays
  fabricated zero/normal values for missing data.
- **Status:** Completed for disconnected, open/no-frame, stale, invalid, and
  fresh-valid inputs using unit/UI tests. Physical device behavior remains
  unverified because no ESP32 was connected.

### P2 — Repair stylesheet scope and verify responsive layouts

- **Location:** `frontend/src/App.css`, `@media (max-width: 760px)` opened
  around line 829 and the closing brace at line 1298.
- **Root cause:** Base styles appended after the mobile breakpoint are
  accidentally inside that breakpoint, and some narrow-screen overrides
  precede those later base declarations.
- **Fix:** Close the media query at the intended boundary, move shared styles
  outside it, and keep breakpoint-specific overrides in their proper media
  blocks.
- **Dependencies:** None.
- **Effort:** 4–8 hours.
- **Verification:** Inspect dashboard and patient detail at 320, 390, 768,
  1024, and 1440 px widths; check for horizontal overflow and verify the
  intended grid and modal layout at each width.
- **Status:** Completed; browser checks found no horizontal overflow at the
  listed widths.

### P2 — Fix frontend effect/lint findings and protect patient selection

- **Location:** `frontend/src/App.tsx`, effects around lines 392–528.
- **Root cause:** Lint reports state updates inside an effect and a missing
  dependency; overlapping patient-history requests can also allow an older
  response to replace the currently selected patient's details.
- **Fix:** Restructure the effect and its dependencies according to React
  lifecycle rules, and cancel or ignore responses that no longer belong to
  the active selection.
- **Dependencies:** None.
- **Effort:** 4–8 hours.
- **Verification:** `npm run lint`, production build, and a test that rapidly
  opens two patients and closes the detail view while requests are pending.
- **Status:** Completed; lint/build pass and stale-response regression test
  passes.

### P2 — Establish application-level automated tests

- **Location:** `backend/` and `frontend/`; frontend scripts in
  `frontend/package.json`.
- **Root cause:** No backend test suite was found, and the frontend has no test
  script. Existing experiment validation does not test application routes or
  UI behavior.
- **Fix:** Add focused tests for API response/error behavior and frontend
  filtering, detail loading, and telemetry states. Add a documented test
  command without replacing the existing build or lint commands.
- **Dependencies:** Implement the P1 endpoint and telemetry changes first so
  tests protect the corrected behavior.
- **Effort:** 1–2 days.
- **Verification:** Run the new targeted suites plus build and lint from a
  documented clean setup; tests must not require raw datasets or live hardware.
- **Status:** Completed; 7 backend tests and 4 frontend tests pass.

### P2 — Make the precomputed-data/inference boundary explicit

- **Location:** `backend/main.py`, `ml/predict.py`, `ml/models/`, and README.
- **Root cause:** The API serves precomputed predictions from a CSV, while
  `ml/predict.py` expects model files that are not present in the checkout.
  The current UI can be mistaken for active model inference.
- **Fix:** Document the current precomputed-data behavior and decide, as a
  separate approved milestone, whether live inference is in scope. Do not
  train, promote experimental models, or change production inference as part
  of the current audit.
- **Dependencies:** A separate user decision and model/data compatibility
  review before any inference implementation.
- **Effort:** 2–4 hours to document the current boundary; live inference is
  unestimated until scoped.
- **Verification:** Confirm routes and UI accurately describe the source and
  timestamp of predictions; any future inference path requires its own tests
  and explicit approval.
- **Status:** Completed for documentation and UI labels; active inference
  remains explicitly deferred.

### P3 — Improve accessibility and maintenance documentation

- **Location:** `frontend/src/App.tsx` detail overlay around lines 832–847;
  `frontend/README.md`; `backend/main.py` root version response.
- **Root cause:** The detail overlay has no dialog semantics or keyboard/focus
  management, the frontend README is still the Vite template, and the root
  endpoint reports version `0.2.0` while FastAPI metadata reports `0.3.0`.
- **Fix:** Add accessible dialog behavior, replace template frontend setup
  text with project-specific instructions, and use one source of truth for
  API version metadata.
- **Dependencies:** None; preserve the existing no-clinical-use limitations
  in all project documentation.
- **Effort:** 0.5–1 day.
- **Verification:** Keyboard and screen-reader-oriented dialog checks,
  documented setup on a clean environment with approved data prerequisites,
  and consistent version values across API endpoints.
- **Status:** Completed; dialog keyboard behavior is tested, setup is
  documented, and the root API reports version `0.3.0`.

## Recommended sequencing

1. Approved scoped P1–P3 corrections are complete and validated.
2. Before any future implementation, separately scope and approve model
   inference, clinical validation, production deployment,
   downstream ML retraining, alert changes, and hardware deployment as
   separate milestones requiring explicit scope and approval.

## Audit commands and observed results

| Check | Result |
|---|---|
| Backend startup and local API exercise | Passed; 139,432 rows, 2,419 ICU stays |
| Corrected critical-priority filter | Passed; API returned the same 110 latest-critical stays as an independent CSV reconstruction |
| Browser dashboard, detail, and responsive layouts | Passed; no console errors or horizontal overflow at 320, 390, 768, 1024, and 1440 px |
| Hardware telemetry | Correctly unavailable; physical ESP32 was not connected |
| Backend `unittest` suite | Passed; 8 tests |
| Frontend `npm test` | Passed; 7 tests |
| Frontend `npm run lint` | Passed |
| Frontend `npm run build` | Passed |
| Production dependency audit (`npm audit --omit=dev`) | 0 vulnerabilities |
| Full dependency audit (current) | 10 moderate development-tooling records from one `minimatch` advisory; 0 high; no forced downgrade |
| Working tree | Contains only the scoped implementation and documentation changes; no raw datasets or model artifacts changed |

Earlier test counts in this plan are historical verification snapshots.
`FINAL_TEST_REPORT.md` is the authoritative source for the final verification
results (10 backend, 5 experimental robustness, 7 frontend; 22 total). These
final counts must not be interpreted as evidence that every historical test
suite was rerun during the final verification.

## Final Phase 4-6 verification

- The running backend loaded 139,432 rows across 2,419 ICU stays. Health,
  summary, patient list, latest state, patient history, and hardware telemetry
  endpoints were exercised against the local processed CSV.
- A browser check selected an actual row from the critical queue and rendered
  its patient history, explanations, vital trends, and disconnected/unavailable
  ESP32 state. Responsive checks at 320, 390, 768, 1024, and 1440 px found no
  horizontal overflow.
- Reconstructing final alert priorities and explanations from the saved source
  fields produced zero mismatches across all 139,432 rows. No critical clinical
  severity was downgraded.
- Final tests: 10 backend tests, 5 experimental robustness tests, and 7
  frontend tests passed. Frontend lint and production build passed.
- The production-only npm audit reports 0 vulnerabilities. The full audit
  reports 10 moderate development-tooling package records, all propagated from
  one `minimatch` advisory; the available full remediation requires a breaking
  downgrade and was not forced.
- The offline batch notification file contains 472 `SUPPRESS` actions,
  including 165 repeated critical rows. It is not consumed by the API; the
  cooldown policy must not be presented as a deployed notification workflow.
- Current status is **DEMONSTRATION READY WITH LIMITATIONS**. This is not
  clinical validation, production readiness, or hardware verification.
