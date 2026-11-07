# Known Limitations

## Data and prediction target

- The application uses a retrospective eICU demo extract. Data access and
  licensing must be handled separately; raw and processed patient-level files
  are local and not committed.
- The prediction label is expired ICU discharge within 0-24 hours of an hourly
  prediction point. It is not a general clinical deterioration target.
- The demo results do not establish clinical utility, patient-care safety,
  clinical validation, or production readiness.
- The risk score and fixed fusion thresholds have not been independently
  validated for clinical calibration or workflow use.

## Runtime and model artifacts

- The backend reads a precomputed CSV; it does not run inference per request.
- `ml/models/` contains no production inference artifacts. Experimental SpO2
  models are isolated and are not used by the app.
- The experimental SpO2 alert replay omitted the anomaly signal and is not a
  complete reproduction of deployed saved-score fusion.
- Rebuilding results depends on local processed inputs and compatible Python
  packages; the data are not bundled in the repository.

## Alerts and notifications

- Saved hourly alert priorities and explanations are rule/model outputs, not
  clinician-adjudicated diagnoses.
- The six-hour cooldown script is offline and is not used by the API. Its
  existing output marks 472 rows as `SUPPRESS`, including 165 repeated
  critical rows. New severe events and critical escalations were not
  suppressed, but repeated unchanged critical rows can be. Do not present
  this batch output as a live notification system or operational suppression
  policy.
- No notification delivery service, escalation workflow, or clinical
  monitoring/ownership process is implemented.

## Hardware

- Physical ESP32 operation and calibration were not verified.
- The backend currently defaults to COM9; the device is optional for the
  software demonstration.
- Missing, invalid, stale, and disconnected sensor data are unavailable.
- The serial telemetry protocol does not provide calibrated remaining bag
  volume or flow rate.
- Historical fluid context is not live sensor telemetry.

## Security and operations

- The full npm audit reports 10 moderate development-tooling package records
  from a transitive `minimatch` advisory. The production-only npm audit is
  clean. See [DEPENDENCY_SECURITY_REVIEW.md](./DEPENDENCY_SECURITY_REVIEW.md).
- The development API has no authentication and is not configured for public
  deployment.
- There is no external validation, privacy/security deployment review, model
  monitoring, rollback system, or production operations plan.
