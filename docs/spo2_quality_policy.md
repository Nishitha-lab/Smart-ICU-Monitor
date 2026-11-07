# SpO2 quality policy and raw-data cross-validation

## Scope

This policy applies to the research preprocessing of periodic eICU SpO2
(`vitalPeriodic.sao2`). It does not change the production pipeline, downstream
features, models, alerts, or application behavior. The audit uses the local
eICU Collaborative Research Database demo files and therefore describes that
demo extract, not the full eICU database or a clinical validation study.

Run the reproducible audit from the project root:

```powershell
python ml\audit_spo2_cross_validation.py
```

The script restricts periodic, lab, respiratory-charting, and aperiodic
observations to the corresponding patient-unit stay and its recorded ICU
discharge interval. Observation offsets are in minutes; periodic values are
bucketed into hours using integer division by 60, as in the existing
preprocessing pipeline. The raw lab labels identify `O2 Sat (%)` in percent
and `paO2` in mm Hg. Respiratory charting labels the direct saturation field
`SaO2`.

## Verified dataset results

The audit independently recomputed SpO2 hourly means from in-stay
`vitalPeriodic.csv.gz` observations and compared them with every patient-hour
in `hourly_vitals_labs_spo2_experiment.csv`.

| Measure | Verified result |
|---|---:|
| Patient-unit stays in `patient.csv.gz` | 2,520 |
| Stays with in-stay periodic observations | 2,367 |
| In-stay periodic observations | 1,628,390 |
| Patient-hours with periodic observations | 138,827 |
| Patient-hours in the experimental file | 139,568 |
| Experimental patient-hours without periodic observations | 741 |
| Non-missing periodic SpO2 observations | 1,436,591 |
| Exact-zero SpO2 observations | 1,554 |
| Nonzero readings at or below 50 | 408 across 109 stays |
| Nonzero readings outside 0–100 | 0 |
| Mixed-zero patient-hours | 240 |
| Zero-only patient-hours | 47 |
| Mixed-zero hours with corrected mean above 80 | 233 |
| Mixed-zero hours with corrected mean above 50 and at most 80 | 6 |
| Mixed-zero hours with corrected mean at or below 50 | 1 |

Across all periodic patient-hours, the zero-excluded hourly mean was at or
below 50 in 33 hours across 21 stays, at or below 70 in 195 hours across 82
stays, and at or below 80 in 494 hours across 189 stays. These counts match
the previously reported thresholds. The experiment's raw means, zero-excluded
means, measurement counts, and quality flags all matched the independent
recomputation for all 138,827 patient-hours with periodic observations; no
periodic patient-hour was missing from the experiment. The additional 741
experimental timeline rows had no periodic observations and were not part of
the raw periodic aggregation comparison.

## Low-reading evidence analysis

The audit examines every in-stay, nonzero periodic SpO2 reading at or below
50. For each reading, it searches the same stay for lab results, respiratory
charting SaO2, and noninvasive mean blood pressure within an inclusive
15-minute window on either side. It records PaO2 separately from direct
saturation measurements and treats blood pressure only as contextual data.

For episode-level summaries, consecutive low readings are grouped into
operational **low-reading clusters** when the gap between their offsets is at
most 15 minutes. A cluster is a reproducible grouping rule, not a clinically
adjudicated hypoxemia episode.

The demo data contained 408 low readings in 186 clusters across 109 stays.
Within the specified windows, the audit found 12 respiratory SaO2 matches,
3 lab O2 saturation matches, 4 PaO2 matches, and 390 noninvasive mean blood
pressure matches. After counting unique direct saturation records per
cluster:

- 3 clusters had a direct saturation at or below 50.
- 6 clusters had a direct saturation at or above 70.
- 177 clusters had no direct saturation reference in the window.
- No cluster had only an intermediate direct saturation (above 50 and below
  70).

The `<=50` and `>=70` labels are audit comparison bands, not validated
clinical cutoffs. PaO2 is not converted into an SpO2 estimate or counted as a
direct saturation confirmation. A nearby blood pressure is never treated as
confirmation of oxygen saturation.

The audit reproduced the previously investigated examples in the raw data:

- Stay 3157219: SpO2 6 at offset 121 and lab O2 Sat 38.7 at offset 134,
  supporting a low direct saturation.
- Stay 3067309: SpO2 48 and lab O2 Sat 86 at offset 345, a direct disagreement;
  PaO2 at that offset is retained as separate context.
- Stay 3204052: SpO2 21 at offset 1455 and respiratory SaO2 92 at 1451
  disagree, while SpO2 values 47 and 43 at offsets 1725–1730 have respiratory
  SaO2 47 at offset 1730.
- Stay 758327: its eight low-reading clusters had no direct saturation
  reference within the specified windows.
- Stay 1229669: nearby mean blood pressure is recorded as context only and
  does not directly validate SpO2.

Lack of a nearby reference is **unverified**, not evidence that the periodic
reading is either correct or erroneous. A nearby charted value can also have
its own measurement or documentation limitations.

## Adopted research quality policy

1. **Exclude exact zero from the candidate hourly mean.** Treat zero as a
   likely invalid monitor artifact for averaging, not as a saturation of 0%.
   Retain the original raw hourly mean so the effect of this exclusion remains
   auditable.
2. **Preserve low nonzero values.** Keep every nonzero value in the observed
   0–100 range, including values at or below 50. Do not remove, cap, winsorize,
   or impute these readings based only on their magnitude.
3. **Retain counts and explicit quality status.** Preserve the valid
   nonzero, exact-zero, and missing counts alongside raw and candidate means.
   `MIXED_ZERO` identifies a zero-contaminated hour whose candidate mean uses
   its nonzero values; `ZERO_ONLY` has no candidate mean; `MISSING_ONLY` has
   no recorded SpO2; and `VALID` means observed without exact zeros, not
   independently clinically validated.
4. **Represent zero-only hours as missing.** Do not substitute zero or carry
   forward a value when no nonzero SpO2 observation is available.
5. **Flag extreme low readings for review without suppressing them.** A
   nonzero SpO2 at or below 50 may be marked for data-quality/clinical review,
   but its disposition requires evidence and must not be inferred from MAP.
   The direct-reference categories in this audit are evidence summaries only;
   they are not automatic labels for model training or clinical decisions.
6. **Keep the evidence trail and scope visible.** Retain the raw inputs,
   time-window matches, and episode-cluster definition with any analysis.
   Reassess the policy if the source population, measurement process, or
   independent validation data changes.

The policy establishes a research preprocessing rule; it does not authorize
downstream retraining or any change to live alerts or patient-facing behavior.

## Audit outputs

The script writes these files under `data/processed/spo2_audit/`:

- `spo2_audit_summary.csv`: dataset-wide counts and cross-validation totals.
- `spo2_hourly_raw_validation.csv`: recomputed and experimental values and
  per-field match indicators for each patient-hour with periodic observations.
- `spo2_low_reading_evidence.csv`: one record per low periodic reading and
  nearby-reference counts.
- `spo2_reference_matches.csv`: row-level reference matches, time deltas,
  values, and whether evidence is direct or contextual.
- `spo2_low_reading_clusters.csv`: low-reading cluster summaries and direct
  saturation evidence classifications.
