from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_leakage_safe_experiment import gap_aware_spo2_features
from replay_experimental_alerts import alert_priority, extract_episodes, replay_notifications
from run_paired_spo2_models import tune_f1_threshold


class SpO2FeatureTests(unittest.TestCase):
    def test_change_and_rolling_window_do_not_cross_gaps_or_stays(self) -> None:
        frame = pd.DataFrame(
            {
                "patientunitstayid": [1, 1, 1, 1, 2, 2],
                "hour": [0, 1, 3, 4, 0, 1],
                "spo2": [90.0, 92.0, 94.0, 96.0, 80.0, 82.0],
            }
        )
        result = gap_aware_spo2_features(frame).sort_values(
            ["patientunitstayid", "hour"]
        )
        values = result.set_index(["patientunitstayid", "hour"])
        self.assertTrue(np.isnan(values.loc[(1, 0), "spo2_change_1h"]))
        self.assertEqual(values.loc[(1, 1), "spo2_change_1h"], 2.0)
        self.assertTrue(np.isnan(values.loc[(1, 3), "spo2_change_1h"]))
        self.assertTrue(np.isnan(values.loc[(1, 3), "spo2_rolling_mean_3h"]))
        self.assertEqual(values.loc[(1, 4), "spo2_rolling_mean_3h"], 95.0)
        self.assertTrue(np.isnan(values.loc[(2, 0), "spo2_change_1h"]))
        self.assertEqual(values.loc[(2, 1), "spo2_rolling_mean_3h"], 81.0)


class ModelThresholdTests(unittest.TestCase):
    def test_threshold_is_selected_from_supplied_validation_labels(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        threshold, validation_f1 = tune_f1_threshold(labels, scores)
        self.assertGreaterEqual(threshold, 0.6)
        self.assertLess(threshold, 0.9)
        self.assertEqual(validation_f1, 1.0)


class AlertReplayTests(unittest.TestCase):
    def test_priority_retains_critical_clinical_state_and_fixed_risk_cuts(self) -> None:
        clinical = pd.Series(
            {
                "clinical_severity_rank": 4,
                "clinical_warning_count_v2": 1,
                "risk_probability": 0.0,
            }
        )
        self.assertEqual(alert_priority(clinical), "critical")
        risk = clinical.copy()
        risk["clinical_severity_rank"] = 0
        risk["clinical_warning_count_v2"] = 0
        risk["risk_probability"] = 0.50
        self.assertEqual(alert_priority(risk), "high")

    def test_cooldown_does_not_suppress_new_event_or_critical_escalation(self) -> None:
        frame = pd.DataFrame(
            {
                "patientunitstayid": [1, 1, 1, 1, 1],
                "hour": [0, 1, 2, 3, 10],
                "final_alert_priority": [
                    "high",
                    "high",
                    "critical",
                    "normal",
                    "high",
                ],
                "clinical_rule_explanation_v2": ["reason A", "reason A", "reason A", "none", "reason A"],
            }
        )
        result = replay_notifications(frame)
        self.assertEqual(result["notification_action"].tolist(), [
            "EMIT", "SUPPRESS", "ESCALATION", "NO_ALERT", "NEW_EVENT"
        ])
        self.assertFalse(
            (
                result["notification_action"].eq("SUPPRESS")
                & (result["new_severe_event"] | result["critical_escalation"])
            ).any()
        )

    def test_episode_gap_rule_keeps_two_hour_gaps_and_splits_larger_gaps(self) -> None:
        frame = pd.DataFrame(
            {
                "patientunitstayid": [1, 1, 1, 1],
                "hour": [0, 1, 3, 6],
                "severe_alert": [True, True, True, True],
                "final_alert_priority": ["high", "high", "critical", "high"],
                "deterioration_label_24h": [0, 0, 1, 0],
                "notification_emitted": [True, False, True, True],
                "notification_action": ["EMIT", "SUPPRESS", "ESCALATION", "NEW_EVENT"],
            }
        )
        episodes = pd.DataFrame(
            extract_episodes(frame, "test_model", "baseline")
        )
        self.assertEqual(len(episodes), 2)
        first = episodes.sort_values("start_hour").iloc[0]
        self.assertEqual(first["alert_hours"], 3)
        self.assertEqual(first["peak_priority"], "critical")
        self.assertEqual(first["target_positive_hours"], 1)


if __name__ == "__main__":
    unittest.main()
