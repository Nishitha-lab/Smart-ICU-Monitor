import unittest

import pandas as pd
from fastapi import HTTPException

import main


def patient_row(
    stay_id: int,
    hour: int,
    priority: str,
    risk: float,
) -> dict:
    return {
        "patientunitstayid": stay_id,
        "hour": hour,
        "final_alert_priority": priority,
        "risk_probability": risk,
        "alert_category": "single-signal",
        "risk_band": "high",
        "anomaly_score": 0.1,
        "ml_anomaly": False,
        "clinical_rule_severity_v2": "normal",
        "clinical_rule_explanation_v2": "",
        "final_alert_explanation": "",
    }


class PatientListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_df = main.df
        main.df = pd.DataFrame(
            [
                patient_row(101, 1, "critical", 0.95),
                patient_row(101, 2, "normal", 0.05),
                patient_row(202, 1, "high", 0.75),
                patient_row(202, 3, "critical", 0.80),
            ]
        )
        self.addCleanup(self.restore_dataset)

    def restore_dataset(self) -> None:
        main.df = self.previous_df

    def test_priority_filter_uses_latest_row_for_each_stay(self) -> None:
        result = main.get_patients(priority="critical", limit=10)

        self.assertEqual(result["count"], 1)
        self.assertEqual(
            result["patients"][0]["patientunitstayid"],
            202,
        )
        self.assertEqual(result["patients"][0]["hour"], 3.0)
        self.assertEqual(result["patients"][0]["priority"], "critical")

    def test_unfiltered_list_uses_latest_priority(self) -> None:
        result = main.get_patients(priority=None, limit=10)

        patient_priorities = {
            patient["patientunitstayid"]: patient["priority"]
            for patient in result["patients"]
        }
        self.assertEqual(patient_priorities, {101: "normal", 202: "critical"})

    def test_invalid_priority_returns_bad_request(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            main.get_patients(priority="urgent", limit=10)

        self.assertEqual(raised.exception.status_code, 400)

    def test_unloaded_dataset_returns_service_unavailable(self) -> None:
        main.df = None

        with self.assertRaises(HTTPException) as raised:
            main.get_patients(priority=None, limit=10)

        self.assertEqual(raised.exception.status_code, 503)

    def test_empty_dataset_returns_empty_patient_list_and_zero_summary(self) -> None:
        main.df = pd.DataFrame(
            columns=[
                "patientunitstayid",
                "hour",
                "final_alert_priority",
                "risk_probability",
                "alert_category",
                "ml_anomaly",
                "clinical_warning",
                "clinical_severe",
                "risk_high",
                "risk_very_high",
                "risk_extreme",
            ]
        )

        self.assertEqual(
            main.get_patients(priority=None, limit=10),
            {"count": 0, "patients": []},
        )
        self.assertEqual(main.get_summary()["total_rows"], 0)
        self.assertEqual(main.get_summary()["icu_stays"], 0)

    def test_unknown_stay_returns_not_found_for_history_and_latest(self) -> None:
        for endpoint in (main.get_patient_history, main.get_patient_latest):
            with self.subTest(endpoint=endpoint.__name__):
                with self.assertRaises(HTTPException) as raised:
                    endpoint(patientunitstayid=-999)

                self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
