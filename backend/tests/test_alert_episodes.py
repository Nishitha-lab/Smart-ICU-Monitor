"""
Unit tests for the Clinical Alert Episode State Machine.
Verifies:
- Consecutive alerts within cooldown consolidate into a single episode.
- Intermittent alerts post-cooldown reset and trigger a distinct second episode.
- Temporal monitoring gaps >= reset_hours reset the active episode.
- State machine never resets and retriggers on the identical observation.
- Episode false alarm rate correctly uses total monitored patient-hours as denominator.
- Negative-stay false alarm rate and stay-level sensitivity calculations.
"""

import numpy as np
import pandas as pd
import pytest
from ml.optimize_ml_engine import simulate_alert_episodes_exact

def test_consecutive_alerts_consolidate_within_cooldown():
    """Consecutive hourly alerts during cooldown should count as 1 episode."""
    df = pd.DataFrame({
        "patientunitstayid": [101] * 10,
        "hour": list(range(10)),
        "deterioration_label_24h": [0] * 10,
    })
    # Alerts at hours 1, 2, 3, 4 within 6h cooldown
    probs = np.array([0.02, 0.90, 0.95, 0.88, 0.85, 0.01, 0.01, 0.01, 0.01, 0.01])
    res = simulate_alert_episodes_exact(df, probs, threshold=0.10, cooldown_hours=6, reset_hours=2)

    assert res["total_false_episodes"] == 1
    assert res["fp_hourly_observations"] == 4
    assert res["negative_stay_false_alarm_rate"] == 1.0

def test_intermittent_alerts_trigger_distinct_episodes():
    """Alerts separated by full cooldown and reset period should count as 2 episodes."""
    df = pd.DataFrame({
        "patientunitstayid": [102] * 15,
        "hour": list(range(15)),
        "deterioration_label_24h": [0] * 15,
    })
    # Episode 1 starts at hour 1 -> cooldown ends at hour 7
    # Hours 7, 8 are below threshold (reset_hours=2) -> resets to QUIET
    # Hour 10 has alert -> triggers Episode 2
    probs = np.zeros(15)
    probs[1] = 0.80
    probs[2] = 0.85
    probs[10] = 0.90

    res = simulate_alert_episodes_exact(df, probs, threshold=0.10, cooldown_hours=6, reset_hours=2)
    assert res["total_false_episodes"] == 2
    assert res["negative_stay_false_alarm_rate"] == 1.0

def test_monitoring_gap_resets_active_episode():
    """A temporal gap in monitoring exceeding reset_hours resets the episode."""
    # Patient monitored at hour 0, 1 (alert), then 10h gap, then hour 12 (alert)
    df = pd.DataFrame({
        "patientunitstayid": [103, 103, 103, 103],
        "hour": [0, 1, 12, 13],
        "deterioration_label_24h": [0, 0, 0, 0],
    })
    probs = np.array([0.01, 0.80, 0.85, 0.01])
    res = simulate_alert_episodes_exact(df, probs, threshold=0.10, cooldown_hours=6, reset_hours=2)

    # Gap from hour 1 to 12 is 11 hours (> 6h cooldown + 2h reset)
    # Should result in 2 distinct episodes
    assert res["total_false_episodes"] == 2

def test_no_same_observation_reset_and_retrigger():
    """State machine cannot reset to QUIET and trigger a new episode on the same observation."""
    df = pd.DataFrame({
        "patientunitstayid": [104] * 10,
        "hour": list(range(10)),
        "deterioration_label_24h": [0] * 10,
    })
    # Hour 0: Alert (Ep 1)
    # Hour 1-5: Below threshold during cooldown
    # Hour 6: Below threshold (below_count=1)
    # Hour 7: Below threshold (below_count=2 -> RESET occurs here)
    # Hour 8: Alert (Ep 2 triggers cleanly on observation 8, NOT observation 7)
    probs = np.array([0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.01])
    res = simulate_alert_episodes_exact(df, probs, threshold=0.10, cooldown_hours=6, reset_hours=2)
    assert res["total_false_episodes"] == 2

def test_stay_level_sensitivity_and_lead_time():
    """Stay-level sensitivity and lead time to final observation on positive stay."""
    df = pd.DataFrame({
        "patientunitstayid": [201] * 8,
        "hour": list(range(8)),
        # Final 4 hours are positive deterioration window (hours 4, 5, 6, 7)
        "deterioration_label_24h": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    # Alert first occurs at hour 5 (inside positive window)
    probs = np.array([0.01, 0.01, 0.01, 0.01, 0.02, 0.85, 0.90, 0.95])
    res = simulate_alert_episodes_exact(df, probs, threshold=0.10, cooldown_hours=6, reset_hours=2)

    assert res["total_positive_stays"] == 1
    assert res["positive_stays_detected"] == 1
    assert res["stay_level_sensitivity"] == 1.0
    # Lead time = final observation hour (7) - first qualifying alert hour (5) = 2.0 hours
    assert res["lead_time_median_hours"] == 2.0
    # No false alarm episodes in negative window
    assert res["total_false_episodes"] == 0

def test_episode_burden_denominator():
    """Verifies that episode FAR uses total monitored patient-hours as denominator."""
    df = pd.DataFrame({
        "patientunitstayid": [301] * 100,
        "hour": list(range(100)),
        "deterioration_label_24h": [0] * 100,
    })
    probs = np.zeros(100)
    # 2 isolated false episodes
    probs[5] = 0.90
    probs[50] = 0.90
    res = simulate_alert_episodes_exact(df, probs, threshold=0.10, cooldown_hours=6, reset_hours=2)

    assert res["total_monitored_hours"] == 100
    assert res["total_false_episodes"] == 2
    # 2 episodes / 100 patient hours = 2.00
    assert res["episode_false_alarms_per_100_hours"] == 2.0
