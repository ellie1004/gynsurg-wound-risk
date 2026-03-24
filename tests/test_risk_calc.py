"""
Unit tests for the GynSurg Wound Complication Risk Calculator.

Tests the core risk calculation engine: standardization, log-odds computation,
logistic transformation, and feature encoding.

Run: python -m pytest tests/test_risk_calc.py -v
"""

import math
import sys
import os

# ---------------------------------------------------------------------------
# Replicate the calculation engine from app.py so tests are self-contained
# ---------------------------------------------------------------------------

LR_COEFFICIENTS = {
    'Age': -0.2955,
    'BMI': 0.0,
    'Hypertension': 0.0,
    'Diabetes': -0.1441,
    'Smoking': -0.2698,
    'Prior_abd_surgery': 0.5169,
    'Pulmonary_disease': 0.0,
    'SC_fat_SP': 0.0142,
    'SC_fat_below_umbilicus': -0.1912,
    'SC_fat_above_umbilicus': 0.0,
    'Preop_albumin': -0.4419,
    'Preop_hemoglobin': 0.8478,
    'Preop_glucose': -0.9009,
    'Dx_malignant': 0.7129,
    'Dx_benign_myoma': 0.0,
    'WT_low_midline': 0.3559,
    'WT_midline': 0.0,
}

POPULATION_STATS = {
    'Age': {'mean': 49.0, 'std': 12.8},
    'BMI': {'mean': 24.0, 'std': 3.6},
    'Hypertension': {'mean': 0.177, 'std': 0.383},
    'Diabetes': {'mean': 0.061, 'std': 0.239},
    'Smoking': {'mean': 0.039, 'std': 0.194},
    'Prior_abd_surgery': {'mean': 0.442, 'std': 0.498},
    'Pulmonary_disease': {'mean': 0.009, 'std': 0.093},
    'SC_fat_SP': {'mean': 2.4, 'std': 0.8},
    'SC_fat_below_umbilicus': {'mean': 2.4, 'std': 0.8},
    'SC_fat_above_umbilicus': {'mean': 2.6, 'std': 1.7},
    'Preop_albumin': {'mean': 4.1, 'std': 0.5},
    'Preop_hemoglobin': {'mean': 11.7, 'std': 1.7},
    'Preop_glucose': {'mean': 106.4, 'std': 36.1},
    'Dx_malignant': {'mean': 0.385, 'std': 0.488},
    'Dx_benign_myoma': {'mean': 0.264, 'std': 0.442},
    'WT_low_midline': {'mean': 0.476, 'std': 0.500},
    'WT_midline': {'mean': 0.208, 'std': 0.407},
}

BASELINE_LOG_ODDS = math.log(8 / 223)


def compute_risk(patient_values: dict) -> tuple[float, dict]:
    """Return (probability, {feature: contribution}) for a patient."""
    log_odds = 0.0
    contributions = {}
    for feat, raw_val in patient_values.items():
        stats = POPULATION_STATS[feat]
        z = (raw_val - stats['mean']) / stats['std'] if stats['std'] > 0 else 0
        coef = LR_COEFFICIENTS[feat]
        c = z * coef
        log_odds += c
        contributions[feat] = c
    total = BASELINE_LOG_ODDS + log_odds
    prob = 1 / (1 + math.exp(-total))
    return prob, contributions


def encode_diagnosis(diagnosis: str) -> tuple[int, int]:
    """Return (Dx_malignant, Dx_benign_myoma)."""
    if "Malignant" in diagnosis:
        return 1, 0
    elif "myoma" in diagnosis:
        return 0, 1
    else:
        return 0, 0


def encode_wound_type(wound_type: str) -> tuple[int, int]:
    """Return (WT_low_midline, WT_midline)."""
    if wound_type == "Low midline":
        return 1, 0
    elif wound_type == "Midline":
        return 0, 1
    else:
        return 0, 0


def make_patient(
    age=49.0, bmi=24.0, htn=0, dm=0, smoking=0, prior_surg=0,
    pul_disease=0, sc_sp=2.4, sc_below=2.4, sc_above=2.6,
    albumin=4.1, hemoglobin=11.7, glucose=106.4,
    dx_malig=0, dx_myoma=0, wt_low=0, wt_mid=0,
):
    """Build a patient_values dict with defaults = population means."""
    return {
        'Age': age, 'BMI': bmi,
        'Hypertension': htn, 'Diabetes': dm, 'Smoking': smoking,
        'Prior_abd_surgery': prior_surg, 'Pulmonary_disease': pul_disease,
        'SC_fat_SP': sc_sp, 'SC_fat_below_umbilicus': sc_below,
        'SC_fat_above_umbilicus': sc_above,
        'Preop_albumin': albumin, 'Preop_hemoglobin': hemoglobin,
        'Preop_glucose': glucose,
        'Dx_malignant': dx_malig, 'Dx_benign_myoma': dx_myoma,
        'WT_low_midline': wt_low, 'WT_midline': wt_mid,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_baseline_patient():
    """All values at exact population means -> risk ~3.5% (baseline prevalence).

    For binary variables, use the population prevalence as the value
    (e.g., Hypertension=0.177) so all z-scores are exactly 0.
    """
    patient = {feat: stats['mean'] for feat, stats in POPULATION_STATS.items()}
    prob, contributions = compute_risk(patient)
    # All z-scores should be 0 -> all contributions 0 -> prob = sigmoid(baseline)
    for feat, c in contributions.items():
        assert abs(c) < 1e-10, f"Contribution for {feat} should be 0, got {c}"
    assert abs(prob - 0.0347) < 0.005, f"Baseline risk should be ~3.5%, got {prob*100:.2f}%"


def test_all_zeros_input():
    """Minimum-like input should compute without error and return valid probability."""
    patient = make_patient(
        age=14, bmi=14.0, htn=0, dm=0, smoking=0, prior_surg=0,
        pul_disease=0, sc_sp=0.5, sc_below=0.2, sc_above=0.2,
        albumin=1.0, hemoglobin=5.0, glucose=50,
        dx_malig=0, dx_myoma=0, wt_low=0, wt_mid=0,
    )
    prob, _ = compute_risk(patient)
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"


def test_high_risk_scenario():
    """High-risk patient: malignant + prior surgery + high Hb + low albumin -> risk > 15%."""
    patient = make_patient(
        age=35, bmi=32, htn=1, dm=1, smoking=1, prior_surg=1,
        sc_sp=2.4, sc_below=2.4, sc_above=2.6,
        albumin=2.8, hemoglobin=14.5, glucose=85,
        dx_malig=1, dx_myoma=0, wt_low=1, wt_mid=0,
    )
    prob, _ = compute_risk(patient)
    assert prob > 0.15, f"High risk patient should be >15%, got {prob*100:.2f}%"


def test_low_risk_scenario():
    """Low-risk patient: benign myoma + no comorbidities + Pfannenstiel -> risk < 3%."""
    patient = make_patient(
        age=55, bmi=22,
        albumin=4.3, hemoglobin=12.0, glucose=100,
        dx_malig=0, dx_myoma=1, wt_low=0, wt_mid=0,
    )
    prob, _ = compute_risk(patient)
    assert prob < 0.03, f"Low risk patient should be <3%, got {prob*100:.2f}%"


def test_coefficient_integrity():
    """LR_COEFFICIENTS must contain exactly 17 variables."""
    assert len(LR_COEFFICIENTS) == 17, f"Expected 17, got {len(LR_COEFFICIENTS)}"
    expected_keys = {
        'Age', 'BMI', 'Hypertension', 'Diabetes', 'Smoking',
        'Prior_abd_surgery', 'Pulmonary_disease',
        'SC_fat_SP', 'SC_fat_below_umbilicus', 'SC_fat_above_umbilicus',
        'Preop_albumin', 'Preop_hemoglobin', 'Preop_glucose',
        'Dx_malignant', 'Dx_benign_myoma', 'WT_low_midline', 'WT_midline',
    }
    assert set(LR_COEFFICIENTS.keys()) == expected_keys


def test_population_stats_integrity():
    """POPULATION_STATS must have 17 variables, all with std > 0."""
    assert len(POPULATION_STATS) == 17, f"Expected 17, got {len(POPULATION_STATS)}"
    for var, stats in POPULATION_STATS.items():
        assert 'mean' in stats, f"Missing mean for {var}"
        assert 'std' in stats, f"Missing std for {var}"
        assert stats['std'] > 0, f"std must be > 0 for {var}, got {stats['std']}"


def test_probability_range():
    """Any combination of extreme inputs should produce probability in [0, 1]."""
    extremes = [
        make_patient(age=14, bmi=14.0, albumin=1.0, hemoglobin=5.0, glucose=50),
        make_patient(age=90, bmi=45.0, albumin=5.5, hemoglobin=18.0, glucose=400),
        make_patient(
            age=90, bmi=45.0, htn=1, dm=1, smoking=1, prior_surg=1, pul_disease=1,
            sc_sp=5.5, sc_below=6.0, sc_above=6.0,
            albumin=1.0, hemoglobin=18.0, glucose=50,
            dx_malig=1, dx_myoma=0, wt_low=1, wt_mid=0,
        ),
    ]
    for i, patient in enumerate(extremes):
        prob, _ = compute_risk(patient)
        assert 0.0 <= prob <= 1.0, f"Extreme case {i}: probability {prob} out of [0,1]"


def test_feature_encoding():
    """Diagnosis and wound type encoding must follow the PRD rules."""
    # Diagnosis
    assert encode_diagnosis("Malignant (cancer/carcinoma)") == (1, 0)
    assert encode_diagnosis("Benign myoma/leiomyoma") == (0, 1)
    assert encode_diagnosis("Other benign") == (0, 0)

    # Wound type
    assert encode_wound_type("Low midline") == (1, 0)
    assert encode_wound_type("Midline") == (0, 1)
    assert encode_wound_type("Pfannenstiel") == (0, 0)
