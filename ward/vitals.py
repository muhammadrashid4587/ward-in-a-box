"""Vital-sign models: plausible ranges and random-walk generation."""

import random

# Normal bands used for alarms (resting adult).
NORMAL = {
    "heart_rate": (50.0, 120.0),
    "spo2": (90.0, 100.0),
    "systolic": (90.0, 160.0),
    "diastolic": (60.0, 100.0),
    "temperature": (35.5, 38.5),
    "resp_rate": (10.0, 24.0),
}

# Patient profiles: the range each vital wanders in for that patient.
# These are deliberately ordinary ward presentations, not edge cases.
PROFILES = {
    "stable": {
        "heart_rate": (62.0, 88.0),
        "spo2": (95.0, 99.0),
        "systolic": (105.0, 130.0),
        "diastolic": (65.0, 85.0),
        "temperature": (36.4, 37.2),
        "resp_rate": (12.0, 18.0),
    },
    "tachycardic": {
        "heart_rate": (102.0, 118.0),
        "spo2": (94.0, 98.0),
        "systolic": (110.0, 140.0),
        "diastolic": (70.0, 90.0),
        "temperature": (36.5, 37.4),
        "resp_rate": (14.0, 22.0),
    },
    "hypoxic": {
        "heart_rate": (85.0, 110.0),
        "spo2": (88.0, 93.0),
        "systolic": (100.0, 135.0),
        "diastolic": (60.0, 85.0),
        "temperature": (36.3, 37.3),
        "resp_rate": (18.0, 24.0),
    },
    "febrile": {
        "heart_rate": (95.0, 115.0),
        "spo2": (94.0, 98.0),
        "systolic": (105.0, 135.0),
        "diastolic": (65.0, 88.0),
        "temperature": (38.0, 39.0),
        "resp_rate": (16.0, 22.0),
    },
}

# Per-tick gaussian noise (sigma) for each vital.
SIGMA = {
    "heart_rate": 1.5,
    "spo2": 0.3,
    "systolic": 2.0,
    "diastolic": 1.5,
    "temperature": 0.05,
    "resp_rate": 0.6,
}


# Patient acuity, 1 (critical) to 4 (low/stable), on a triage-like scale.
# Acuity drives two things: how volatile a patient's vitals are, and how
# wide the range they wander in is. Level 3 is the neutral default: it
# reproduces exactly the behaviour of the profiles on their own, so
# existing scenarios are unchanged. A critical patient swings twice as
# hard and wanders 35% further from the profile midpoint, so they cross
# alarm bands more often; a low-acuity patient is steadier and stays
# closer to the middle.
ACUITY_SIGMA = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.6}
ACUITY_WIDEN = {1: 0.35, 2: 0.15, 3: 0.0, 4: -0.15}

# Hard plausible envelope: widened ranges are clamped inside these so
# no acuity level can push a vital somewhere a real monitor would never
# show. SpO2 cannot exceed 100.
PLAUSIBLE = {
    "heart_rate": (30.0, 200.0),
    "spo2": (70.0, 100.0),
    "systolic": (60.0, 220.0),
    "diastolic": (30.0, 140.0),
    "temperature": (34.0, 41.0),
    "resp_rate": (6.0, 40.0),
}


def validate_acuity(acuity):
    """Acuity must be an int in 1..4; anything else is a scenario bug."""
    if not isinstance(acuity, int) or isinstance(acuity, bool) or acuity not in ACUITY_SIGMA:
        raise ValueError("acuity must be an int in 1..4, got %r" % (acuity,))
    return acuity


def effective_ranges(profile, acuity=3):
    """A profile's vital ranges widened (or narrowed) by acuity."""
    validate_acuity(acuity)
    if profile not in PROFILES:
        raise ValueError("unknown profile: %r" % profile)
    widen = ACUITY_WIDEN[acuity]
    out = {}
    for key, (lo, hi) in PROFILES[profile].items():
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 * (1.0 + widen)
        plo, phi = PLAUSIBLE[key]
        out[key] = (round(max(plo, mid - half), 2), round(min(phi, mid + half), 2))
    return out


def initial_signs(profile, rng, acuity=3):
    """Draw a starting set of vitals inside the profile's range."""
    ranges = effective_ranges(profile, acuity)
    return {k: round(rng.uniform(*v), 2) for k, v in ranges.items()}


def step(signs, profile, rng, dt=1.0, acuity=3):
    """One random-walk step, mean-reverting to the profile midpoint."""
    validate_acuity(acuity)
    ranges = effective_ranges(profile, acuity)
    sigma = ACUITY_SIGMA[acuity]
    out = {}
    for key, (lo, hi) in ranges.items():
        mid = (lo + hi) / 2.0
        value = signs[key] + rng.gauss(0, SIGMA[key] * sigma * dt) + (mid - signs[key]) * 0.05
        out[key] = round(max(lo, min(hi, value)), 2)
    return out


def alarms_for(signs):
    """Alarm names for any vital outside its normal band."""
    alarms = []
    hr = signs["heart_rate"]
    if hr < NORMAL["heart_rate"][0]:
        alarms.append("bradycardia")
    elif hr > NORMAL["heart_rate"][1]:
        alarms.append("tachycardia")
    if signs["spo2"] < NORMAL["spo2"][0]:
        alarms.append("low_spo2")
    if signs["systolic"] > NORMAL["systolic"][1]:
        alarms.append("hypertension")
    elif signs["systolic"] < NORMAL["systolic"][0]:
        alarms.append("hypotension")
    if signs["temperature"] > NORMAL["temperature"][1]:
        alarms.append("fever")
    elif signs["temperature"] < NORMAL["temperature"][0]:
        alarms.append("hypothermia")
    if signs["resp_rate"] > NORMAL["resp_rate"][1]:
        alarms.append("tachypnea")
    return alarms
