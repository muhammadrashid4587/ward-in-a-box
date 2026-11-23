"""NEWS2 early-warning scoring (National Early Warning Score 2).

The Royal College of Physicians' NEWS2 scores six physiological
parameters — respiratory rate, SpO2, systolic blood pressure, heart
rate, temperature, and consciousness — 0 to 3 each, and adds them up.
The aggregate drives the response: 0-4 is low risk, 5-6 (or any
single parameter scoring 3, a "red score") is medium risk and needs
an urgent review by the ward team, 7 or more is high risk and needs
an urgent review by the critical care team.

Two parameters are not modelled here, and the score says so:

- Consciousness (ACVPU) is not a vital this simulator generates, so
  it is left out entirely rather than guessed. A real NEWS2 always
  includes it; a score from this ward is therefore a partial score
  and must never be treated as a clinical assessment.
- SpO2 uses Scale 1 (patients not on supplemental oxygen). The ward
  has no oxygen-therapy state, so Scale 2 does not apply.

The score is advisory only: this is a test tool for health
software, not medical software. It exists so a client can be tested
against a ward whose patients visibly deteriorate — a hypoxic
patient sits at medium risk on a red SpO2 score, an alarm storm
pushes a febrile patient to high — and so the escalation (who
deteriorated, when, how far) is recorded rather than just the raw
vitals.
"""

# The response each risk level calls for, in NEWS2's own terms.
RESPONSE = {
    "low": "routine monitoring (minimum 12-hourly)",
    "medium": "urgent review by ward team (hourly monitoring)",
    "high": "urgent review by critical care team (continuous monitoring)",
}


def score_resp_rate(rr):
    """NEWS2 points for respiratory rate (breaths per minute)."""
    if rr <= 8:
        return 3
    if rr <= 11:
        return 1
    if rr <= 20:
        return 0
    if rr <= 24:
        return 2
    return 3


def score_spo2(spo2):
    """NEWS2 points for SpO2, Scale 1 (no supplemental oxygen)."""
    if spo2 <= 91:
        return 3
    if spo2 <= 93:
        return 2
    if spo2 <= 95:
        return 1
    return 0


def score_systolic(sys):
    """NEWS2 points for systolic blood pressure (mmHg)."""
    if sys <= 90:
        return 3
    if sys <= 100:
        return 2
    if sys <= 110:
        return 1
    if sys <= 219:
        return 0
    return 3


def score_heart_rate(hr):
    """NEWS2 points for heart rate (beats per minute)."""
    if hr <= 40:
        return 3
    if hr <= 50:
        return 1
    if hr <= 90:
        return 0
    if hr <= 110:
        return 1
    if hr <= 130:
        return 2
    return 3


def score_temperature(temp):
    """NEWS2 points for temperature (degrees C)."""
    if temp <= 35.0:
        return 3
    if temp <= 36.0:
        return 1
    if temp <= 38.0:
        return 0
    if temp <= 39.0:
        return 1
    return 2


def score_signs(signs):
    """Score a set of vitals.

    Returns ``{"score": total, "risk": "low"|"medium"|"high",
    "red_score": bool, "components": {parameter: points, ...},
    "response": str}``. ``red_score`` is True when any single
    parameter scores 3 — that alone makes the risk medium, even
    when the aggregate is low.
    """
    components = {
        "resp_rate": score_resp_rate(signs["resp_rate"]),
        "spo2": score_spo2(signs["spo2"]),
        "systolic": score_systolic(signs["systolic"]),
        "heart_rate": score_heart_rate(signs["heart_rate"]),
        "temperature": score_temperature(signs["temperature"]),
    }
    total = sum(components.values())
    red = any(v == 3 for v in components.values())
    if total >= 7:
        risk = "high"
    elif total >= 5 or red:
        risk = "medium"
    else:
        risk = "low"
    return {
        "score": total,
        "risk": risk,
        "red_score": red,
        "components": components,
        "response": RESPONSE[risk],
    }
