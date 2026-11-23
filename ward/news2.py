"""NEWS2 early-warning scoring (National Early Warning Score 2).

The Royal College of Physicians' NEWS2 scores six physiological
parameters — respiratory rate, SpO2, systolic blood pressure, heart
rate, temperature, and consciousness — 0 to 3 each, and adds them up.
The aggregate drives the response: 0-4 is low risk, 5-6 (or any
single parameter scoring 3, a "red score") is medium risk and needs
an urgent review by the ward team, 7 or more is high risk and needs
an urgent review by the critical care team.

Consciousness is the sixth parameter, scored on the ACVPU scale:
Alert scores 0, anything else — Confusion, responding to Voice,
responding to Pain, or Unresponsive — scores 3. It is the only
parameter that is not a vital: it is set per patient in the scenario
and does not change over a run, so a patient who starts confused
stays confused. That is a deliberate honesty limit, not an oversight:
the simulator generates no consciousness dynamics, so it records a
fixed state rather than inventing one — a patient whose consciousness
changes mid-run is outside what this ward models.

SpO2 is scored on Scale 1 for patients not on supplemental oxygen
and on Scale 2 for patients on oxygen (see ``score_spo2``). The
flag is per patient, set in the scenario; oxygen therapy is a
scoring flag only — it does not change the vitals the monitor
generates, so a patient on oxygen keeps whatever SpO2 their
profile produces.

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


def score_spo2(spo2, on_oxygen=False):
    """NEWS2 points for SpO2.

    Scale 1 when the patient is not on supplemental oxygen, Scale 2
    when they are. Scale 2 is NEWS2's oxygen-target table: a patient
    on oxygen is expected to sit at 88-92, so 93-96 scores 2 and 97
    or above scores 3 — the same SpO2 that is fine on air is a
    warning on oxygen.
    """
    if on_oxygen:
        if spo2 <= 83:
            return 3
        if spo2 <= 85:
            return 2
        if spo2 <= 87:
            return 1
        if spo2 <= 92:
            return 0
        if spo2 <= 96:
            return 2
        return 3
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


# Consciousness on the ACVPU scale: Alert, Confusion, responding to
# Voice, responding to Pain, Unresponsive. Lowercase in code and in
# scenario files; NEWS2 scores Alert 0 and anything else 3.
ACVPU_LEVELS = ("alert", "confusion", "voice", "pain", "unresponsive")


def validate_consciousness(level):
    """A consciousness level must be one of the ACVPU levels."""
    if level not in ACVPU_LEVELS:
        raise ValueError(
            "consciousness must be one of %s, got %r"
            % (", ".join(ACVPU_LEVELS), level)
        )
    return level


def score_consciousness(level):
    """NEWS2 points for consciousness: Alert 0, anything else 3."""
    validate_consciousness(level)
    return 0 if level == "alert" else 3


def score_signs(signs, on_oxygen=False, consciousness="alert"):
    """Score a set of vitals.

    Returns ``{"score": total, "risk": "low"|"medium"|"high",
    "red_score": bool, "spo2_scale": 1|2,
    "components": {parameter: points, ...},
    "response": str}``. ``red_score`` is True when any single
    parameter scores 3 — that alone makes the risk medium, even
    when the aggregate is low. ``spo2_scale`` says which SpO2
    table was used: 2 when the patient is on oxygen, 1 otherwise.
    ``consciousness`` is the patient's ACVPU level: Alert scores 0,
    anything else scores 3 — so a confused patient is medium risk
    on the consciousness red score alone, whatever their vitals say.
    """
    components = {
        "resp_rate": score_resp_rate(signs["resp_rate"]),
        "spo2": score_spo2(signs["spo2"], on_oxygen),
        "systolic": score_systolic(signs["systolic"]),
        "heart_rate": score_heart_rate(signs["heart_rate"]),
        "temperature": score_temperature(signs["temperature"]),
        "consciousness": score_consciousness(consciousness),
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
        "spo2_scale": 2 if on_oxygen else 1,
        "components": components,
        "response": RESPONSE[risk],
    }
