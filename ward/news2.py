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
parameter that is not a vital: it is set per patient in the scenario,
and by default it does not change over a run. With
``consciousness_dynamics`` enabled for a patient, it does: the level
steps one ACVPU level at a time — down after the patient's
vitals-only NEWS2 risk (the score with consciousness held at Alert,
so the level itself cannot drive the decision) stays medium or high
for ``deteriorate_after_s`` seconds, up after it stays low for
``recover_after_s`` seconds. A deteriorating patient slides from
Alert toward Unresponsive while their vitals stay bad and climbs
back as they recover; a patient whose vitals never leave low risk
never changes. Every change is recorded with its time, so the
escalation (who deteriorated, when, how far) is recorded rather
than just the raw vitals.

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


# Consciousness dynamics, per patient: how long the vitals-only
# NEWS2 risk must hold before the ACVPU level steps one level.
# ``deteriorate_after_s`` is consecutive seconds at medium or high
# vitals risk before stepping one level worse; ``recover_after_s``
# is consecutive seconds at low vitals risk before stepping one
# level better. The clock is deterministic — no randomness — so a
# seeded run reproduces the same changes, and enabling dynamics
# never changes the vitals a monitor generates.
DEFAULT_CONSCIOUSNESS_DYNAMICS = {
    "deteriorate_after_s": 5.0,
    "recover_after_s": 10.0,
}


def validate_consciousness_dynamics(value):
    """Normalize a consciousness-dynamics config.

    None means off (the level never changes); True means the
    defaults; a dict may set either threshold. Returns the
    normalized dict, or None when dynamics are off.
    """
    if value is None:
        return None
    if value is True:
        return dict(DEFAULT_CONSCIOUSNESS_DYNAMICS)
    if not isinstance(value, dict):
        raise ValueError(
            "consciousness_dynamics must be a dict or true, got %r"
            % (value,)
        )
    unknown = sorted(set(value) - set(DEFAULT_CONSCIOUSNESS_DYNAMICS))
    if unknown:
        raise ValueError(
            "consciousness_dynamics has unknown keys: %s"
            % ", ".join(unknown)
        )
    out = dict(DEFAULT_CONSCIOUSNESS_DYNAMICS)
    for key, threshold in value.items():
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or threshold <= 0
        ):
            raise ValueError(
                "consciousness_dynamics %s must be a number > 0, got %r"
                % (key, threshold)
            )
        out[key] = float(threshold)
    return out


# NEWS2 trend / rate-of-change alerts: how a rising score is
# recognised. Two rules, either of which raises a trend alert:
# ``jump`` — the score rose by at least ``delta`` from the earliest
# entry inside the trailing ``window_s`` seconds — and ``creep`` —
# over the last ``trend_ticks`` entries the score rose on more ticks
# than it fell and the net rise is at least ``min_rise``. The jump
# rule catches a patient who deteriorates fast; the creep rule
# catches one who drifts up slowly, a point a tick, without ever
# jumping. Defaults: a rise of 2 within 60 seconds, or a net rise
# of 2 over the last 10 ticks with more rises than falls — a
# one-point wobble does not count as a trend.
DEFAULT_NEWS2_TREND = {
    "window_s": 60.0,
    "delta": 2,
    "trend_ticks": 10,
    "min_rise": 2,
}


def validate_news2_trend(value):
    """Normalize a NEWS2 trend config.

    None means the defaults; a dict may set ``window_s`` (seconds
    of trailing history the jump rule looks at, a number above 0),
    ``delta`` (score rise within that window that raises an alert,
    an int of at least 1), ``trend_ticks`` (how many recent ticks
    the creep rule looks at, an int of at least 2), and
    ``min_rise`` (net rise over those ticks that counts as a
    sustained upward slope, an int of at least 1). Returns the
    normalized dict.
    """
    if value is None:
        return dict(DEFAULT_NEWS2_TREND)
    if not isinstance(value, dict):
        raise ValueError(
            "news2_trend must be a dict, got %r" % (value,)
        )
    unknown = sorted(set(value) - set(DEFAULT_NEWS2_TREND))
    if unknown:
        raise ValueError(
            "news2_trend has unknown keys: %s" % ", ".join(unknown)
        )
    out = dict(DEFAULT_NEWS2_TREND)
    for key, threshold in value.items():
        if key == "window_s":
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or threshold <= 0
            ):
                raise ValueError(
                    "news2_trend window_s must be a number > 0, got %r"
                    % (threshold,)
                )
            out[key] = float(threshold)
        elif key == "trend_ticks":
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int)
                or threshold < 2
            ):
                raise ValueError(
                    "news2_trend trend_ticks must be an int >= 2, got %r"
                    % (threshold,)
                )
            out[key] = threshold
        else:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int)
                or threshold < 1
            ):
                raise ValueError(
                    "news2_trend %s must be an int >= 1, got %r"
                    % (key, threshold)
                )
            out[key] = threshold
    return out


def trend_for(history, window_s=60.0, delta=2, trend_ticks=10, min_rise=2):
    """The trend of a NEWS2 score history.

    ``history`` is the bed's score entries in tick order, each with
    ``"t"`` and ``"score"``. Returns ``{"rising": bool,
    "rule": "jump"|"creep"|None, "start_t": seconds|None,
    "start_score": int|None, "current_score": int|None,
    "delta": int, "window_s": float}``. ``start_score`` is the score
    the rise is measured from — the earliest entry inside the
    trailing window for the jump rule, the first of the last
    ``trend_ticks`` entries for the creep rule — ``current_score``
    is the latest score, and ``delta`` is the rise between them.
    With no history, nothing is rising and the scores are None.
    """
    if not history:
        return {
            "rising": False,
            "rule": None,
            "start_t": None,
            "start_score": None,
            "current_score": None,
            "delta": 0,
            "window_s": float(window_s),
        }
    entries = list(history)
    current = entries[-1]
    now = current["t"]
    window = [e for e in entries if e["t"] >= now - window_s]
    start = window[0]
    jump_delta = current["score"] - start["score"]
    if jump_delta >= delta:
        return {
            "rising": True,
            "rule": "jump",
            "start_t": start["t"],
            "start_score": start["score"],
            "current_score": current["score"],
            "delta": jump_delta,
            "window_s": float(window_s),
        }
    recent = entries[-trend_ticks:]
    if len(recent) >= 2:
        net = recent[-1]["score"] - recent[0]["score"]
        rises = sum(
            1
            for a, b in zip(recent, recent[1:])
            if b["score"] > a["score"]
        )
        falls = sum(
            1
            for a, b in zip(recent, recent[1:])
            if b["score"] < a["score"]
        )
        if net >= min_rise and rises > falls:
            return {
                "rising": True,
                "rule": "creep",
                "start_t": recent[0]["t"],
                "start_score": recent[0]["score"],
                "current_score": current["score"],
                "delta": net,
                "window_s": float(window_s),
            }
    return {
        "rising": False,
        "rule": None,
        "start_t": start["t"],
        "start_score": start["score"],
        "current_score": current["score"],
        "delta": jump_delta,
        "window_s": float(window_s),
    }


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
