"""Discharge-readiness flagging: when is a patient ready to leave?

A patient is flagged discharge-ready when two things are true:

- their vitals have been stable for ``stable_window_s`` seconds —
  no alarm raised in that window, measured from the last time their
  alarms cleared (or from admission, if they never alarmed), and
- all their orders are complete — no medication dose outstanding:
  nothing still waiting to be given, and no missed dose that has
  not been followed by a given one. A patient with no medications
  has no orders outstanding, so they are complete by default.

"Orders" here means medication orders: the only orders a patient
carries in this simulator. A planned discharge order does not block
readiness — a patient whose discharge is already ordered is exactly
the patient this flag is for. The flag is advisory only: this is a
test tool, not a clinical judgement.
"""

# Default stability window: five minutes of quiet vitals.
DEFAULT_STABLE_WINDOW_S = 300.0


class DischargeReadiness:
    """Discharge-readiness configuration.

    ``stable_window_s`` is how long a patient's vitals must have
    been alarm-free before they count as stable (> 0).
    """

    def __init__(self, stable_window_s=DEFAULT_STABLE_WINDOW_S):
        if (
            not isinstance(stable_window_s, (int, float))
            or isinstance(stable_window_s, bool)
            or stable_window_s <= 0
        ):
            raise ValueError(
                "stable_window_s must be > 0, got %r" % (stable_window_s,)
            )
        self.stable_window_s = float(stable_window_s)
