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

With ``auto_order`` on, the ward acts on the flag: the first tick a
bed is discharge-ready, the ward raises a discharge-planning order
for that bed (source ``"auto"``), so the patient is discharged
``lead_time_s`` later through the normal planning flow. One order
per stay — a bed with a live order already in flight gets no
second one, and a new patient admitted to the bed later gets a
fresh order of their own. Off by default, so wards that do not
ask for it behave exactly as before.

``auto_order_lead_time_s`` overrides the lead time for those auto
orders only: a ward that discharges a stable patient an hour after
rounds can still send a ready patient home in minutes, without
changing the default lead time for scheduled or manual orders.
When it is not set, auto orders use the planner's default lead
time, exactly as before.
"""

# Default stability window: five minutes of quiet vitals.
DEFAULT_STABLE_WINDOW_S = 300.0


class DischargeReadiness:
    """Discharge-readiness configuration.

    ``stable_window_s`` is how long a patient's vitals must have
    been alarm-free before they count as stable (> 0). ``auto_order``
    raises a discharge-planning order the first tick a bed is
    flagged ready (off by default). ``auto_order_lead_time_s``
    overrides the lead time for those auto orders only (the
    planner's default when unset); it must be > 0 when set.
    """

    def __init__(
        self,
        stable_window_s=DEFAULT_STABLE_WINDOW_S,
        auto_order=False,
        auto_order_lead_time_s=None,
    ):
        if (
            not isinstance(stable_window_s, (int, float))
            or isinstance(stable_window_s, bool)
            or stable_window_s <= 0
        ):
            raise ValueError(
                "stable_window_s must be > 0, got %r" % (stable_window_s,)
            )
        if not isinstance(auto_order, bool):
            raise ValueError(
                "auto_order must be a bool, got %r" % (auto_order,)
            )
        if auto_order_lead_time_s is not None and (
            not isinstance(auto_order_lead_time_s, (int, float))
            or isinstance(auto_order_lead_time_s, bool)
            or auto_order_lead_time_s <= 0
        ):
            raise ValueError(
                "auto_order_lead_time_s must be > 0, got %r"
                % (auto_order_lead_time_s,)
            )
        self.stable_window_s = float(stable_window_s)
        self.auto_order = auto_order
        self.auto_order_lead_time_s = (
            None
            if auto_order_lead_time_s is None
            else float(auto_order_lead_time_s)
        )
