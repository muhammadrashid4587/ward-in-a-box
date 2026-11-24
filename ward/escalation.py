"""NEWS2-driven staffing escalation.

When several patients are at high NEWS2 risk at once, the ward is
in trouble and the roster on shift cannot keep up: alarms wait
longer for acknowledgement, vitals checks slip, and responses to
escalations are due later, because all of those clocks run on the
staffing load factor. A real ward answers that by calling in extra
staff. This module is that call: when the number of deterioration
episodes open at high risk reaches ``threshold_high``, the ward
escalates — ``extra_nurses`` additional nurses join the effective
roster, so the load factor falls and the ack, check, and response
clocks speed up again from the next tick.

The escalation is hysteresis, not a switch: once it is active it
stays active while the count is at or above the threshold, and
after the count drops below it the escalation holds for
``cooldown_s`` more seconds before standing down, so a ward
hovering at the threshold does not flap extra nurses in and out
every tick. Every escalation is recorded — when it raised, when it
cleared, how many high episodes were open when it raised and at
their worst, and which beds they were — so a client can see that
the ward called for help and when, not just that the load factor
moved.

Off by default: a ward with no ``news2_staffing_escalation`` config
behaves exactly as before, with no extra nurses ever called in.
The escalation never touches the vital-sign alarm list, and it
never changes the NEWS2 scores themselves — it only changes how
fast the ward responds to them.
"""

# Defaults: escalate when 2 episodes are open at high risk, call
# in 1 extra nurse, and hold the escalation for 60 s after the
# count drops below the threshold before standing down.
DEFAULT_NEWS2_STAFFING_ESCALATION = {
    "threshold_high": 2,
    "extra_nurses": 1,
    "cooldown_s": 60.0,
}


def validate_news2_staffing_escalation(value):
    """Normalize a NEWS2 staffing-escalation config.

    None means off (no escalation ever fires); a dict may set
    ``threshold_high`` (open high-risk episodes that trigger an
    escalation, an int of at least 1), ``extra_nurses``
    (additional nurses called in while an escalation is active,
    an int of at least 1), and ``cooldown_s`` (seconds the
    escalation holds after the open-high count drops below the
    threshold, a number of at least 0). Returns the normalized
    dict.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            "news2_staffing_escalation must be a dict, got %r"
            % (value,)
        )
    unknown = sorted(
        set(value) - set(DEFAULT_NEWS2_STAFFING_ESCALATION)
    )
    if unknown:
        raise ValueError(
            "news2_staffing_escalation has unknown keys: %s"
            % ", ".join(unknown)
        )
    out = dict(DEFAULT_NEWS2_STAFFING_ESCALATION)
    for key, threshold in value.items():
        if key == "cooldown_s":
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or threshold < 0
            ):
                raise ValueError(
                    "news2_staffing_escalation cooldown_s must be a "
                    "number >= 0, got %r" % (threshold,)
                )
            out[key] = float(threshold)
        else:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int)
                or threshold < 1
            ):
                raise ValueError(
                    "news2_staffing_escalation %s must be an int >= 1, "
                    "got %r" % (key, threshold)
                )
            out[key] = threshold
    return out


class StaffingEscalation:
    """One ward's NEWS2-driven staffing escalation state.

    ``enabled`` False means the ward never escalates — the
    config was absent — and every method is a no-op reporting
    the unconfigured shape. With a config, ``tick(now,
    open_high_beds)`` runs the state machine each tick and
    ``extra_nurses_now`` says how many additional nurses are on
    the effective roster right now.
    """

    def __init__(self, threshold_high=2, extra_nurses=1,
                 cooldown_s=60.0, enabled=True):
        self.enabled = enabled
        self.threshold_high = threshold_high
        self.extra_nurses = extra_nurses
        self.cooldown_s = float(cooldown_s)
        self.active = False
        self.escalations_raised = 0
        self.history = []  # closed escalations, in raise order
        self._current = None  # the open escalation, if active
        self._cooldown_until = None

    @property
    def extra_nurses_now(self):
        """Additional nurses on the effective roster right now."""
        if not self.enabled or not self.active:
            return 0
        return self.extra_nurses

    def tick(self, now, open_high_beds):
        """Advance the escalation state machine to ``now``.

        ``open_high_beds`` is the list of beds with a
        deterioration episode open at high NEWS2 risk. Raises an
        escalation when the count first reaches the threshold,
        holds it while the count stays at or above it, and stands
        it down ``cooldown_s`` after the count first drops below
        it — unless the count climbs back to the threshold first,
        which cancels the cooldown.
        """
        now = round(now, 3)
        if not self.enabled:
            return
        open_high = len(open_high_beds)
        if not self.active:
            if open_high >= self.threshold_high:
                self.active = True
                self.escalations_raised += 1
                self._current = {
                    "raised_t": now,
                    "cleared_t": None,
                    "open_high_at_raise": open_high,
                    "peak_open_high": open_high,
                    "extra_nurses": self.extra_nurses,
                    "beds": sorted(open_high_beds),
                }
                self._cooldown_until = None
            return
        if open_high > self._current["peak_open_high"]:
            self._current["peak_open_high"] = open_high
        for bed in open_high_beds:
            if bed not in self._current["beds"]:
                self._current["beds"].append(bed)
        self._current["beds"].sort()
        if open_high >= self.threshold_high:
            self._cooldown_until = None
            return
        if self._cooldown_until is None:
            self._cooldown_until = round(now + self.cooldown_s, 3)
        elif now >= self._cooldown_until:
            self._current["cleared_t"] = now
            self.history.append(self._current)
            self._current = None
            self.active = False
            self._cooldown_until = None

    def summary(self, now, open_high_beds):
        """Escalation state as plain JSON data."""
        escalations = [dict(e) for e in self.history]
        if self._current is not None:
            escalations.append(dict(self._current))
        return {
            "configured": self.enabled,
            "active": self.active,
            "threshold_high": self.threshold_high if self.enabled else None,
            "extra_nurses": self.extra_nurses if self.enabled else 0,
            "open_high_episodes": len(open_high_beds),
            "escalations_raised": self.escalations_raised,
            "escalations": escalations,
        }
