"""Staffing and shift model: who is on the ward, and what that does to care.

A ward runs on two shifts — day (07:00-19:00) and night (19:00-07:00)
— each rostered with nurses, doctors, and techs. The number that
matters for response times is the nurse-to-patient ratio: one nurse
can keep up with up to four occupied beds. Past that, every alarm
takes longer to acknowledge and every routine vitals check slips
later, in proportion to how overloaded the nurses are. The first
fifteen minutes after a shift change are a handover, when response is
half as fast again even with a full roster.

The model is deliberately deterministic: given the roster, the shift,
and the occupied beds, an alarm raised now is acknowledged exactly
``BASE_ACK_S * load_factor`` seconds later, and a vitals check is due
every ``CHECK_INTERVAL_S * load_factor`` seconds. The simulation's
randomness lives in the vitals and the alarms, not in the staff, so a
seeded run shows the staffing effect and nothing else.
"""

ROLES = ("nurse", "doctor", "tech")
SHIFTS = ("day", "night")

# Day shift runs 07:00-19:00, night 19:00-07:00.
DAY_START_HOUR = 7
DAY_END_HOUR = 19
SECONDS_PER_DAY = 86400

# Seconds for a nurse to acknowledge an alarm with a full roster.
BASE_ACK_S = 30.0
# Occupied beds one nurse can keep up with before response degrades.
ADEQUATE_PATIENTS_PER_NURSE = 4
# Seconds between routine vitals checks with a full roster.
CHECK_INTERVAL_S = 60.0
# A check counts as delayed when it lands this far past the 60 s rhythm.
CHECK_DELAYED_AFTER_S = 75.0
# Minutes after a shift change that count as handover.
HANDOVER_WINDOW_S = 900.0
# Response slowdown during handover, on top of the load factor.
HANDOVER_FACTOR = 1.5
# Load factor when no nurse is on shift at all.
NO_NURSE_FACTOR = 4.0

# Default roster: 12 beds, 3 nurses on days (1:4, just adequate),
# 2 on nights (1:6, understaffed) — so a run across a shift change
# shows the difference without any configuration.
DEFAULT_ROSTER = {
    "day": {"nurse": 3, "doctor": 1, "tech": 1},
    "night": {"nurse": 2, "doctor": 1, "tech": 0},
}


def _validate_roster(roster, shift):
    if not isinstance(roster, dict):
        raise ValueError("staffing %s shift must be an object" % shift)
    for role, count in roster.items():
        if role not in ROLES:
            raise ValueError("unknown staff role: %r" % role)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(
                "staffing %s %s must be a non-negative int, got %r"
                % (shift, role, count)
            )
    return {role: roster.get(role, 0) for role in ROLES}


class Staffing:
    """Roster, shifts, alarm acknowledgement, and vitals checks.

    ``elapsed`` is seconds since the run started; the ward clock starts
    at ``start_hour`` (08:00 by default, an hour into the day shift), so
    elapsed 0 is mid-shift, not a handover.
    """

    def __init__(self, day=None, night=None, start_hour=8):
        if not isinstance(start_hour, int) or isinstance(start_hour, bool):
            raise ValueError("start_hour must be an int in 0..23, got %r" % (start_hour,))
        if not 0 <= start_hour <= 23:
            raise ValueError("start_hour must be an int in 0..23, got %r" % (start_hour,))
        self.start_hour = start_hour
        roster = dict(DEFAULT_ROSTER)
        if day is not None:
            roster["day"] = day
        if night is not None:
            roster["night"] = night
        self.roster = {s: _validate_roster(roster[s], s) for s in SHIFTS}
        self._pending = []  # {"bed", "alarm", "raised_t", "ack_t"}
        self._acked_times = []  # seconds from raise to ack
        self.raised = 0
        self.acked = 0
        self.checks_performed = 0
        self.checks_delayed = 0
        self._next_check_at = {}  # bed_id -> elapsed of next due check
        self._last_check_at = {}  # bed_id -> elapsed of last check
        # Extra nurses called in by a NEWS2 staffing escalation
        # (see ward.escalation): set by the ward each tick, 0
        # when no escalation is active. They count toward the
        # nurse-to-patient ratio exactly like rostered nurses, so
        # an escalation lowers the load factor from the next tick.
        self.escalation_nurses = 0
        # Staff lost to active staff-shortage chaos events (see
        # ward.chaos): set by the ward each tick as a
        # ``{role: count}`` map, {} when the roster is whole.
        # Lost nurses come off the effective roster, so a sick
        # call raises the load factor from the next tick; lost
        # doctors and techs are recorded but change nothing,
        # because the model keys response times off nurses alone.
        self.shortage = {}

    def shift_at(self, elapsed):
        """Which shift is on at ``elapsed`` seconds into the run."""
        hour = (self.start_hour + elapsed / 3600.0) % 24
        return "day" if DAY_START_HOUR <= hour < DAY_END_HOUR else "night"

    def on_shift(self, elapsed):
        """Staff counts on duty at ``elapsed`` seconds into the run."""
        return dict(self.roster[self.shift_at(elapsed)])

    def _seconds_since_shift_start(self, elapsed):
        """Seconds since the current shift began (handover clock)."""
        clock_s = (self.start_hour * 3600.0 + elapsed) % SECONDS_PER_DAY
        boundaries = (DAY_START_HOUR * 3600.0, DAY_END_HOUR * 3600.0)
        return min((clock_s - b) % SECONDS_PER_DAY for b in boundaries)

    def in_handover(self, elapsed):
        """True during the first 15 minutes of a shift."""
        return self._seconds_since_shift_start(elapsed) < HANDOVER_WINDOW_S

    def effective_nurses(self, elapsed):
        """Nurses on duty at ``elapsed``, roster plus any called in
        by a NEWS2 staffing escalation, minus any lost to an
        active staff shortage — never below zero, so losing the
        whole roster leaves the ward with no nurse at all."""
        return max(
            0,
            self.on_shift(elapsed)["nurse"]
            + self.escalation_nurses
            - self.shortage.get("nurse", 0),
        )

    def patients_per_nurse(self, occupied_beds, elapsed):
        """Occupied beds per nurse on shift, or None with no nurse.

        Counts escalation nurses: a ward that called in extra
        staff has more nurses on the effective roster than the
        shift roster alone.
        """
        nurses = self.effective_nurses(elapsed)
        if nurses <= 0:
            return None
        return occupied_beds / nurses

    def load_factor(self, occupied_beds, elapsed):
        """How much slower care is than baseline at this moment.

        1.0 with an adequate roster, rising in proportion to the
        nurse overload past 1:4, times 1.5 during handover.
        """
        ppn = self.patients_per_nurse(occupied_beds, elapsed)
        if ppn is None:
            factor = NO_NURSE_FACTOR
        else:
            factor = max(1.0, ppn / ADEQUATE_PATIENTS_PER_NURSE)
        if self.in_handover(elapsed):
            factor *= HANDOVER_FACTOR
        return factor

    def alarm_raised(self, bed, alarm, now, occupied_beds):
        """Register a raised alarm; it will be acked after the delay."""
        now = round(now, 3)
        delay = BASE_ACK_S * self.load_factor(occupied_beds, now)
        self._pending.append(
            {"bed": bed, "alarm": alarm, "raised_t": now, "ack_t": round(now + delay, 3)}
        )
        self.raised += 1

    def tick(self, now, occupied_beds, online_beds):
        """Advance acks and vitals checks to ``now``.

        ``occupied_beds`` is the number of admitted beds;
        ``online_beds`` the admitted beds whose monitor is up.
        """
        now = round(now, 3)
        factor = self.load_factor(occupied_beds, now)
        for pending in list(self._pending):
            if pending["ack_t"] <= now:
                self._pending.remove(pending)
                self._acked_times.append(
                    round(pending["ack_t"] - pending["raised_t"], 3)
                )
                self.acked += 1
        for bed in online_beds:
            due = self._next_check_at.get(bed, 0.0)
            if now >= due:
                last = self._last_check_at.get(bed)
                if last is not None and now - last > CHECK_DELAYED_AFTER_S:
                    self.checks_delayed += 1
                self.checks_performed += 1
                self._last_check_at[bed] = now
                self._next_check_at[bed] = round(now + CHECK_INTERVAL_S * factor, 3)

    def mean_time_to_ack(self):
        """Mean seconds from raise to ack, or None before the first ack."""
        if not self._acked_times:
            return None
        return round(sum(self._acked_times) / len(self._acked_times), 3)

    def summary(self, now, occupied_beds):
        """Staffing state and care metrics as plain JSON data.

        ``on_shift`` is the rostered staff; ``escalation_nurses``
        is how many extra nurses a NEWS2 staffing escalation has
        called in (0 without one) and ``shortage`` is the staff
        an active staff-shortage event has taken off the roster
        ({} with none) — the ratio and the load factor count
        both.
        """
        ppn = self.patients_per_nurse(occupied_beds, now)
        return {
            "t": round(now, 3),
            "shift": self.shift_at(now),
            "handover": self.in_handover(now),
            "on_shift": self.on_shift(now),
            "escalation_nurses": self.escalation_nurses,
            "shortage": dict(self.shortage),
            "occupied_beds": occupied_beds,
            "patients_per_nurse": None if ppn is None else round(ppn, 2),
            "adequate": ppn is not None and ppn <= ADEQUATE_PATIENTS_PER_NURSE,
            "load_factor": round(self.load_factor(occupied_beds, now), 3),
            "alarms": {
                "raised": self.raised,
                "acked": self.acked,
                "pending": len(self._pending),
                "mean_time_to_ack_s": self.mean_time_to_ack(),
            },
            "checks": {
                "performed": self.checks_performed,
                "delayed": self.checks_delayed,
            },
        }
