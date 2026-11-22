"""One bed's virtual monitor: emits vitals, obeys chaos state."""

import random
from collections import deque

from .vitals import alarms_for, initial_signs, step, validate_acuity

# Where an alarm storm drags each vital. All of them trip an alarm.
ALARM_TARGETS = {
    "heart_rate": 145.0,
    "spo2": 84.0,
    "systolic": 175.0,
    "diastolic": 110.0,
    "temperature": 39.5,
    "resp_rate": 30.0,
}

# Maximum alarm events kept per bed. Older events drop off the
# history list, but total_raised still counts every alarm ever raised.
MAX_ALARM_HISTORY = 1000


class Monitor:
    """A bedside monitor for one bed.

    ``tick(now)`` advances the monitor one step and returns the new
    reading, or ``None`` while the monitor is dropped out. Chaos state
    is set by the chaos engine as ``*_until`` timestamps (elapsed
    seconds); the monitor clears each one when its time passes.
    """

    def __init__(self, bed_id, profile="stable", rng=None, acuity=3):
        self.bed_id = bed_id
        self.profile = profile
        self.acuity = validate_acuity(acuity)
        self.rng = rng if rng is not None else random.Random()
        self.signs = initial_signs(profile, self.rng, self.acuity)
        self.seq = 0
        # False until the ward admits this bed (scenario timeline); an
        # unadmitted bed emits no readings at all.
        self.admitted = True
        self.offline_until = None
        self.stale_until = None
        self.alarm_until = None
        self.last_reading = None
        self.alarm_history = deque(maxlen=MAX_ALARM_HISTORY)
        self.total_raised = 0
        self._active_alarms = set()
        # Bed turnover: after a discharge the bed is cleaning until
        # ``cleaning_until`` (elapsed seconds), then available again.
        # ``turnovers`` counts discharges; ``total_cleaning_s`` is the
        # total time this bed has spent being cleaned.
        self.cleaning_until = None
        self.turnovers = 0
        self.total_cleaning_s = 0.0
        # Alarms raised by the most recent tick; the ward reads this to
        # hand them to the staffing model for acknowledgement.
        self.raised_this_tick = []

    def state(self, now):
        """Bed state at elapsed time ``now``.

        ``"occupied"`` while a patient is admitted, ``"cleaning"`` after
        a discharge until the turnover period ends, ``"available"``
        once the bed can take a new admission.
        """
        if self.admitted:
            return "occupied"
        if self.cleaning_until is not None and now < self.cleaning_until:
            return "cleaning"
        return "available"

    def discharge(self, now, cleaning_s):
        """Discharge the patient: the bed goes into cleaning.

        ``cleaning_s`` is how long turnover takes; the bed is not
        re-admittable until it has elapsed. Raises ValueError if the
        bed is not occupied. The discharged patient's alarms clear.
        """
        if cleaning_s <= 0:
            raise ValueError("cleaning_s must be > 0")
        if not self.admitted:
            raise ValueError("bed %d is not occupied" % self.bed_id)
        self._record_alarm_transitions([], now)
        self.admitted = False
        self.cleaning_until = round(now + cleaning_s, 3)
        self.turnovers += 1
        self.total_cleaning_s = round(self.total_cleaning_s + cleaning_s, 3)

    def admit(self, now, profile=None, acuity=None):
        """Admit a new patient to an available bed.

        Raises ValueError if the bed is occupied or still cleaning.
        A new admission starts with fresh vitals, so the previous
        patient's readings and alarms do not carry over.
        """
        state = self.state(now)
        if state == "occupied":
            raise ValueError("bed %d is already occupied" % self.bed_id)
        if state == "cleaning":
            raise ValueError(
                "bed %d is cleaning until t=%s" % (self.bed_id, self.cleaning_until)
            )
        if profile is not None:
            self.profile = profile
        if acuity is not None:
            self.acuity = validate_acuity(acuity)
        self.signs = initial_signs(self.profile, self.rng, self.acuity)
        self.seq = 0
        self.last_reading = None
        self.cleaning_until = None
        self.admitted = True

    def turnover_summary(self, now):
        """Per-bed turnover state and totals as plain JSON data."""
        return {
            "bed": self.bed_id,
            "state": self.state(now),
            "cleaning_until": self.cleaning_until
            if self.state(now) == "cleaning"
            else None,
            "turnovers": self.turnovers,
            "total_cleaning_s": self.total_cleaning_s,
        }

    def tick(self, now, dt=1.0):
        self.raised_this_tick = []
        if self.cleaning_until is not None and now >= self.cleaning_until:
            self.cleaning_until = None
        if not self.admitted:
            return None
        if self.offline_until is not None:
            if now < self.offline_until:
                return None
            self.offline_until = None

        if self.stale_until is not None and now < self.stale_until:
            if self.last_reading is not None:
                stale = dict(self.last_reading)
                stale["stale"] = True
                stale["t"] = round(now, 3)
                self.last_reading = stale
                return stale
            # No reading yet: fall through and generate one.
        self.stale_until = None

        if self.alarm_until is not None and now < self.alarm_until:
            for key, target in ALARM_TARGETS.items():
                self.signs[key] = round(
                    self.signs[key] + (target - self.signs[key]) * 0.35, 2
                )
        else:
            self.alarm_until = None
            self.signs = step(self.signs, self.profile, self.rng, dt, self.acuity)

        self.seq += 1
        reading = {
            "bed": self.bed_id,
            "seq": self.seq,
            "t": round(now, 3),
            "profile": self.profile,
            "acuity": self.acuity,
            "stale": False,
            **self.signs,
        }
        reading["alarms"] = alarms_for(self.signs)
        self._record_alarm_transitions(reading["alarms"], now)
        self.last_reading = reading
        return reading

    def _record_alarm_transitions(self, alarms, now):
        """Record raised/cleared events when alarms cross a band edge.

        An event is ``{"alarm": name, "raised_t": t, "cleared_t": t|None}``.
        A stale reading repeats the previous alarms, so it records nothing;
        while the monitor is offline no reading is emitted, so its alarms
        are neither raised nor cleared, and transitions are computed
        against the last reading it emitted when it comes back.
        """
        now = round(now, 3)
        current = set(alarms)
        raised = sorted(current - self._active_alarms)
        self.raised_this_tick = raised
        for name in raised:
            self.alarm_history.append(
                {"alarm": name, "raised_t": now, "cleared_t": None}
            )
            self.total_raised += 1
        for name in sorted(self._active_alarms - current):
            for event in reversed(self.alarm_history):
                if event["alarm"] == name and event["cleared_t"] is None:
                    event["cleared_t"] = now
                    break
        self._active_alarms = current

    def alarm_rate(self, now, window_s=60.0):
        """Alarms raised per minute over the trailing ``window_s`` seconds."""
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        start = now - window_s
        raised = sum(
            1 for e in self.alarm_history if start <= e["raised_t"] <= now
        )
        return round(raised / (window_s / 60.0), 3)

    def alarm_summary(self, now, window_s=60.0):
        """Per-bed alarm history and rate metrics as plain JSON data."""
        return {
            "bed": self.bed_id,
            "profile": self.profile,
            "acuity": self.acuity,
            "active": sorted(self._active_alarms),
            "total_raised": self.total_raised,
            "rate_per_min": self.alarm_rate(now, window_s),
            "history": [dict(e) for e in self.alarm_history],
        }

    def read(self):
        """Most recent reading, or None if the monitor never emitted one."""
        return self.last_reading
