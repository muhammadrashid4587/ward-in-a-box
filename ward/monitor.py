"""One bed's virtual monitor: emits vitals, obeys chaos state."""

import random
from collections import deque

from .vitals import alarms_for, initial_signs, step

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

    def __init__(self, bed_id, profile="stable", rng=None):
        self.bed_id = bed_id
        self.profile = profile
        self.rng = rng if rng is not None else random.Random()
        self.signs = initial_signs(profile, self.rng)
        self.seq = 0
        self.offline_until = None
        self.stale_until = None
        self.alarm_until = None
        self.last_reading = None
        self.alarm_history = deque(maxlen=MAX_ALARM_HISTORY)
        self.total_raised = 0
        self._active_alarms = set()

    def tick(self, now, dt=1.0):
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
            self.signs = step(self.signs, self.profile, self.rng, dt)

        self.seq += 1
        reading = {
            "bed": self.bed_id,
            "seq": self.seq,
            "t": round(now, 3),
            "profile": self.profile,
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
        for name in sorted(current - self._active_alarms):
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
            "active": sorted(self._active_alarms),
            "total_raised": self.total_raised,
            "rate_per_min": self.alarm_rate(now, window_s),
            "history": [dict(e) for e in self.alarm_history],
        }

    def read(self):
        """Most recent reading, or None if the monitor never emitted one."""
        return self.last_reading
