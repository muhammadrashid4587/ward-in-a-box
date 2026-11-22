"""One bed's virtual monitor: emits vitals, obeys chaos state."""

import random

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
        self.last_reading = reading
        return reading

    def read(self):
        """Most recent reading, or None if the monitor never emitted one."""
        return self.last_reading
