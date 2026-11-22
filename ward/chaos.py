"""Chaos events: device dropout, alarm storms, stale readings, partitions,
discharges."""

VALID_TYPES = ("dropout", "alarm_storm", "stale", "partition", "discharge")


class ChaosEvent:
    """One chaos event.

    ``type`` is one of dropout, alarm_storm, stale, partition, discharge.
    ``at`` is the start time in elapsed seconds, ``duration`` in seconds.
    ``bed`` targets one bed; ``beds`` targets several. ``partition`` is
    ward-wide and ignores bed/beds. ``discharge`` is instant — it fires
    once at ``at`` and ``duration`` is ignored, so it may be omitted.
    """

    def __init__(self, type, at, duration=None, bed=None, beds=None):
        if type not in VALID_TYPES:
            raise ValueError("unknown chaos type: %r" % type)
        if type == "discharge":
            duration = 0 if duration is None else duration
            if duration < 0:
                raise ValueError("chaos duration must be >= 0")
        elif duration is None or duration <= 0:
            raise ValueError("chaos duration must be > 0")
        if at < 0:
            raise ValueError("chaos start time must be >= 0")
        self.type = type
        self.at = float(at)
        self.duration = float(duration)
        self.bed = bed
        self.beds = list(beds) if beds is not None else None

    @property
    def end(self):
        return self.at + self.duration

    def active(self, now):
        return self.at <= now < self.end

    def targets(self, ward):
        """Bed indexes this event applies to (all beds if unspecified)."""
        if self.beds is not None:
            return self.beds
        if self.bed is not None:
            return [self.bed]
        return list(range(len(ward.monitors)))


class ChaosEngine:
    """Applies active chaos events to a ward each tick."""

    def __init__(self, events):
        self.events = list(events)
        # Discharge events fire once, when their time is reached.
        self._discharged = set()

    def apply(self, ward, now):
        ward.partitioned = any(
            ev.type == "partition" and ev.active(now) for ev in self.events
        )
        for ev in self.events:
            if ev.type == "discharge":
                if id(ev) not in self._discharged and now >= ev.at:
                    self._discharged.add(id(ev))
                    for b in ev.targets(ward):
                        if ward.monitors[b].admitted:
                            ward.monitors[b].discharge(now, ward.cleaning_s)
                continue
            if not ev.active(now):
                continue
            if ev.type == "dropout":
                for b in ev.targets(ward):
                    ward.monitors[b].offline_until = ev.end
            elif ev.type == "stale":
                for b in ev.targets(ward):
                    ward.monitors[b].stale_until = ev.end
            elif ev.type == "alarm_storm":
                for b in ev.targets(ward):
                    ward.monitors[b].alarm_until = ev.end
