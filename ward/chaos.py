"""Chaos events: device dropout, alarm storms, stale readings, partitions,
discharges, pharmacy stockouts, slow sensor drift."""

from .vitals import NORMAL

VALID_TYPES = ("dropout", "alarm_storm", "stale", "partition", "discharge", "stockout", "sensor_drift")


def validate_drift(drift):
    """A sensor-drift config: ``{vital: rate_per_second, ...}``.

    Each vital must be one the ward measures, each rate a finite
    number — positive drifts the reading up, negative drifts it
    down — and at least one rate must be non-zero, or the event
    does nothing at all.
    """
    if not isinstance(drift, dict) or not drift:
        raise ValueError(
            "sensor_drift needs a drift map, got %r" % (drift,)
        )
    out = {}
    for vital, rate in drift.items():
        if vital not in NORMAL:
            raise ValueError(
                "sensor_drift: unknown vital: %r" % (vital,)
            )
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or rate != rate
            or rate in (float("inf"), float("-inf"))
        ):
            raise ValueError(
                "sensor_drift: rate for %r must be a finite number, got %r"
                % (vital, rate)
            )
        if rate != 0:
            out[vital] = float(rate)
    if not out:
        raise ValueError("sensor_drift needs at least one non-zero rate")
    return out


class ChaosEvent:
    """One chaos event.

    ``type`` is one of dropout, alarm_storm, stale, partition, discharge,
    stockout, sensor_drift. ``at`` is the start time in elapsed seconds,
    ``duration`` in seconds. ``bed`` targets one bed; ``beds`` targets
    several. ``partition`` is ward-wide and ignores bed/beds. ``discharge``
    is instant — it fires once at ``at`` and ``duration`` is ignored, so it
    may be omitted. ``stockout`` needs ``med`` (the medication name,
    matched case-insensitively): while it is active, doses of that
    medication due on the target beds are skipped — recorded as
    ``stockout`` in the dose history, counted in the medication
    metrics, and flagged with a ``missed_dose`` alarm — because the
    pharmacy has none to give. Doses already pending when the stockout
    starts still resolve normally; only doses that fall due inside the
    window are skipped. ``sensor_drift`` needs ``drift`` (a map of
    vital to drift rate per second, see :func:`validate_drift`):
    while it is active, each target bed's sensor bias for those
    vitals grows by ``rate * dt`` every tick, so the reported
    reading creeps away from the patient's true vitals — a slow
    miscalibration, not a step change. The bias stays once the
    event ends (a drifted sensor does not heal itself) and is
    cleared only when a new patient is admitted to the bed.
    """

    def __init__(self, type, at, duration=None, bed=None, beds=None, med=None, drift=None):
        if type not in VALID_TYPES:
            raise ValueError("unknown chaos type: %r" % type)
        if type == "stockout":
            if not isinstance(med, str) or not med.strip():
                raise ValueError("stockout needs a medication name, got %r" % (med,))
        if type == "sensor_drift":
            drift = validate_drift(drift)
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
        self.med = med.strip().lower() if isinstance(med, str) else None
        self.drift = drift

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

    def stockouts(self, ward, now):
        """Active pharmacy stockouts: ``{bed_id: {med, ...}}``.

        Only beds that actually have the stocked-out medication in
        their schedule are listed, so a ward-wide stockout of insulin
        does not flag beds with no insulin prescribed.
        """
        out = {}
        for ev in self.events:
            if ev.type != "stockout" or not ev.active(now):
                continue
            for b in ev.targets(ward):
                meds = ward.monitors[b].meds.meds
                if any(m["name"].lower() == ev.med for m in meds):
                    out.setdefault(b, set()).add(ev.med)
        return out

    def apply(self, ward, now, dt=1.0):
        ward.partitioned = any(
            ev.type == "partition" and ev.active(now) for ev in self.events
        )
        for ev in self.events:
            if ev.type == "discharge":
                if id(ev) not in self._discharged and now >= ev.at:
                    self._discharged.add(id(ev))
                    for b in ev.targets(ward):
                        # Through Ward.discharge, so a full cleaning
                        # roster queues the discharge like any other
                        # instead of starting cleaning at once.
                        if ward.monitors[b].admitted:
                            ward.discharge(b)
                continue
            if not ev.active(now):
                continue
            if ev.type == "sensor_drift":
                # The bias grows by rate * dt every tick the event
                # is active, so the drift is a ramp, not a step —
                # and it keeps growing for the whole window, which
                # is what makes the NEWS2 score creep instead of
                # jumping. The monitor holds the accumulated bias
                # and reports the drifted values; the bias itself
                # is pure arithmetic on dt, no RNG, so a seeded
                # run reproduces it exactly.
                for b in ev.targets(ward):
                    ward.monitors[b].apply_drift(ev.drift, dt)
            elif ev.type == "dropout":
                for b in ev.targets(ward):
                    ward.monitors[b].offline_until = ev.end
            elif ev.type == "stale":
                for b in ev.targets(ward):
                    ward.monitors[b].stale_until = ev.end
            elif ev.type == "alarm_storm":
                for b in ev.targets(ward):
                    ward.monitors[b].alarm_until = ev.end
