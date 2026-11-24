"""Chaos events: device dropout, alarm storms, stale readings, partitions,
discharges, pharmacy stockouts, slow sensor drift, staff shortages, bed
blockers, oxygen outages, power outages."""

from .staffing import ROLES
from .vitals import NORMAL, PLAUSIBLE

VALID_TYPES = ("dropout", "alarm_storm", "stale", "partition", "discharge", "stockout", "sensor_drift", "staff_shortage", "bed_blocker", "oxygen_outage", "power_outage")

# Why a bed is blocked: awaiting a porter to move the patient,
# an isolation clean after an infectious patient, or an equipment
# fault on the bed itself.
BLOCK_REASONS = ("porter", "isolation_clean", "equipment_fault")


def validate_power(power):
    """A power-outage config: ``{"battery_s": ..., "reboot_s": ...}``.

    ``battery_s`` is required: how many seconds of battery backup
    a bedside monitor has once the mains fails — a positive finite
    number. While the battery holds the monitor keeps emitting
    readings, flagged ``on_battery``; once it dies the monitor goes
    dark until power returns. ``reboot_s`` is how many seconds a
    monitor that went dark takes to come back once power returns
    (default 0 — it comes straight back) — a non-negative finite
    number. A config with no positive battery backup does nothing
    at all, so it is rejected.
    """
    if not isinstance(power, dict):
        raise ValueError(
            "power_outage needs a power config, got %r" % (power,)
        )
    battery_s = power.get("battery_s")
    if (
        not isinstance(battery_s, (int, float))
        or isinstance(battery_s, bool)
        or battery_s != battery_s
        or battery_s in (float("inf"), float("-inf"))
        or battery_s <= 0
    ):
        raise ValueError(
            "power_outage: battery_s must be a positive finite "
            "number, got %r" % (battery_s,)
        )
    reboot_s = power.get("reboot_s", 0)
    if (
        not isinstance(reboot_s, (int, float))
        or isinstance(reboot_s, bool)
        or reboot_s != reboot_s
        or reboot_s in (float("inf"), float("-inf"))
        or reboot_s < 0
    ):
        raise ValueError(
            "power_outage: reboot_s must be a non-negative finite "
            "number, got %r" % (reboot_s,)
        )
    return {
        "battery_s": float(battery_s),
        "reboot_s": float(reboot_s),
    }


def validate_oxygen_outage(oxygen):
    """An oxygen-outage config: ``{"rate": ..., "floor": ..., "recovery_rate": ...}``.

    ``rate`` is required: how many SpO2 points a patient on oxygen
    loses per second while the supply is down — a positive finite
    number. ``floor`` is the lowest SpO2 the outage can push a
    patient to (default 82.0, inside the plausible SpO2 envelope).
    ``recovery_rate`` is how many points per second the patient
    regains once the supply returns (default ``rate``) — a
    positive finite number. A config with no positive fall rate
    does nothing at all, so it is rejected.
    """
    if not isinstance(oxygen, dict):
        raise ValueError(
            "oxygen_outage needs an oxygen config, got %r" % (oxygen,)
        )
    rate = oxygen.get("rate")
    if (
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or rate != rate
        or rate in (float("inf"), float("-inf"))
        or rate <= 0
    ):
        raise ValueError(
            "oxygen_outage: rate must be a positive finite number, got %r"
            % (rate,)
        )
    floor = oxygen.get("floor", 82.0)
    if (
        not isinstance(floor, (int, float))
        or isinstance(floor, bool)
        or floor != floor
        or floor in (float("inf"), float("-inf"))
        or not PLAUSIBLE["spo2"][0] <= floor <= PLAUSIBLE["spo2"][1]
    ):
        raise ValueError(
            "oxygen_outage: floor must be within the plausible SpO2 "
            "range %r, got %r" % (PLAUSIBLE["spo2"], floor)
        )
    recovery_rate = oxygen.get("recovery_rate", rate)
    if (
        not isinstance(recovery_rate, (int, float))
        or isinstance(recovery_rate, bool)
        or recovery_rate != recovery_rate
        or recovery_rate in (float("inf"), float("-inf"))
        or recovery_rate <= 0
    ):
        raise ValueError(
            "oxygen_outage: recovery_rate must be a positive finite "
            "number, got %r" % (recovery_rate,)
        )
    return {
        "rate": float(rate),
        "floor": float(floor),
        "recovery_rate": float(recovery_rate),
    }


def validate_block_reason(reason):
    """A bed-blocker reason: one of ``porter``, ``isolation_clean``,
    or ``equipment_fault`` — what is keeping the bed out of use.
    """
    if reason not in BLOCK_REASONS:
        raise ValueError(
            "bed_blocker reason must be one of 'porter', "
            "'isolation_clean', 'equipment_fault', got %r" % (reason,)
        )
    return reason


def validate_shortage(shortage):
    """A staff-shortage config: ``{role: count_lost, ...}``.

    Each role must be one the ward rosters, each count a
    non-negative int — how many of that role call in sick for
    the event's window — and at least one count must be
    non-zero, or the event does nothing at all.
    """
    if not isinstance(shortage, dict) or not shortage:
        raise ValueError(
            "staff_shortage needs a shortage map, got %r" % (shortage,)
        )
    out = {}
    for role, count in shortage.items():
        if role not in ROLES:
            raise ValueError(
                "staff_shortage: unknown staff role: %r" % (role,)
            )
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(
                "staff_shortage: count for %r must be a non-negative int, got %r"
                % (role, count)
            )
        if count != 0:
            out[role] = count
    if not out:
        raise ValueError("staff_shortage needs at least one non-zero count")
    return out


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
    ``staff_shortage`` needs ``shortage`` (a map of staff role to
    how many of that role are lost, see :func:`validate_shortage`):
    while it is active, that many staff are off the effective
    roster — a nurse calling in sick mid-shift — so the
    nurse-to-patient ratio rises and alarms take longer to
    acknowledge and vitals checks slip, exactly as a thin roster
    would. It is ward-wide like ``partition`` and ignores
    ``bed``/``beds``. Only lost nurses change the modelled care:
    a lost doctor or tech is recorded in the staffing summary
    but slows nothing, because the model keys response times
    off nurses alone. ``bed_blocker`` needs ``reason`` (see
    :func:`validate_block_reason`): while it is active, the
    target beds are blocked — a bed awaiting a porter, an
    isolation clean, or with an equipment fault — so they cannot
    take a new admission and their patient cannot be discharged
    until the block clears. A discharge attempted while blocked
    is held and goes through the moment the block ends, with the
    wait recorded as a discharge delay, exactly like a discharge
    held behind a full cleaning roster. ``oxygen_outage`` needs
    ``oxygen`` (see :func:`validate_oxygen_outage`): while it is
    active, a patient on oxygen loses supplemental oxygen, so
    their true SpO2 falls by ``rate`` points per second down to
    ``floor`` — a real desaturation, not a sensor error, so the
    low_spo2 alarm, the NEWS2 score, trend alerts, and
    escalations all respond exactly as they would to a patient
    genuinely deteriorating. Patients not on oxygen are
    untouched: the outage is a supply failure, and only the
    patients the supply was feeding feel it. When the supply
    returns the patient recovers at ``recovery_rate`` points per
    second. Without ``bed``/``beds`` it is ward-wide, like
    ``partition``. ``power_outage`` needs ``power`` (see
    :func:`validate_power`): while it is active, the mains is
    down, so every target monitor runs on battery backup for
    ``battery_s`` seconds — still emitting readings, flagged
    ``on_battery`` — and then goes dark: no readings at all,
    alarms neither raise nor clear, exactly like a monitor that
    dropped out. When power returns, a monitor that went dark
    takes ``reboot_s`` seconds to come back; one that never went
    dark is back on mains at once. The patient is untouched —
    their vitals keep walking, unseen — because the outage is a
    monitoring failure, not a clinical one. Without
    ``bed``/``beds`` it is ward-wide, like ``partition``: the
    mains serves the whole ward.
    """

    def __init__(self, type, at, duration=None, bed=None, beds=None, med=None, drift=None, shortage=None, reason=None, oxygen=None, power=None):
        if type not in VALID_TYPES:
            raise ValueError("unknown chaos type: %r" % type)
        if type == "stockout":
            if not isinstance(med, str) or not med.strip():
                raise ValueError("stockout needs a medication name, got %r" % (med,))
        if type == "sensor_drift":
            drift = validate_drift(drift)
        if type == "staff_shortage":
            shortage = validate_shortage(shortage)
        if type == "bed_blocker":
            reason = validate_block_reason(reason)
        if type == "oxygen_outage":
            oxygen = validate_oxygen_outage(oxygen)
        if type == "power_outage":
            power = validate_power(power)
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
        self.shortage = shortage
        self.reason = reason
        self.oxygen = oxygen
        self.power = power

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
        # Oxygen-outage events fire their start hook once, on the
        # first tick inside the window, and their end hook once,
        # on the first tick after it.
        self._oxygen_started = set()
        self._oxygen_ended = set()
        # Power-outage events fire their start hook once, on the
        # first tick inside the window, and their end hook once,
        # on the first tick after it — a second outage on the
        # same beds is a new event, so it fires again.
        self._power_started = set()
        self._power_ended = set()
        # Sensor-drift events fire their start hook once, on the
        # first tick inside the window, and their end hook once,
        # on the first tick after it — a second drift on the
        # same beds is a new event, so it fires again.
        self._sensor_drift_started = set()
        self._sensor_drift_ended = set()

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

    def shortage(self, ward, now):
        """Staff lost to active staff-shortage events: ``{role: count}``.

        Ward-wide, so bed/beds on the event are ignored, and
        overlapping events add up — two sick calls for one nurse
        each lose two nurses. Only roles with a non-zero loss are
        listed, so an empty dict means the roster is whole.
        """
        out = {}
        for ev in self.events:
            if ev.type != "staff_shortage" or not ev.active(now):
                continue
            for role, count in ev.shortage.items():
                out[role] = out.get(role, 0) + count
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
            # Oxygen outage: the start hook fires once, on the
            # first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario code
            # hears the supply fail and return exactly once per
            # event, even across a run that ticks past the end.
            # The check runs before the active gate below, so an
            # end is caught on the tick the window closes.
            if ev.type == "oxygen_outage":
                if ev.active(now):
                    if id(ev) not in self._oxygen_started:
                        self._oxygen_started.add(id(ev))
                        ward._oxygen_outage_started(ev, now)
                elif (
                    id(ev) in self._oxygen_started
                    and id(ev) not in self._oxygen_ended
                ):
                    self._oxygen_ended.add(id(ev))
                    ward._oxygen_outage_ended(ev, now)
            # Power outage: the start hook fires once, on the
            # first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario code
            # hears the mains fail and return exactly once per
            # event, even across a run that ticks past the end.
            # The check runs before the active gate below, so an
            # end is caught on the tick the window closes.
            if ev.type == "power_outage":
                if ev.active(now):
                    if id(ev) not in self._power_started:
                        self._power_started.add(id(ev))
                        ward._power_outage_started(ev, now)
                elif (
                    id(ev) in self._power_started
                    and id(ev) not in self._power_ended
                ):
                    self._power_ended.add(id(ev))
                    ward._power_outage_ended(ev, now)
            # Sensor drift: the start hook fires once, on the
            # first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario
            # code hears the sensor start miscalibrating and
            # the drift window close exactly once per event,
            # even across a run that ticks past the end. The
            # check runs before the active gate below, so an
            # end is caught on the tick the window closes.
            if ev.type == "sensor_drift":
                if ev.active(now):
                    if id(ev) not in self._sensor_drift_started:
                        self._sensor_drift_started.add(id(ev))
                        ward._sensor_drift_started(ev, now)
                elif (
                    id(ev) in self._sensor_drift_started
                    and id(ev) not in self._sensor_drift_ended
                ):
                    self._sensor_drift_ended.add(id(ev))
                    ward._sensor_drift_ended(ev, now)
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
            elif ev.type == "bed_blocker":
                # The bed is out of use for the window: it cannot
                # take a new admission and its patient cannot be
                # discharged until the block clears. The monitor
                # holds the block end and the reason; the ward
                # releases any discharge held by the block the
                # tick it ends (see Ward._release_blocked_discharges).
                for b in ev.targets(ward):
                    ward.monitors[b].blocked_until = ev.end
                    ward.monitors[b].blocked_reason = ev.reason
            elif ev.type == "oxygen_outage":
                # The oxygen supply is down for the window. The
                # monitor holds the outage end and the config; the
                # deficit itself grows and recovers in the monitor
                # tick (see Monitor._step_oxygen), so a bed that is
                # offline or stale this tick does not desaturate
                # on a reading nobody saw.
                for b in ev.targets(ward):
                    ward.monitors[b].oxygen_outage_until = ev.end
                    ward.monitors[b].oxygen_outage_cfg = ev.oxygen
            elif ev.type == "power_outage":
                # The mains is down for the window. The monitor
                # holds the outage end and the config, and when
                # the outage starts; the battery deadline and the
                # reboot are computed from those in the monitor
                # tick (see Monitor._power_state), so a bed that
                # is offline or stale this tick still runs its
                # battery down on wall-clock time — the UPS does
                # not wait for a reading nobody saw. A fresh
                # outage (no outage currently recorded) restarts
                # the battery clock and clears the went-dark
                # flag; an overlapping event only extends the
                # window, so the battery keeps draining from
                # when power first failed.
                for b in ev.targets(ward):
                    m = ward.monitors[b]
                    if (
                        m.power_outage_until is None
                        or now >= m.power_outage_until
                    ):
                        m.power_outage_since = ev.at
                        m.power_went_dark = False
                    m.power_outage_until = ev.end
                    m.power_outage_cfg = ev.power
            elif ev.type == "dropout":
                for b in ev.targets(ward):
                    ward.monitors[b].offline_until = ev.end
            elif ev.type == "stale":
                for b in ev.targets(ward):
                    ward.monitors[b].stale_until = ev.end
            elif ev.type == "alarm_storm":
                for b in ev.targets(ward):
                    ward.monitors[b].alarm_until = ev.end
