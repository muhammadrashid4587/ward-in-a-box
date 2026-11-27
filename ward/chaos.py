"""Chaos events: device dropout, alarm storms, stale readings, partitions,
discharges, pharmacy stockouts, slow sensor drift, staff shortages, bed
blockers, oxygen outages, power outages, network outages, medication
stockouts, infusion pump failures, lab-result delays, ventilator failures,
dialysis failures."""

from .staffing import ROLES
from .vitals import NORMAL, PLAUSIBLE

VALID_TYPES = ("dropout", "alarm_storm", "stale", "partition", "discharge", "stockout", "sensor_drift", "staff_shortage", "bed_blocker", "oxygen_outage", "power_outage", "network_outage", "medication_stockout", "pump_failure", "lab_delay", "ventilator_failure", "dialysis_failure")

# Default readings a monitor buffers locally while the network is
# down, per bed, before the oldest drops off — ten minutes at
# 1 Hz, so an ordinary outage loses nothing.
DEFAULT_NETWORK_BACKLOG_CAP = 300

# Why a bed is blocked: awaiting a porter to move the patient,
# an isolation clean after an infectious patient, or an equipment
# fault on the bed itself.
BLOCK_REASONS = ("porter", "isolation_clean", "equipment_fault")


def validate_ventilator(ventilator):
    """A ventilator-failure config: ``{"drift": ..., "recovery_rate": ...}``.

    ``drift`` is required: a map of vital to drift rate per
    second (see :func:`validate_drift`) — how fast the
    patient's true vitals move once their ventilator has
    stopped, because they have lost ventilatory support: a
    ventilator that stops lets the SpO2 fall and the
    respiratory rate climb. ``recovery_rate`` is how fast
    each vital's drift shrinks back toward zero per second
    once the ventilator is running again (default the
    largest absolute drift rate) — a positive finite number.
    A config with no drift does nothing at all, so it is
    rejected. Which patient is affected is not part of the
    config: it is the patient's own ventilator (see
    ``Monitor.on_ventilator``) — the ventilator supports
    whoever is on it, and a bed with no ventilator has no
    ventilator to fail.
    """
    if not isinstance(ventilator, dict):
        raise ValueError(
            "ventilator_failure needs a ventilator config, got %r"
            % (ventilator,)
        )
    drift = validate_drift(ventilator.get("drift"))
    recovery_rate = ventilator.get(
        "recovery_rate", max(abs(r) for r in drift.values())
    )
    if (
        not isinstance(recovery_rate, (int, float))
        or isinstance(recovery_rate, bool)
        or recovery_rate != recovery_rate
        or recovery_rate in (float("inf"), float("-inf"))
        or recovery_rate <= 0
    ):
        raise ValueError(
            "ventilator_failure: recovery_rate must be a positive "
            "finite number, got %r" % (recovery_rate,)
        )
    return {
        "drift": drift,
        "recovery_rate": float(recovery_rate),
    }


def validate_dialysis(dialysis):
    """A dialysis-failure config: ``{"drift": ..., "recovery_rate": ...}``.

    ``drift`` is required: a map of vital to drift rate per
    second (see :func:`validate_drift`) — how fast the
    patient's true vitals move once their dialysis machine
    has stopped, because they have lost renal replacement:
    a dialysis machine that stops lets fluid build up, so
    the blood pressure climbs and the heart rate climbs with
    it. ``recovery_rate`` is how fast each vital's drift
    shrinks back toward zero per second once the machine is
    running again (default the largest absolute drift rate)
    — a positive finite number. A config with no drift does
    nothing at all, so it is rejected. Which patient is
    affected is not part of the config: it is the patient's
    own dialysis (see ``Monitor.on_dialysis``) — the machine
    treats whoever is on it, and a bed with no dialysis has
    no machine to fail.
    """
    if not isinstance(dialysis, dict):
        raise ValueError(
            "dialysis_failure needs a dialysis config, got %r"
            % (dialysis,)
        )
    drift = validate_drift(dialysis.get("drift"))
    recovery_rate = dialysis.get(
        "recovery_rate", max(abs(r) for r in drift.values())
    )
    if (
        not isinstance(recovery_rate, (int, float))
        or isinstance(recovery_rate, bool)
        or recovery_rate != recovery_rate
        or recovery_rate in (float("inf"), float("-inf"))
        or recovery_rate <= 0
    ):
        raise ValueError(
            "dialysis_failure: recovery_rate must be a positive "
            "finite number, got %r" % (recovery_rate,)
        )
    return {
        "drift": drift,
        "recovery_rate": float(recovery_rate),
    }


def validate_lab_delay(lab):
    """A lab-delay config: ``{"name": ..., "delay_s": ...}``.

    ``name`` is required: the lab result that comes back late —
    a non-empty string, matched case-insensitively against the
    patients' ordered results. ``delay_s`` is required: how many
    seconds late the result lands — a positive finite number. A
    config with no name or no delay does nothing at all, so it
    is rejected. Which results are delayed is not part of the
    timing: a result due inside the event's window is delayed,
    one due outside it is not, and a result that has already
    landed cannot be delayed at all.
    """
    if not isinstance(lab, dict):
        raise ValueError(
            "lab_delay needs a lab config, got %r" % (lab,)
        )
    name = lab.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "lab_delay needs a lab name, got %r" % (name,)
        )
    delay_s = lab.get("delay_s")
    if (
        not isinstance(delay_s, (int, float))
        or isinstance(delay_s, bool)
        or delay_s != delay_s
        or delay_s in (float("inf"), float("-inf"))
        or delay_s <= 0
    ):
        raise ValueError(
            "lab_delay: delay_s must be a positive finite number, "
            "got %r" % (delay_s,)
        )
    return {
        "name": name.strip().lower(),
        "delay_s": float(delay_s),
    }


def validate_medication_stockout(medication):
    """A medication-stockout config: ``{"med": ..., "drift": ..., "recovery_rate": ...}``.

    ``med`` is required: the medication that runs out of stock — a
    non-empty string, matched case-insensitively against the
    patients' schedules. ``drift`` is required: a map of vital to
    drift rate per second (see :func:`validate_drift`) — how fast
    the patient's true vitals move once a dose of that medication
    has actually been missed, because the drug is no longer holding
    them where it was: a missed beta-blocker lets the heart rate
    climb, a missed antihypertensive lets the pressure rise.
    ``recovery_rate`` is how fast each vital's drift shrinks back
    toward zero per second once the stock is back (default the
    largest absolute drift rate) — a positive finite number. A
    config with no medication or no drift does nothing at all, so
    it is rejected.
    """
    if not isinstance(medication, dict):
        raise ValueError(
            "medication_stockout needs a medication config, got %r"
            % (medication,)
        )
    med = medication.get("med")
    if not isinstance(med, str) or not med.strip():
        raise ValueError(
            "medication_stockout needs a medication name, got %r"
            % (med,)
        )
    drift = validate_drift(medication.get("drift"))
    recovery_rate = medication.get(
        "recovery_rate", max(abs(r) for r in drift.values())
    )
    if (
        not isinstance(recovery_rate, (int, float))
        or isinstance(recovery_rate, bool)
        or recovery_rate != recovery_rate
        or recovery_rate in (float("inf"), float("-inf"))
        or recovery_rate <= 0
    ):
        raise ValueError(
            "medication_stockout: recovery_rate must be a positive "
            "finite number, got %r" % (recovery_rate,)
        )
    return {
        "med": med.strip().lower(),
        "drift": drift,
        "recovery_rate": float(recovery_rate),
    }


def validate_pump(pump):
    """An infusion-pump-failure config: ``{"drift": ..., "recovery_rate": ...}``.

    ``drift`` is required: a map of vital to drift rate per
    second (see :func:`validate_drift`) — how fast the
    patient's true vitals move once their infusion has
    stopped, because the drug is no longer holding them where
    it was: a stalled insulin infusion lets the heart rate
    climb, a stalled noradrenaline infusion lets the pressure
    fall. ``recovery_rate`` is how fast each vital's drift
    shrinks back toward zero per second once the pump is
    running again (default the largest absolute drift rate) —
    a positive finite number. A config with no drift does
    nothing at all, so it is rejected. Which drug stops is
    not part of the config: it is the patient's own infusion
    (see ``Monitor.infusion``) — the pump delivers whatever
    the patient is on, and a bed with no infusion running has
    no pump to fail.
    """
    if not isinstance(pump, dict):
        raise ValueError(
            "pump_failure needs a pump config, got %r" % (pump,)
        )
    drift = validate_drift(pump.get("drift"))
    recovery_rate = pump.get(
        "recovery_rate", max(abs(r) for r in drift.values())
    )
    if (
        not isinstance(recovery_rate, (int, float))
        or isinstance(recovery_rate, bool)
        or recovery_rate != recovery_rate
        or recovery_rate in (float("inf"), float("-inf"))
        or recovery_rate <= 0
    ):
        raise ValueError(
            "pump_failure: recovery_rate must be a positive "
            "finite number, got %r" % (recovery_rate,)
        )
    return {
        "drift": drift,
        "recovery_rate": float(recovery_rate),
    }


def validate_network(network):
    """A network-outage config: ``{"backlog_cap": ...}``.

    ``backlog_cap`` is how many readings a monitor buffers
    locally while the link to the central station is down, per
    bed — a positive int (default
    ``DEFAULT_NETWORK_BACKLOG_CAP``). When the buffer is full the
    oldest reading drops off, counted as dropped, so a long
    outage loses readings instead of growing without bound. The
    config is optional: an outage with no config buffers with
    the default cap, because the outage itself — readings taken
    locally but not delivered — is meaningful without one.
    """
    if network is None:
        return {"backlog_cap": DEFAULT_NETWORK_BACKLOG_CAP}
    if not isinstance(network, dict):
        raise ValueError(
            "network_outage needs a network config, got %r" % (network,)
        )
    backlog_cap = network.get(
        "backlog_cap", DEFAULT_NETWORK_BACKLOG_CAP
    )
    if (
        not isinstance(backlog_cap, int)
        or isinstance(backlog_cap, bool)
        or backlog_cap < 1
    ):
        raise ValueError(
            "network_outage: backlog_cap must be a positive int, "
            "got %r" % (backlog_cap,)
        )
    return {"backlog_cap": backlog_cap}


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
    mains serves the whole ward. ``network_outage`` takes an
    optional ``network`` config (see :func:`validate_network`):
    while it is active, the target monitors cannot report to the
    central station / EHR — the readings keep being taken
    locally, flagged ``delivered: False``, and buffered in a
    per-bed backlog capped at ``backlog_cap`` readings (the
    oldest drops off when it fills), and alarms raised while
    down never reach the dashboard — and on recovery the backlog
    syncs at once: the buffered readings and the undelivered
    alarms are delivered, and the backlog empties. Without
    ``bed``/``beds`` it is ward-wide, like ``partition``: the
    link serves the whole ward. ``medication_stockout`` needs
    ``medication`` (see :func:`validate_medication_stockout`):
    a named medication runs out of stock for the window, so
    doses of it that fall due on the target beds are skipped —
    recorded with status ``stockout`` in the dose history,
    counted in the medication metrics, and flagged with a
    ``missed_dose`` alarm, exactly like a plain ``stockout`` —
    and the patients who actually miss a dose drift: their true
    vitals move by the configured ``drift`` rates per second
    while the stock is out, so the alarms, the NEWS2 score,
    trend alerts, and escalations all respond exactly as they
    would to a patient genuinely deteriorating off their drug.
    A patient prescribed the medication but with no dose due
    inside the window misses nothing and never drifts. Once
    the stock returns the drift shrinks back toward zero at
    ``recovery_rate`` per second — the replenishment is the
    recovery, so a patient still waiting on their next dose
    starts recovering when the stock comes back, not when the
    dose is finally given. Without ``bed``/``beds`` it applies
    to every bed that has the medication prescribed, like
    ``stockout``. ``pump_failure`` needs ``pump`` (see
    :func:`validate_pump`): a bedside infusion pump fails for
    the window — an occlusion, a dead battery, a door left
    open — so the infusion it was delivering stops and the
    patient on that bed drifts off their drug: their true
    vitals move by the configured ``drift`` rates per second
    while the pump is down, so the alarms, the NEWS2 score,
    trend alerts, and escalations all respond exactly as they
    would to a patient genuinely deteriorating off their
    infusion, and a ``pump_failure`` alarm raises while the
    pump is down. Only a bed with an infusion running is
    affected — a bed with no infusion has no pump to fail —
    and which drug stops is the patient's own infusion, not
    part of the event. Once the pump is running again the
    drift shrinks back toward zero at ``recovery_rate`` per
    second.
    ``lab_delay`` needs ``lab`` (see
    :func:`validate_lab_delay`): the lab is running slow for
    the window — a backed-up analyser, a porter queue, a lost
    sample — so a result due inside the window lands
    ``delay_s`` seconds late instead of on time. While a result
    is past its due time and still not back, the bed raises a
    ``lab_overdue`` alarm, which clears the tick the result
    lands, and a result still pending is an outstanding order,
    so the patient is not discharge-ready until every result
    is back. A result due outside the window lands on time,
    a result that has already landed cannot be delayed, and a
    bed with no result of that name ordered is untouched.
    One event delays a result once, however long its window
    runs, and overlapping events for the same result add up.
    ``ventilator_failure`` needs ``ventilator`` (see
    :func:`validate_ventilator`): a bedside ventilator fails
    for the window — a circuit disconnect, a failed flow
    sensor, a power fault in the ventilator itself — so the
    patient on that bed loses ventilatory support and drifts:
    their true vitals move by the configured ``drift`` rates
    per second while the ventilator is down, so the alarms,
    the NEWS2 score, trend alerts, and escalations all
    respond exactly as they would to a patient genuinely
    deteriorating off their ventilator, and a
    ``ventilator_failure`` alarm raises while the ventilator
    is down. Only a bed with a ventilator is affected — a bed
    with no ventilator has no ventilator to fail — and which
    patient is on one is the patient's own flag, not part of
    the event. Once the ventilator is running again the drift
    shrinks back toward zero at ``recovery_rate`` per second.
    ``dialysis_failure`` needs ``dialysis`` (see
    :func:`validate_dialysis`): a bedside dialysis machine
    fails for the window — a blood leak, a failed pump, a
    clotted dialyser — so the patient on that bed loses renal
    replacement and drifts: their true vitals move by the
    configured ``drift`` rates per second while the machine
    is down, so the alarms, the NEWS2 score, trend alerts,
    and escalations all respond exactly as they would to a
    patient genuinely deteriorating off their dialysis, and
    a ``dialysis_failure`` alarm raises while the machine
    is down. Only a bed on dialysis is affected — a bed
    with no dialysis has no machine to fail — and which
    patient is on one is the patient's own flag, not part of
    the event. Once the machine is running again the drift
    shrinks back toward zero at ``recovery_rate`` per second.
    """

    def __init__(self, type, at, duration=None, bed=None, beds=None, med=None, drift=None, shortage=None, reason=None, oxygen=None, power=None, network=None, medication=None, pump=None, lab=None, ventilator=None, dialysis=None):
        if type not in VALID_TYPES:
            raise ValueError("unknown chaos type: %r" % type)
        if type == "stockout":
            if not isinstance(med, str) or not med.strip():
                raise ValueError("stockout needs a medication name, got %r" % (med,))
        if type == "medication_stockout":
            medication = validate_medication_stockout(medication)
            med = medication["med"]
        if type == "pump_failure":
            pump = validate_pump(pump)
        if type == "lab_delay":
            lab = validate_lab_delay(lab)
        if type == "ventilator_failure":
            ventilator = validate_ventilator(ventilator)
        if type == "dialysis_failure":
            dialysis = validate_dialysis(dialysis)
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
        if type == "network_outage":
            network = validate_network(network)
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
        self.network = network
        self.medication = medication
        self.pump = pump
        self.lab = lab
        self.ventilator = ventilator
        self.dialysis = dialysis

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
        # Bed-blocker events fire their start hook once, on the
        # first tick inside the window, and their end hook once,
        # on the first tick after it — a second block on the
        # same bed is a new event, so it fires again.
        self._bed_blocker_started = set()
        self._bed_blocker_ended = set()
        # Network-outage events fire their start hook once, on
        # the first tick inside the window, and their end hook
        # once, on the first tick after it — a second outage on
        # the same beds is a new event, so it fires again.
        self._network_started = set()
        self._network_ended = set()
        # Medication-stockout events fire their start hook once,
        # on the first tick inside the window, and their end hook
        # once, on the first tick after it — a second stockout
        # on the same beds is a new event, so it fires again.
        self._medication_stockout_started = set()
        self._medication_stockout_ended = set()
        # Pump-failure events fire their start hook once, on the
        # first tick inside the window, and their end hook once,
        # on the first tick after it — a second failure on the
        # same beds is a new event, so it fires again.
        self._pump_failure_started = set()
        self._pump_failure_ended = set()
        # Lab-delay events fire their start hook once, on the
        # first tick inside the window, and their end hook once,
        # on the first tick after it — a second slow lab on the
        # same beds is a new event, so it fires again.
        self._lab_delay_started = set()
        self._lab_delay_ended = set()
        # Ventilator-failure events fire their start hook once,
        # on the first tick inside the window, and their end
        # hook once, on the first tick after it — a second
        # failure on the same beds is a new event, so it fires
        # again.
        self._ventilator_failure_started = set()
        self._ventilator_failure_ended = set()
        # Dialysis-failure events fire their start hook once,
        # on the first tick inside the window, and their end
        # hook once, on the first tick after it — a second
        # failure on the same beds is a new event, so it fires
        # again.
        self._dialysis_failure_started = set()
        self._dialysis_failure_ended = set()
        # Staff-shortage events fire their start hook once, on
        # the first tick inside the window, and their end hook
        # once, on the first tick after it — a second sick call
        # is a new event, so it fires again.
        self._staff_shortage_started = set()
        self._staff_shortage_ended = set()

    def stockouts(self, ward, now):
        """Active pharmacy stockouts: ``{bed_id: {med, ...}}``.

        Only beds that actually have the stocked-out medication in
        their schedule are listed, so a ward-wide stockout of insulin
        does not flag beds with no insulin prescribed. A
        medication-stockout event stocks out its medication the same
        way — the dose skipping is identical, the drift is what sets
        it apart — so both types feed this map.
        """
        out = {}
        for ev in self.events:
            if ev.type not in ("stockout", "medication_stockout"):
                continue
            if not ev.active(now):
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
            # Bed blocker: the start hook fires once, on the
            # first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario
            # code hears the bed go out of use and come back
            # exactly once per event, even across a run that
            # ticks past the end. The check runs before the
            # active gate below, so an end is caught on the
            # tick the window closes.
            if ev.type == "bed_blocker":
                if ev.active(now):
                    if id(ev) not in self._bed_blocker_started:
                        self._bed_blocker_started.add(id(ev))
                        ward._bed_blocker_started(ev, now)
                elif (
                    id(ev) in self._bed_blocker_started
                    and id(ev) not in self._bed_blocker_ended
                ):
                    self._bed_blocker_ended.add(id(ev))
                    ward._bed_blocker_ended(ev, now)
            # Network outage: the start hook fires once, on the
            # first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario
            # code hears the link fail and return exactly once
            # per event, even across a run that ticks past the
            # end. The check runs before the active gate below,
            # so an end is caught on the tick the window
            # closes.
            if ev.type == "network_outage":
                if ev.active(now):
                    if id(ev) not in self._network_started:
                        self._network_started.add(id(ev))
                        ward._network_outage_started(ev, now)
                elif (
                    id(ev) in self._network_started
                    and id(ev) not in self._network_ended
                ):
                    self._network_ended.add(id(ev))
                    ward._network_outage_ended(ev, now)
            # Medication stockout: the start hook fires once,
            # on the first tick inside the window, and the end
            # hook once, on the first tick after it — so
            # scenario code hears the pharmacy run out of a
            # medication and the stock return exactly once per
            # event, even across a run that ticks past the end.
            # The check runs before the active gate below, so
            # an end is caught on the tick the window closes.
            if ev.type == "medication_stockout":
                if ev.active(now):
                    if id(ev) not in self._medication_stockout_started:
                        self._medication_stockout_started.add(id(ev))
                        ward._medication_stockout_started(ev, now)
                elif (
                    id(ev) in self._medication_stockout_started
                    and id(ev) not in self._medication_stockout_ended
                ):
                    self._medication_stockout_ended.add(id(ev))
                    ward._medication_stockout_ended(ev, now)
            # Pump failure: the start hook fires once, on the
            # first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario
            # code hears the pump fail and restart exactly once
            # per event, even across a run that ticks past the
            # end. The check runs before the active gate below,
            # so an end is caught on the tick the window closes.
            if ev.type == "pump_failure":
                if ev.active(now):
                    if id(ev) not in self._pump_failure_started:
                        self._pump_failure_started.add(id(ev))
                        ward._pump_failure_started(ev, now)
                elif (
                    id(ev) in self._pump_failure_started
                    and id(ev) not in self._pump_failure_ended
                ):
                    self._pump_failure_ended.add(id(ev))
                    ward._pump_failure_ended(ev, now)
            # Lab delay: the start hook fires once, on the
            # first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario
            # code hears the lab run slow and return to normal
            # exactly once per event, even across a run that
            # ticks past the end. The check runs before the
            # active gate below, so an end is caught on the
            # tick the window closes.
            if ev.type == "lab_delay":
                if ev.active(now):
                    if id(ev) not in self._lab_delay_started:
                        self._lab_delay_started.add(id(ev))
                        ward._lab_delay_started(ev, now)
                elif (
                    id(ev) in self._lab_delay_started
                    and id(ev) not in self._lab_delay_ended
                ):
                    self._lab_delay_ended.add(id(ev))
                    ward._lab_delay_ended(ev, now)
            # Ventilator failure: the start hook fires once, on
            # the first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario
            # code hears the ventilator fail and restart exactly
            # once per event, even across a run that ticks past
            # the end. The check runs before the active gate
            # below, so an end is caught on the tick the window
            # closes.
            if ev.type == "ventilator_failure":
                if ev.active(now):
                    if id(ev) not in self._ventilator_failure_started:
                        self._ventilator_failure_started.add(id(ev))
                        ward._ventilator_failure_started(ev, now)
                elif (
                    id(ev) in self._ventilator_failure_started
                    and id(ev) not in self._ventilator_failure_ended
                ):
                    self._ventilator_failure_ended.add(id(ev))
                    ward._ventilator_failure_ended(ev, now)
            # Dialysis failure: the start hook fires once, on
            # the first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario
            # code hears the dialysis machine fail and restart
            # exactly once per event, even across a run that
            # ticks past the end. The check runs before the
            # active gate below, so an end is caught on the
            # tick the window closes.
            if ev.type == "dialysis_failure":
                if ev.active(now):
                    if id(ev) not in self._dialysis_failure_started:
                        self._dialysis_failure_started.add(id(ev))
                        ward._dialysis_failure_started(ev, now)
                elif (
                    id(ev) in self._dialysis_failure_started
                    and id(ev) not in self._dialysis_failure_ended
                ):
                    self._dialysis_failure_ended.add(id(ev))
                    ward._dialysis_failure_ended(ev, now)
            # Staff shortage: the start hook fires once, on the
            # first tick inside the window, and the end hook
            # once, on the first tick after it — so scenario
            # code hears the sick call and the roster coming
            # whole again exactly once per event, even across
            # a run that ticks past the end. The check runs
            # before the active gate below, so an end is
            # caught on the tick the window closes.
            if ev.type == "staff_shortage":
                if ev.active(now):
                    if id(ev) not in self._staff_shortage_started:
                        self._staff_shortage_started.add(id(ev))
                        ward._staff_shortage_started(ev, now)
                elif (
                    id(ev) in self._staff_shortage_started
                    and id(ev) not in self._staff_shortage_ended
                ):
                    self._staff_shortage_ended.add(id(ev))
                    ward._staff_shortage_ended(ev, now)
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
            elif ev.type == "network_outage":
                # The link to the central station is down for the
                # window. The monitor holds the outage end and
                # the config; the buffering and the sync happen
                # in the monitor tick (see Monitor._buffer_reading
                # and Monitor._sync_network_backlog), so a bed
                # that is offline or stale this tick buffers
                # nothing — there is no fresh reading to hold —
                # while its backlog still syncs when the link
                # returns. An overlapping event only extends the
                # window, so the backlog keeps growing from when
                # the link first failed.
                for b in ev.targets(ward):
                    m = ward.monitors[b]
                    m.network_outage_until = ev.end
                    m.network_outage_cfg = ev.network
            elif ev.type == "medication_stockout":
                # The pharmacy is out of this medication for the
                # window. The monitor holds the outage end and
                # the config; the drift itself grows and recovers
                # in the monitor tick (see
                # Monitor._step_med_stockout), gated on the
                # patient actually having missed a dose of that
                # medication, so a bed with no dose due inside
                # the window never drifts. An overlapping event
                # only extends the window and refreshes the
                # config, like the network outage above.
                for b in ev.targets(ward):
                    m = ward.monitors[b]
                    m.med_stockout_until = ev.end
                    m.med_stockout_cfg = ev.medication
            elif ev.type == "pump_failure":
                # The bedside infusion pump has failed for the
                # window. The monitor holds the failure end and
                # the config; the drift itself grows and recovers
                # in the monitor tick (see
                # Monitor._step_pump_failure), gated on the bed
                # actually having an infusion running, so a bed
                # with no infusion never drifts. An overlapping
                # event only extends the window and refreshes
                # the config, like the medication stockout
                # above.
                for b in ev.targets(ward):
                    m = ward.monitors[b]
                    m.pump_failure_until = ev.end
                    m.pump_failure_cfg = ev.pump
            elif ev.type == "lab_delay":
                # The lab is running slow for the window. A
                # result due inside it lands late instead of on
                # time; the monitor's schedule holds the pushed
                # landing time (see LabSchedule.apply_delay),
                # gated on the bed actually having a result of
                # that name due inside the window, so a bed with
                # no such result is untouched. The delay is
                # applied once per event — the schedule
                # remembers which events already delayed each
                # result — so a long window does not keep
                # pushing the same result out tick after tick.
                for b in ev.targets(ward):
                    ward.monitors[b].apply_lab_delay(ev)
            elif ev.type == "ventilator_failure":
                # The bedside ventilator has failed for the
                # window. The monitor holds the failure end and
                # the config; the drift itself grows and recovers
                # in the monitor tick (see
                # Monitor._step_ventilator_failure), gated on
                # the bed actually having a ventilator, so a bed
                # with no ventilator never drifts. An overlapping
                # event only extends the window and refreshes
                # the config, like the pump failure above.
                for b in ev.targets(ward):
                    m = ward.monitors[b]
                    m.ventilator_failure_until = ev.end
                    m.ventilator_failure_cfg = ev.ventilator
            elif ev.type == "dialysis_failure":
                # The bedside dialysis machine has failed for
                # the window. The monitor holds the failure end
                # and the config; the drift itself grows and
                # recovers in the monitor tick (see
                # Monitor._step_dialysis_failure), gated on
                # the bed actually being on dialysis, so a bed
                # not on dialysis never drifts. An overlapping
                # event only extends the window and refreshes
                # the config, like the ventilator failure above.
                for b in ev.targets(ward):
                    m = ward.monitors[b]
                    m.dialysis_failure_until = ev.end
                    m.dialysis_failure_cfg = ev.dialysis
            elif ev.type == "dropout":
                for b in ev.targets(ward):
                    ward.monitors[b].offline_until = ev.end
            elif ev.type == "stale":
                for b in ev.targets(ward):
                    ward.monitors[b].stale_until = ev.end
            elif ev.type == "alarm_storm":
                for b in ev.targets(ward):
                    ward.monitors[b].alarm_until = ev.end
