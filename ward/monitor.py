"""One bed's virtual monitor: emits vitals, obeys chaos state."""

import random
from collections import deque

from .alert_hooks import (
    validate_discharge_readiness_hooks,
    validate_escalation_hooks,
    validate_missed_dose_hooks,
    validate_trend_alert_hooks,
)
from .meds import MedSchedule
from .news2 import (
    ACVPU_LEVELS,
    BASE_RESPONSE_S,
    RESPONSE,
    score_signs,
    trend_for,
    validate_consciousness,
    validate_consciousness_dynamics,
    validate_news2_trend,
)
from .vitals import PLAUSIBLE, alarms_for, initial_signs, step, validate_acuity

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

# Maximum NEWS2 score entries kept per bed. Older scores drop off
# the history list; the run summary still counts every tick scored.
MAX_NEWS2_HISTORY = 1000


class Monitor:
    """A bedside monitor for one bed.

    ``tick(now)`` advances the monitor one step and returns the new
    reading, or ``None`` while the monitor is dropped out. Chaos state
    is set by the chaos engine as ``*_until`` timestamps (elapsed
    seconds); the monitor clears each one when its time passes.
    """

    def __init__(self, bed_id, profile="stable", rng=None, acuity=3, meds=None, on_oxygen=False, consciousness="alert", consciousness_dynamics=None, news2_trend=None, trend_alert_hooks=None, escalation_hooks=None, missed_dose_hooks=None, discharge_readiness_hooks=None):
        self.bed_id = bed_id
        self.profile = profile
        self.acuity = validate_acuity(acuity)
        self.on_oxygen = self._validate_on_oxygen(on_oxygen)
        self.consciousness = validate_consciousness(consciousness)
        self.consciousness_dynamics = validate_consciousness_dynamics(
            consciousness_dynamics
        )
        self.news2_trend = validate_news2_trend(news2_trend)
        # Trend-alert hooks: the ward's configured webhook and/or
        # in-process callback, invoked when a trend alert fires.
        # Off by default — no hooks, no invocation, no behaviour
        # change. Shared across the ward's monitors, so the ward
        # totals count every delivery.
        self.trend_alert_hooks = validate_trend_alert_hooks(trend_alert_hooks)
        # Escalation hooks: the ward's configured webhook and/or
        # in-process callback, invoked when a deterioration
        # episode opens — the patient's NEWS2 risk leaving low.
        # Same machinery as the trend-alert hooks, off by default.
        # Shared across the ward's monitors, so the ward totals
        # count every delivery.
        self.escalation_hooks = validate_escalation_hooks(escalation_hooks)
        # Missed-dose hooks: the ward's configured webhook and/or
        # in-process callback, invoked when one of this bed's doses
        # is missed or skipped for a stockout. Same machinery as
        # the trend-alert hooks, off by default. Shared across the
        # ward's monitors, so the ward totals count every delivery.
        # The schedule owns the object from here on.
        self.missed_dose_hooks = validate_missed_dose_hooks(
            missed_dose_hooks
        )
        # Discharge-readiness hooks: the ward's configured webhook
        # and/or in-process callback, invoked when this bed's
        # readiness flag flips — to ready or back to not-ready.
        # Same machinery as the trend-alert hooks, off by default.
        # Shared across the ward's monitors, so the ward totals
        # count every delivery.
        self.discharge_readiness_hooks = (
            validate_discharge_readiness_hooks(discharge_readiness_hooks)
        )
        # Every consciousness change this stay, as
        # {"from": level, "to": level, "t": seconds}; empty while
        # dynamics are off or the level has never moved.
        self.consciousness_history = []
        # Consecutive seconds the vitals-only risk has held in one
        # direction; each resets when the risk flips.
        self._deteriorating_s = 0.0
        self._recovering_s = 0.0
        self.meds = MedSchedule(
            bed_id, meds, missed_dose_hooks=self.missed_dose_hooks
        )
        self.rng = rng if rng is not None else random.Random()
        self.signs = initial_signs(profile, self.rng, self.acuity)
        # Slow sensor drift: the accumulated bias per vital, in the
        # vital's own units, grown by the chaos engine while a
        # sensor_drift event is active (see :meth:`apply_drift`).
        # Only vitals with a non-zero bias are listed. The bias is
        # a property of the sensor, not the patient: it is added to
        # the reading the monitor reports, so the NEWS2 score and
        # the alarms see the drifted values, while ``self.signs``
        # keeps the patient's true vitals and the random walk is
        # never fed the drift — a miscalibrated sensor does not
        # change the patient, only what the ward sees.
        self.sensor_drift = {}
        # Bed blocker: a bed out of use while ``blocked_until``
        # (elapsed seconds) has not passed, for ``blocked_reason``
        # (``porter``, ``isolation_clean``, or ``equipment_fault``).
        # While blocked the bed cannot take a new admission and
        # its patient cannot be discharged; a discharge attempted
        # while blocked is held in ``blocked_discharge_pending``
        # from ``blocked_discharge_since`` (when it was first
        # attempted, or the expected time for a planned one) until
        # the block clears — see :meth:`mark_blocked_discharge`.
        self.blocked_until = None
        self.blocked_reason = None
        self.blocked_discharge_pending = False
        self.blocked_discharge_since = None
        # Oxygen outage: how many SpO2 points this patient has
        # lost to a failed oxygen supply, while
        # ``oxygen_outage_until`` (elapsed seconds) has not passed
        # and ``oxygen_outage_cfg`` carries the event's rate,
        # floor, and recovery rate. Unlike sensor drift this is a
        # real desaturation — the patient is genuinely losing
        # saturation, not mis-measured — so it feeds the alarms
        # and the NEWS2 score exactly like a falling SpO2 would,
        # and it only ever touches a patient on oxygen: a patient
        # not on oxygen has no supply to lose. The deficit
        # recovers once the supply returns and is cleared on a
        # new admission, like the drift bias.
        self.oxygen_deficit = 0.0
        self.oxygen_outage_until = None
        self.oxygen_outage_cfg = None
        # Whether the deficit has reached the outage's floor this
        # stay: set the tick the deficit first hits the cap (the
        # walked SpO2 minus the floor), so the ward can fire one
        # floor-hit hook per patient per outage, and cleared when
        # the deficit recovers to zero, so a later outage can fire
        # again. The ward consumes the flag and remembers which
        # beds it already fired for; the monitor only reports
        # that the floor was reached.
        self.oxygen_floor_hit = False
        # Power outage: the mains failed at ``power_outage_since``
        # (elapsed seconds) and returns at
        # ``power_outage_until``, with ``power_outage_cfg``
        # carrying the event's battery backup and reboot delay.
        # While the battery holds the monitor keeps emitting
        # readings, flagged ``on_battery``; once the battery
        # dies the monitor goes dark — no readings, alarms
        # neither raise nor clear — until power returns and it
        # has rebooted. ``power_went_dark`` remembers that the
        # battery died during this outage, so a monitor that
        # went dark waits out its reboot even after the mains
        # returns; ``power_dark_ticks`` counts every tick this
        # stay the monitor was dark. The battery is a property
        # of the monitor, not the patient: it drains on
        # wall-clock time from when power first failed, and a
        # new admission clears the dark flag but not the
        # deadline — the UPS does not recharge because the
        # patient changed.
        self.power_outage_until = None
        self.power_outage_cfg = None
        self.power_outage_since = None
        self.power_went_dark = False
        self.power_dark_ticks = 0
        self.seq = 0
        # False until the ward admits this bed (scenario timeline); an
        # unadmitted bed emits no readings at all.
        self.admitted = True
        self.offline_until = None
        self.stale_until = None
        self.alarm_until = None
        self.last_reading = None
        self.alarm_history = deque(maxlen=MAX_ALARM_HISTORY)
        # Every NEWS2 score this stay, in tick order, as
        # {"t": seconds, "score": int, "risk": str, "red_score": bool,
        # "spo2_scale": 1|2, "components": {parameter: points, ...},
        # "consciousness": level}. One entry per fresh reading — a
        # stale tick repeats the previous score, so it records
        # nothing — capped at MAX_NEWS2_HISTORY, oldest first out.
        self.news2_history = deque(maxlen=MAX_NEWS2_HISTORY)
        self.total_raised = 0
        self._active_alarms = set()
        # Bed turnover: after a discharge the bed is cleaning until
        # ``cleaning_until`` (elapsed seconds), then available again.
        # ``turnovers`` counts discharges; ``total_cleaning_s`` is the
        # total time this bed has spent being cleaned.
        self.cleaning_until = None
        self.turnovers = 0
        self.total_cleaning_s = 0.0
        # A discharge ordered while every cleaning slot is busy does
        # not start cleaning at once: the patient stays in the bed
        # (still monitored) until a slot frees up. ``pending_since``
        # is when the discharge was first attempted — the expected
        # discharge time — so the delay is measurable; the totals
        # below accumulate it across discharges.
        self.awaiting_cleaning = False
        self.pending_since = None
        self.discharge_delays = 0
        self.total_discharge_delay_s = 0.0
        # Alarms raised by the most recent tick; the ward reads this to
        # hand them to the staffing model for acknowledgement.
        self.raised_this_tick = []
        # Discharge readiness: vitals count as stable from
        # ``_stable_since`` — the last time the patient's alarms
        # cleared, or admission when they never alarmed. None while
        # an alarm is active.
        self._stable_since = 0.0
        # Readiness flips this stay, in tick order: one event per
        # transition of the discharge-ready flag, to ready or back
        # to not-ready (see :meth:`note_readiness`). ``_ready`` is
        # the flag as of the last tick the ward noted; it resets
        # silently on discharge, so a new patient in the bed
        # starts from not-ready.
        self._ready = False
        self.readiness_history = []
        # Early warning: a deterioration episode is open while the
        # patient's NEWS2 risk is medium or high — raised at the
        # first tick the risk leaves low, closed when it returns.
        # ``escalation_history`` keeps every closed episode; the
        # open one lives in ``_escalation`` until it closes. Each
        # episode also measures the response: the team NEWS2 calls
        # for at the risk it raised at is due ``BASE_RESPONSE_S``
        # times the staffing load factor later, and the episode
        # records when the response landed (``responded_t``) and
        # how long it took (``time_to_respond_s``) — or neither,
        # when the episode closed before anyone responded.
        self._escalation = None
        self.escalation_history = []
        self.escalations_raised = 0
        # NEWS2 trend alerts: a trend alert is open while the
        # patient's score is rising by the configured rules — a
        # jump of ``delta`` within ``window_s`` seconds, or a
        # sustained upward creep over the last ``trend_ticks``
        # ticks. ``trend_alert_history`` keeps every closed alert;
        # the open one lives in ``_trend_alert`` until it closes.
        self._trend_alert = None
        self.trend_alert_history = []
        self.trend_alerts_raised = 0

    @staticmethod
    def _validate_on_oxygen(on_oxygen):
        if not isinstance(on_oxygen, bool):
            raise ValueError(
                "on_oxygen must be a bool, got %r" % (on_oxygen,)
            )
        return on_oxygen

    def is_blocked(self, now):
        """True while a bed-blocker event holds this bed: it cannot
        take a new admission and its patient cannot be discharged
        until the block clears."""
        return self.blocked_until is not None and now < self.blocked_until

    def mark_blocked_discharge(self, now, expected_t=None):
        """Hold a discharge that was attempted while the bed is
        blocked: the patient stays in the bed until the block
        clears, and the wait is measured from ``expected_t`` (a
        planned discharge's expected time) or ``now``.

        Returns True when this is a fresh hold — False when a
        discharge is already held for this bed, so a second
        attempt during the same block does not restart the
        clock or double-count the hold.
        """
        if self.blocked_discharge_pending:
            return False
        self.blocked_discharge_pending = True
        self.blocked_discharge_since = round(
            expected_t if expected_t is not None else now, 3
        )
        return True

    def oxygen_outage_detail(self, now):
        """Per-bed oxygen-outage state as plain JSON data.

        ``outage_active`` is True while an oxygen-outage event
        holds the ward's supply, with when it ends; ``deficit``
        is how many SpO2 points this patient has lost to it
        (zero for a patient not on oxygen, who has no supply to
        lose), and ``desaturating`` is True while any deficit
        remains — including after the outage ends, while the
        patient is still recovering.
        """
        active = (
            self.oxygen_outage_until is not None
            and now < self.oxygen_outage_until
        )
        return {
            "bed": self.bed_id,
            "admitted": self.admitted,
            "on_oxygen": self.on_oxygen,
            "outage_active": active,
            "outage_until": (
                self.oxygen_outage_until if active else None
            ),
            "deficit": self.oxygen_deficit,
            "desaturating": self.oxygen_deficit > 0,
        }

    def _power_state(self, now):
        """The monitor's power state at elapsed time ``now``.

        Pure computation, no side effects: ``"mains"`` while the
        mains is up, ``"battery"`` while an outage is running and
        the battery still holds, ``"dark"`` once the battery has
        died (or while the mains is back but the monitor has not
        rebooted yet — that is ``"rebooting"``). ``battery_until``
        is when the battery dies, ``reboot_until`` when a dark
        monitor comes back. A bed with no patient has no monitor
        running, so it is never on battery or dark even while the
        mains is down — though the outage itself is still active
        for the ward.
        """
        if (
            self.power_outage_until is None
            or self.power_outage_cfg is None
            or self.power_outage_since is None
        ):
            return {
                "state": "mains",
                "outage_active": False,
                "outage_until": None,
                "battery_until": None,
                "reboot_until": None,
            }
        cfg = self.power_outage_cfg
        battery_until = round(
            self.power_outage_since + cfg["battery_s"], 3
        )
        reboot_until = round(
            self.power_outage_until + cfg["reboot_s"], 3
        )
        if now < self.power_outage_until:
            if not self.admitted:
                return {
                    "state": "mains",
                    "outage_active": True,
                    "outage_until": self.power_outage_until,
                    "battery_until": battery_until,
                    "reboot_until": None,
                }
            if self.power_went_dark or now >= battery_until:
                return {
                    "state": "dark",
                    "outage_active": True,
                    "outage_until": self.power_outage_until,
                    "battery_until": battery_until,
                    "reboot_until": None,
                }
            return {
                "state": "battery",
                "outage_active": True,
                "outage_until": self.power_outage_until,
                "battery_until": battery_until,
                "reboot_until": None,
            }
        if self.admitted and self.power_went_dark:
            if now < reboot_until:
                return {
                    "state": "rebooting",
                    "outage_active": False,
                    "outage_until": None,
                    "battery_until": None,
                    "reboot_until": reboot_until,
                }
        return {
            "state": "mains",
            "outage_active": False,
            "outage_until": None,
            "battery_until": None,
            "reboot_until": None,
        }

    def power_dark(self, now):
        """True while this monitor is dark from a power outage:
        its battery died, or it is still rebooting after power
        returned. False on mains, on battery, or with no patient
        in the bed."""
        return self._power_state(now)["state"] in ("dark", "rebooting")

    def power_outage_detail(self, now):
        """Per-bed power-outage state as plain JSON data.

        ``outage_active`` is True while a power-outage event
        holds the ward's mains, with when it returns;
        ``on_battery`` is True while the monitor is running on
        its battery backup, ``dark`` while the battery has died
        or the monitor is still rebooting, ``rebooting`` while
        power is back but the monitor has not come up yet, and
        ``dark_ticks`` counts every tick this stay the monitor
        was dark — so a client can tell a bed that lost its
        monitor from one that is merely on battery, and how long
        the ward has been blind on that bed.
        """
        state = self._power_state(now)
        return {
            "bed": self.bed_id,
            "admitted": self.admitted,
            "outage_active": state["outage_active"],
            "outage_until": state["outage_until"],
            "on_battery": state["state"] == "battery",
            "dark": state["state"] in ("dark", "rebooting"),
            "rebooting": state["state"] == "rebooting",
            "battery_until": state["battery_until"],
            "reboot_until": state["reboot_until"],
            "dark_ticks": self.power_dark_ticks,
        }

    def bed_blocker_detail(self, now):
        """Per-bed bed-blocker state as plain JSON data.

        ``blocked`` is True while a bed-blocker event holds the
        bed, with the reason and when the block ends; ``blocked``
        is False once the window has passed, even before the next
        tick clears the stored fields. ``discharge_pending`` is
        True while a discharge attempted during the block is
        still waiting for the block to clear.
        """
        blocked = self.is_blocked(now)
        return {
            "bed": self.bed_id,
            "admitted": self.admitted,
            "blocked": blocked,
            "reason": self.blocked_reason if blocked else None,
            "blocked_until": self.blocked_until if blocked else None,
            "discharge_pending": self.blocked_discharge_pending,
        }

    def state(self, now):
        """Bed state at elapsed time ``now``.

        ``"occupied"`` while a patient is admitted, ``"cleaning"`` after
        a discharge until the turnover period ends, ``"blocked"`` while
        a bed-blocker event holds an otherwise free bed, ``"available"``
        once the bed can take a new admission.
        """
        if self.admitted:
            return "occupied"
        if self.cleaning_until is not None and now < self.cleaning_until:
            return "cleaning"
        if self.is_blocked(now):
            return "blocked"
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
        if self.awaiting_cleaning:
            raise ValueError(
                "bed %d discharge is already pending" % self.bed_id
            )
        self.begin_cleaning(now, cleaning_s)

    def queue_discharge(self, now, expected_t=None):
        """Hold a discharge until a cleaning slot frees up.

        The patient stays in the bed — admitted, still monitored —
        and the bed starts cleaning when the ward calls
        :meth:`begin_cleaning`. ``expected_t`` is when the discharge
        was due (a planned discharge's expected time, or ``now`` for
        an immediate one); the delay is measured from it.
        """
        if not self.admitted:
            raise ValueError("bed %d is not occupied" % self.bed_id)
        if self.awaiting_cleaning:
            return
        self.awaiting_cleaning = True
        # A discharge that was held by a bed blocker and is now
        # queued behind a full cleaning roster keeps its original
        # clock: the delay is measured from when the discharge
        # was first attempted (or its expected time), not from
        # when the block cleared, so the blocked wait is not lost.
        if self.blocked_discharge_since is not None:
            self.pending_since = round(self.blocked_discharge_since, 3)
            self.blocked_discharge_since = None
            self.blocked_discharge_pending = False
        else:
            self.pending_since = round(
                expected_t if expected_t is not None else now, 3
            )

    def begin_cleaning(self, now, cleaning_s):
        """Start turnover for a discharged (or queued) patient.

        Records the discharge delay when one was queued: how long
        past the expected discharge time cleaning actually started.
        A discharge held by a bed blocker counts the same way: the
        wait from when it was first attempted (or its expected
        time, for a planned one) to cleaning starting is a
        discharge delay, so a blocked bed shows up in the same
        delay totals as a bed held behind a full cleaning roster.
        """
        if self.pending_since is not None:
            delay = round(now - self.pending_since, 3)
            if delay > 0:
                self.discharge_delays += 1
                self.total_discharge_delay_s = round(
                    self.total_discharge_delay_s + delay, 3
                )
            self.pending_since = None
        if self.blocked_discharge_since is not None:
            delay = round(now - self.blocked_discharge_since, 3)
            if delay > 0:
                self.discharge_delays += 1
                self.total_discharge_delay_s = round(
                    self.total_discharge_delay_s + delay, 3
                )
            self.blocked_discharge_since = None
        self.blocked_discharge_pending = False
        self.awaiting_cleaning = False
        self._record_alarm_transitions([], now)
        self._close_escalation(now)
        self._close_trend_alert(now)
        # A discharge ends the stay: the readiness flag resets
        # silently — the patient left, they did not stop being
        # ready — so the next patient in the bed starts from
        # not-ready and their first flip is a real one. The flip
        # history stays, like the alarm and escalation histories,
        # so the run totals still count this stay's flips.
        self._ready = False
        self.meds.reset()
        self.admitted = False
        self.cleaning_until = round(now + cleaning_s, 3)
        self.turnovers += 1
        self.total_cleaning_s = round(self.total_cleaning_s + cleaning_s, 3)

    def admit(self, now, profile=None, acuity=None, on_oxygen=None, consciousness=None):
        """Admit a new patient to an available bed.

        Raises ValueError if the bed is occupied, still cleaning,
        or blocked by a bed-blocker event.
        A new admission starts with fresh vitals, so the previous
        patient's readings and alarms do not carry over. The new
        patient is not on oxygen unless ``on_oxygen`` says so —
        the previous patient's oxygen flag never carries over — and
        is Alert unless ``consciousness`` says otherwise: the
        previous patient's consciousness never carries over either.
        """
        state = self.state(now)
        if state == "occupied":
            raise ValueError("bed %d is already occupied" % self.bed_id)
        if state == "cleaning":
            raise ValueError(
                "bed %d is cleaning until t=%s" % (self.bed_id, self.cleaning_until)
            )
        if state == "blocked":
            raise ValueError(
                "bed %d is blocked (%s) until t=%s"
                % (self.bed_id, self.blocked_reason, self.blocked_until)
            )
        if profile is not None:
            self.profile = profile
        if acuity is not None:
            self.acuity = validate_acuity(acuity)
        self.on_oxygen = self._validate_on_oxygen(
            False if on_oxygen is None else on_oxygen
        )
        self.consciousness = validate_consciousness(
            "alert" if consciousness is None else consciousness
        )
        # A new patient starts with no consciousness history and
        # both clocks at zero; the dynamics config itself is the
        # bed's scenario setting and stays across admissions.
        self.consciousness_history = []
        self._deteriorating_s = 0.0
        self._recovering_s = 0.0
        # A new patient starts with no NEWS2 history either: the
        # previous patient's scores do not carry over.
        self.news2_history.clear()
        # A new admission recalibrates the sensor: the drifted
        # bias belonged to the previous patient's stay, and the
        # new patient starts from a clean sensor. The oxygen
        # deficit goes the same way: the new patient starts
        # fully saturated, and an outage still running starts
        # desaturating them again from zero on the next tick.
        # A power outage goes the same way for the dark flag:
        # the new patient starts with a live monitor, and an
        # outage still running starts draining the battery
        # again from when power first failed — the UPS does not
        # recharge because the patient changed, so the deadline
        # is unchanged, only the dark count restarts.
        self.sensor_drift = {}
        self.oxygen_deficit = 0.0
        self.oxygen_floor_hit = False
        self.power_went_dark = False
        self.power_dark_ticks = 0
        self.signs = initial_signs(self.profile, self.rng, self.acuity)
        self.seq = 0
        self.last_reading = None
        self.cleaning_until = None
        self.admitted = True
        self._stable_since = round(now, 3)
        self.meds.reset()

    def meds_tick(self, now, load_factor, stockout_meds=()):
        """Advance the medication schedule (no-op while unadmitted)."""
        if self.admitted:
            self.meds.tick(now, load_factor, stockout_meds)

    def apply_drift(self, rates, dt=1.0):
        """Grow the sensor bias by ``rate * dt`` per vital.

        Called by the chaos engine once per tick while a
        sensor_drift event is active. The bias is pure
        arithmetic — no RNG — so a seeded run reproduces it
        exactly, and it is stored per vital in that vital's own
        units (beats per minute, percent SpO2, and so on). A
        vital whose accumulated bias returns to exactly zero
        drops out of the map, so ``sensor_drift`` only ever
        lists vitals that are actually off. The bias persists
        after the event ends: a drifted sensor stays drifted
        until the bed is re-admitted.
        """
        for vital, rate in rates.items():
            bias = round(self.sensor_drift.get(vital, 0.0) + rate * dt, 3)
            if bias == 0:
                self.sensor_drift.pop(vital, None)
            else:
                self.sensor_drift[vital] = bias

    def _step_oxygen(self, now, dt=1.0):
        """Grow or recover the oxygen deficit for this tick.

        Only a patient on oxygen is affected: while an
        oxygen-outage event is active their SpO2 falls by the
        event's ``rate`` per second, down to its ``floor`` — the
        deficit is capped so the walked SpO2 minus the deficit
        never drops below the floor, and the floor is enforced
        again on the reported value (see
        :meth:`reported_signs`), so a downward wander of the
        walk cannot push the reading under it either. Once the
        supply returns the deficit shrinks by the event's
        ``recovery_rate`` per second until the patient is back
        to their walked saturation. The deficit is pure
        arithmetic on dt — no RNG — so a seeded run reproduces
        it exactly. A patient not on oxygen never grows a
        deficit, and a deficit of zero stays zero.
        """
        if not self.on_oxygen:
            return
        if (
            self.oxygen_outage_until is not None
            and now < self.oxygen_outage_until
            and self.oxygen_outage_cfg is not None
        ):
            cfg = self.oxygen_outage_cfg
            room = max(0.0, self.signs["spo2"] - cfg["floor"])
            grown = self.oxygen_deficit + cfg["rate"] * dt
            self.oxygen_deficit = round(min(grown, room), 3)
            # The floor is reached when the fall would have grown
            # past the whole room above it: the patient's reported
            # SpO2 is now sitting on the floor, and it stays there
            # while the supply is down. The check uses the
            # pre-round growth, so rounding the deficit to three
            # decimals can never hide the hit. A patient whose
            # walked SpO2 is already at or below the floor has no
            # room to fall, so there is no outage-driven hit to
            # report for them.
            if room > 0 and grown >= room:
                self.oxygen_floor_hit = True
        elif self.oxygen_deficit > 0:
            rate = (
                self.oxygen_outage_cfg["recovery_rate"]
                if self.oxygen_outage_cfg is not None
                else 1.0
            )
            self.oxygen_deficit = round(
                max(0.0, self.oxygen_deficit - rate * dt), 3
            )
            if self.oxygen_deficit == 0:
                self.oxygen_floor_hit = False

    def reported_signs(self):
        """The vitals as the sensor reports them: true vitals
        plus the accumulated drift bias, clamped inside the
        plausible envelope so a runaway drift saturates at the
        edge of what a real monitor would ever show. A patient
        desaturating from a failed oxygen supply reports the
        walked SpO2 minus the deficit — a real fall, not a
        sensor error — and the deficit never pushes the reported
        SpO2 below the outage's floor, however far the walk
        wanders; the sensor-drift bias still shows in full
        whenever the deficit leaves room above the floor."""
        out = {}
        for key, value in self.signs.items():
            lo, hi = PLAUSIBLE[key]
            v = value + self.sensor_drift.get(key, 0.0)
            if key == "spo2" and self.oxygen_deficit > 0:
                floor = (
                    self.oxygen_outage_cfg["floor"]
                    if self.oxygen_outage_cfg is not None
                    else lo
                )
                deficit = min(
                    self.oxygen_deficit, max(0.0, v - floor)
                )
                v -= deficit
            out[key] = round(max(lo, min(hi, v)), 2)
        return out

    def sensor_drift_detail(self):
        """Per-bed sensor-drift state as plain JSON data.

        ``drift`` is the accumulated bias per vital (empty while
        the sensor is clean), ``drifting`` is True while any
        vital is off — so a client can tell a clean sensor from
        a drifted one at a glance.
        """
        return {
            "bed": self.bed_id,
            "admitted": self.admitted,
            "drift": dict(self.sensor_drift),
            "drifting": bool(self.sensor_drift),
        }

    def turnover_summary(self, now):
        """Per-bed turnover state and totals as plain JSON data."""
        return {
            "bed": self.bed_id,
            "state": self.state(now),
            "blocked": self.is_blocked(now),
            "blocked_reason": (
                self.blocked_reason if self.is_blocked(now) else None
            ),
            "cleaning_until": self.cleaning_until
            if self.state(now) == "cleaning"
            else None,
            "turnovers": self.turnovers,
            "total_cleaning_s": self.total_cleaning_s,
            "awaiting_cleaning": self.awaiting_cleaning,
            "discharge_expected_t": self.pending_since,
            "discharge_delays": self.discharge_delays,
            "total_discharge_delay_s": self.total_discharge_delay_s,
        }

    def tick(self, now, dt=1.0, load_factor=1.0):
        self.raised_this_tick = []
        if self.cleaning_until is not None and now >= self.cleaning_until:
            self.cleaning_until = None
        if self.blocked_until is not None and now >= self.blocked_until:
            self.blocked_until = None
            self.blocked_reason = None
        if not self.admitted:
            return None
        # A power outage runs the monitor on battery until the
        # battery dies, then the monitor goes dark: no readings,
        # alarms neither raise nor clear, exactly like a dropout
        # — and it stays dark through its reboot once power
        # returns. The battery drains on wall-clock time from
        # when power first failed, so a bed that is offline or
        # stale this tick still runs its battery down. The
        # patient is untouched: their vitals keep walking,
        # unseen, and the walk resumes from where it was when
        # the monitor comes back.
        power = self._power_state(now)
        if power["state"] in ("dark", "rebooting"):
            if power["state"] == "dark" and power["outage_active"]:
                # The battery just died (or died earlier in this
                # outage): remember it, so the monitor waits out
                # its reboot once power returns instead of coming
                # straight back.
                self.power_went_dark = True
            self.power_dark_ticks += 1
            return None
        if power["state"] == "mains" and self.power_outage_until is not None:
            # The outage is over and the reboot is done (or the
            # battery never died): back on mains, clear the
            # outage state.
            self.power_outage_until = None
            self.power_outage_cfg = None
            self.power_outage_since = None
            self.power_went_dark = False
        on_battery = power["state"] == "battery"
        if self.offline_until is not None:
            if now < self.offline_until:
                return None
            self.offline_until = None

        if self.stale_until is not None and now < self.stale_until:
            if self.last_reading is not None:
                stale = dict(self.last_reading)
                stale["stale"] = True
                stale["t"] = round(now, 3)
                stale["on_battery"] = on_battery
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

        # An oxygen outage desaturates a patient on oxygen before
        # the reading is built, so this tick's reading, alarms,
        # and NEWS2 score already carry the fall — and a patient
        # recovering after the supply returns climbs back just
        # as fast. A patient not on oxygen is untouched.
        self._step_oxygen(now, dt)

        # Consciousness moves before the reading is built, so this
        # tick's reading, score, and snapshot already carry the new
        # level — a change is visible in the same tick it happens.
        self._step_consciousness(now, dt)

        self.seq += 1
        # The reading reports the drifted values: the sensor bias
        # is added to the true vitals and clamped inside the
        # plausible envelope, so a runaway drift saturates at the
        # edge of what a real monitor would ever show instead of
        # reporting something impossible. The alarms and the NEWS2
        # score are computed from these reported values — that is
        # the point of the drift: the ward reacts to what the
        # sensor says, and a slowly drifting sensor makes the
        # score creep and the readiness flag flip gradually,
        # exactly as a real miscalibration would.
        reported = self.reported_signs()
        reading = {
            "bed": self.bed_id,
            "seq": self.seq,
            "t": round(now, 3),
            "profile": self.profile,
            "acuity": self.acuity,
            "on_oxygen": self.on_oxygen,
            "consciousness": self.consciousness,
            "stale": False,
            "on_battery": on_battery,
            "sensor_drift": dict(self.sensor_drift),
            "oxygen_deficit": self.oxygen_deficit,
            **reported,
        }
        reading["alarms"] = alarms_for(reported)
        if self.meds.missed_active:
            # A missed dose raises an alarm until that medication's
            # next dose is given.
            reading["alarms"] = reading["alarms"] + ["missed_dose"]
        # NEWS2 early-warning score for this tick's vitals. The
        # score is a parallel track to the alarm list: it never
        # adds or removes an alarm, so alarm counts and readiness
        # are exactly as before — it only records how sick the
        # patient looks on the NEWS2 scale.
        reading["news2"] = score_signs(
            reported, self.on_oxygen, self.consciousness
        )
        self._track_escalation(reading["news2"], now, load_factor)
        # This tick's score joins the bed's history, so a client can
        # see how the patient got here, not just where they are now.
        # Stale ticks never reach this point: they return the
        # previous reading above, so a repeated score is not
        # recorded twice.
        self.news2_history.append(
            {
                "t": round(now, 3),
                "score": reading["news2"]["score"],
                "risk": reading["news2"]["risk"],
                "red_score": reading["news2"]["red_score"],
                "spo2_scale": reading["news2"]["spo2_scale"],
                "components": dict(reading["news2"]["components"]),
                "consciousness": self.consciousness,
            }
        )
        # A rising score raises a trend alert — the score's own
        # rate of change, not a new vital-sign alarm.
        self._track_trend(now)
        self._record_alarm_transitions(reading["alarms"], now)
        self.last_reading = reading
        return reading

    def _step_consciousness(self, now, dt):
        """Move the ACVPU level one step, if it is due.

        The driver is the vitals-only NEWS2 risk — the score with
        consciousness held at Alert — because the level itself
        scores 3 whenever it is not Alert: judging by the total
        risk would make recovery impossible, since a non-Alert
        patient is medium risk on the consciousness red score
        alone. While the vitals risk is medium or high the
        deterioration clock runs and the recovery clock resets;
        while it is low the recovery clock runs and the
        deterioration clock resets. A clock reaching its threshold
        steps the level one ACVPU level in that direction and
        restarts at zero, so a patient slides down one level at a
        time while their vitals stay bad and climbs back one level
        at a time as they recover — never jumping straight from
        Alert to Unresponsive. Deterministic and RNG-free, so
        enabling dynamics never changes the vitals a monitor
        generates: a seeded run with dynamics on emits the same
        signs and alarms as the same run with them off.
        """
        dyn = self.consciousness_dynamics
        if dyn is None:
            return
        vitals_risk = score_signs(self.signs, self.on_oxygen, "alert")["risk"]
        idx = ACVPU_LEVELS.index(self.consciousness)
        if vitals_risk in ("medium", "high"):
            self._recovering_s = 0.0
            if idx >= len(ACVPU_LEVELS) - 1:
                self._deteriorating_s = 0.0
                return
            self._deteriorating_s = round(self._deteriorating_s + dt, 3)
            if self._deteriorating_s >= dyn["deteriorate_after_s"]:
                self._record_consciousness_change(idx + 1, now)
                self._deteriorating_s = 0.0
        else:
            self._deteriorating_s = 0.0
            if idx <= 0:
                self._recovering_s = 0.0
                return
            self._recovering_s = round(self._recovering_s + dt, 3)
            if self._recovering_s >= dyn["recover_after_s"]:
                self._record_consciousness_change(idx - 1, now)
                self._recovering_s = 0.0

    def _record_consciousness_change(self, new_idx, now):
        """Record one ACVPU step, with its time."""
        old = self.consciousness
        self.consciousness = ACVPU_LEVELS[new_idx]
        self.consciousness_history.append(
            {"from": old, "to": self.consciousness, "t": round(now, 3)}
        )

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
        if raised:
            # Vitals stopped being stable: the clock restarts when
            # the alarms clear.
            self._stable_since = None
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
        if self._active_alarms and not current:
            # The last active alarm just cleared: vitals are stable
            # again from now. (No-op when nothing was active — the
            # clock keeps running from admission or the last clear.)
            self._stable_since = now
        self._active_alarms = current

    def _track_escalation(self, news2, now, load_factor=1.0):
        """Open, update, or close the deterioration episode.

        An episode is open while the NEWS2 risk is medium or high.
        It records when it raised, the worst score it reached, and
        the worst risk — so a client can see a patient who was
        briefly high and came back, not just where they are now.
        It also measures the response: the team NEWS2 calls for at
        the risk the episode raised at (ward team for medium,
        critical care team for high) is due ``BASE_RESPONSE_S``
        times the staffing load factor at the moment it raised
        later, and the episode records when that response landed
        (``responded_t``, the scheduled time, not the tick it was
        noticed on — the same convention as an alarm ack) and how
        long it took (``time_to_respond_s``). An episode that
        closes — the risk returns to low, or the patient is
        discharged — before the response is due keeps both fields
        null: it cleared before anyone responded, and that is
        worth knowing too. When the response falls due on the
        very tick the risk returns to low, it still lands — the
        team was already on its way — so the due check runs before
        the close. Like the score itself, the response
        never touches the vital-sign alarm list.
        """
        now = round(now, 3)
        risk = news2["risk"]
        if self._escalation is not None:
            self._maybe_respond(now)
        if risk == "low":
            self._close_escalation(now)
            return
        if self._escalation is None:
            due = round(now + BASE_RESPONSE_S * load_factor, 3)
            self._escalation = {
                "raised_t": now,
                "cleared_t": None,
                "peak_score": news2["score"],
                "peak_risk": risk,
                "response": RESPONSE[risk],
                "response_due_t": due,
                "responded_t": None,
                "time_to_respond_s": None,
                # True once a NEWS2 staffing escalation has
                # covered this episode — the ward called in extra
                # nurses while this patient was at high risk.
                "staffing_escalated": False,
            }
            self.escalations_raised += 1
            # An opening escalation invokes the ward's hooks, if
            # any are configured — a webhook POST (dispatched in a
            # daemon thread, so it never blocks the tick) and/or an
            # in-process callback — and the episode records whether
            # the hook was invoked, so a snapshot, a recording, or
            # the run summary can say whether the escalation
            # reached the outside world. A hook failure is logged
            # and counted, never fatal to the tick. A failed
            # webhook POST is retried with backoff in the delivery
            # thread, and the episode records the retries and the
            # final delivery outcome (``hook_retries``,
            # ``hook_final_status``), updated when the delivery
            # settles — so an episode read right after it opens
            # may still say "pending".
            result = self.escalation_hooks.fire(
                {
                    "event": "escalation",
                    "bed": self.bed_id,
                    **self._escalation,
                },
                record=self._escalation,
            )
            self._escalation["hook_invoked"] = result["invoked"]
            self._escalation["hook_errors"] = result["errors"]
        else:
            if news2["score"] > self._escalation["peak_score"]:
                self._escalation["peak_score"] = news2["score"]
            if risk == "high":
                self._escalation["peak_risk"] = "high"

    def open_high_episode(self):
        """True while a deterioration episode is open at high risk.

        The episode is open and the latest reading scored high —
        the ward counts these to decide whether to call in extra
        staff. An unadmitted bed has no open episode.
        """
        if (
            self._escalation is None
            or not self.admitted
            or self.last_reading is None
        ):
            return False
        news2 = self.last_reading.get("news2")
        return news2 is not None and news2["risk"] == "high"

    def mark_staffing_escalated(self):
        """Flag the open deterioration episode as covered by a
        staffing escalation — the ward called in extra nurses
        while this patient was at high risk. No-op with no open
        episode."""
        if self._escalation is not None:
            self._escalation["staffing_escalated"] = True

    def _maybe_respond(self, now):
        """Record the response when it falls due.

        The response lands at its scheduled time
        (``response_due_t``), whether or not the risk has already
        returned to low on this tick — the team was dispatched
        when the escalation raised, so it arrives on schedule.
        """
        ep = self._escalation
        if ep is None or ep["responded_t"] is not None:
            return
        if now >= ep["response_due_t"]:
            ep["responded_t"] = ep["response_due_t"]
            ep["time_to_respond_s"] = round(
                ep["response_due_t"] - ep["raised_t"], 3
            )

    def _close_escalation(self, now):
        """Close the open deterioration episode, if any."""
        if self._escalation is not None:
            self._escalation["cleared_t"] = round(now, 3)
            self.escalation_history.append(self._escalation)
            self._escalation = None

    def _track_trend(self, now):
        """Open, update, or close the NEWS2 trend alert.

        A trend alert is open while the score is rising by the
        configured rules (see :func:`ward.news2.trend_for`). It
        records when it raised, the score the rise started from
        and when, the worst score it reached, and the biggest rise
        — so a client can see a patient who climbed fast and came
        back, not just where they are now — and whether the ward's
        trend-alert hooks were invoked for it (``hook_invoked``,
        False when no hook is configured). It closes when the
        score stops rising: the window slides forward and the
        rise out of it shrinks, or the recent ticks stop climbing.
        Like the score itself, it never touches the vital-sign
        alarm list.
        """
        now = round(now, 3)
        trend = trend_for(self.news2_history, **self.news2_trend)
        if trend["rising"]:
            if self._trend_alert is None:
                self._trend_alert = {
                    "raised_t": now,
                    "cleared_t": None,
                    "start_t": trend["start_t"],
                    "start_score": trend["start_score"],
                    "peak_score": trend["current_score"],
                    "peak_delta": trend["delta"],
                    "rule": trend["rule"],
                }
                self.trend_alerts_raised += 1
                # A fired trend alert invokes the ward's hooks, if
                # any are configured — a webhook POST (dispatched in
                # a daemon thread, so it never blocks the tick) and/or
                # an in-process callback — and the alert records
                # whether the hook was invoked, so a snapshot, a
                # recording, or the run summary can say whether the
                # alert reached the outside world. A hook failure is
                # logged and counted, never fatal to the tick. A
                # failed webhook POST is retried with backoff in
                # the delivery thread, and the alert records the
                # retries and the final delivery outcome
                # (``hook_retries``, ``hook_final_status``),
                # updated when the delivery settles — so an alert
                # read right after it raises may still say
                # "pending".
                result = self.trend_alert_hooks.fire(
                    {
                        "event": "trend_alert",
                        "bed": self.bed_id,
                        **self._trend_alert,
                    },
                    record=self._trend_alert,
                )
                self._trend_alert["hook_invoked"] = result["invoked"]
                self._trend_alert["hook_errors"] = result["errors"]
            else:
                if trend["current_score"] > self._trend_alert["peak_score"]:
                    self._trend_alert["peak_score"] = trend["current_score"]
                if trend["delta"] > self._trend_alert["peak_delta"]:
                    self._trend_alert["peak_delta"] = trend["delta"]
            return
        self._close_trend_alert(now)

    def _close_trend_alert(self, now):
        """Close the open trend alert, if any."""
        if self._trend_alert is not None:
            self._trend_alert["cleared_t"] = round(now, 3)
            self.trend_alert_history.append(self._trend_alert)
            self._trend_alert = None

    def trend_alerts(self):
        """Every trend alert this stay: closed ones, then the open
        one if there is one."""
        alerts = [dict(e) for e in self.trend_alert_history]
        if self._trend_alert is not None:
            alerts.append(dict(self._trend_alert))
        return alerts

    def news2_detail(self, now):
        """Per-patient early-warning state as plain JSON data.

        ``news2`` is the score from the latest reading (None while
        the bed has no patient or no reading yet); ``escalated`` is
        True while a deterioration episode is open; ``episodes``
        lists the closed ones with raised/cleared times and peak
        score and risk, and each episode carries its response
        timing — which team NEWS2 calls for, when the response
        was due and when it landed, and how long it took, or nulls
        when the episode closed before anyone responded — and
        whether the ward's escalation hooks were invoked for it
        (``hook_invoked``) and the retries and final delivery
        outcome (``hook_retries``, ``hook_final_status``) — and
        whether a NEWS2 staffing escalation covered it
        (``staffing_escalated``, True once the ward called in
        extra nurses while the episode was open at high risk).
        ``trend``
        is the current rate of change of the score (see
        :func:`ward.news2.trend_for`), and ``trend_alerts`` lists
        every trend alert this stay with raised/cleared times and
        peak score and rise, and whether the ward's hooks were
        invoked for it (``hook_invoked``) and the retries and
        final delivery outcome (``hook_retries``,
        ``hook_final_status``).
        """
        reading = self.last_reading
        offline = (
            self.offline_until is not None and now < self.offline_until
        )
        news2 = (
            reading.get("news2")
            if reading is not None and self.admitted and not offline
            else None
        )
        episodes = [dict(e) for e in self.escalation_history]
        if self._escalation is not None:
            episodes.append(dict(self._escalation))
        return {
            "bed": self.bed_id,
            "admitted": self.admitted,
            "on_oxygen": self.on_oxygen,
            "consciousness": self.consciousness,
            "consciousness_history": [
                dict(e) for e in self.consciousness_history
            ],
            "consciousness_changes": len(self.consciousness_history),
            "news2": news2,
            "escalated": self._escalation is not None,
            "escalations_raised": self.escalations_raised,
            "episodes": episodes,
            "trend": trend_for(self.news2_history, **self.news2_trend),
            "trend_alert_active": self._trend_alert is not None,
            "trend_alerts_raised": self.trend_alerts_raised,
            "trend_alerts": self.trend_alerts(),
        }

    def news2_history_detail(self, now):
        """Per-patient NEWS2 score history as plain JSON data.

        ``history`` is every score this stay, one entry per fresh
        reading in tick order (stale repeats are not recorded);
        ``current`` is the latest entry, or None before the first
        reading; ``peak_score`` / ``peak_risk`` are the worst score
        reached and the risk at that score. After a discharge the
        departed patient's history stays until the next admission
        clears it, with ``admitted`` false. ``trend`` is the
        current rate of change of the score and ``trend_alerts``
        lists every trend alert this stay.
        """
        history = [dict(e) for e in self.news2_history]
        peak = max(history, key=lambda e: e["score"], default=None)
        return {
            "bed": self.bed_id,
            "admitted": self.admitted,
            "profile": self.profile,
            "on_oxygen": self.on_oxygen,
            "consciousness": self.consciousness,
            "history": history,
            "points": len(history),
            "peak_score": peak["score"] if peak is not None else None,
            "peak_risk": peak["risk"] if peak is not None else None,
            "current": dict(history[-1]) if history else None,
            "trend": trend_for(self.news2_history, **self.news2_trend),
            "trend_alert_active": self._trend_alert is not None,
            "trend_alerts_raised": self.trend_alerts_raised,
            "trend_alerts": self.trend_alerts(),
        }

    def stable_s(self, now):
        """Seconds the patient's vitals have been alarm-free: since
        the last alarm cleared, or since admission when they never
        alarmed; 0 while an alarm is active."""
        if self._stable_since is None:
            return 0.0
        return round(now - self._stable_since, 3)

    def note_readiness(self, now, ready):
        """Record a discharge-readiness flip, if the flag changed.

        Called once per tick by the ward with the bed's current
        readiness (see :meth:`discharge_ready`). A flip — to
        ready or back to not-ready — records one event in
        ``readiness_history`` and invokes the ward's
        discharge-readiness hooks once, with a
        ``readiness_flip`` payload carrying the new state, when
        it flipped, how long the vitals have been stable,
        whether the medication orders are complete, and the
        NEWS2 score at the flip — so the outside world hears
        the moment a patient becomes ready (or stops being
        ready), and can see what the early-warning score said
        when it happened. No event when the flag is unchanged,
        so a bed that stays ready for an hour fires once, not
        once a tick. The event records whether the hook was
        invoked (``hook_invoked``, False when no hook is
        configured) and any synchronous error
        (``hook_errors``), with the retries and final delivery
        outcome (``hook_retries``, ``hook_final_status``)
        updated when a webhook delivery settles — the same
        convention as the trend-alert and escalation hooks. A
        hook failure is logged and counted, never fatal to the
        tick.
        """
        ready = bool(ready)
        if ready == self._ready:
            return
        now = round(now, 3)
        self._ready = ready
        news2 = (
            self.last_reading.get("news2")
            if self.last_reading is not None
            else None
        )
        event = {
            "bed": self.bed_id,
            "event": "readiness_flip",
            "ready": ready,
            "ready_t": now,
            "stable_s": self.stable_s(now) if self.admitted else 0.0,
            "orders_complete": self.meds.orders_complete(),
            "news2": dict(news2) if news2 is not None else None,
        }
        self.readiness_history.append(event)
        result = self.discharge_readiness_hooks.fire(
            dict(event),
            record=event,
        )
        event["hook_invoked"] = result["invoked"]
        event["hook_errors"] = result["errors"]

    def discharge_ready(self, now, stable_window_s):
        """True when the patient is discharge-ready: admitted, vitals
        stable for at least ``stable_window_s`` seconds, and all
        medication orders complete."""
        if not self.admitted:
            return False
        return (
            self.stable_s(now) >= stable_window_s
            and self.meds.orders_complete()
        )

    def readiness_detail(self, now, stable_window_s, auto_ordered=False):
        """Per-patient discharge-readiness state as plain JSON data.

        ``auto_ordered`` is True when the ward has raised an
        automatic discharge-planning order for this bed's current
        stay (one that is still planned or delayed). ``news2`` is
        the score from the latest reading (None while the bed has
        no patient or no reading yet), and ``readiness_events``
        lists every readiness flip — to ready or back to
        not-ready — with when it flipped and the NEWS2 score at
        the flip, and whether the ward's hooks were invoked for
        it (``hook_invoked``) and the retries and final delivery
        outcome (``hook_retries``, ``hook_final_status``)."""
        news2 = (
            self.last_reading.get("news2")
            if self.last_reading is not None and self.admitted
            else None
        )
        return {
            "bed": self.bed_id,
            "ready": self.discharge_ready(now, stable_window_s),
            "admitted": self.admitted,
            "stable_s": self.stable_s(now) if self.admitted else 0.0,
            "stable_window_s": float(stable_window_s),
            "orders_complete": self.meds.orders_complete(),
            "auto_ordered": bool(auto_ordered),
            "news2": dict(news2) if news2 is not None else None,
            "readiness_events": [dict(e) for e in self.readiness_history],
        }

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
