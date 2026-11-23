"""One bed's virtual monitor: emits vitals, obeys chaos state."""

import random
from collections import deque

from .meds import MedSchedule
from .news2 import (
    ACVPU_LEVELS,
    score_signs,
    validate_consciousness,
    validate_consciousness_dynamics,
)
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

    def __init__(self, bed_id, profile="stable", rng=None, acuity=3, meds=None, on_oxygen=False, consciousness="alert", consciousness_dynamics=None):
        self.bed_id = bed_id
        self.profile = profile
        self.acuity = validate_acuity(acuity)
        self.on_oxygen = self._validate_on_oxygen(on_oxygen)
        self.consciousness = validate_consciousness(consciousness)
        self.consciousness_dynamics = validate_consciousness_dynamics(
            consciousness_dynamics
        )
        # Every consciousness change this stay, as
        # {"from": level, "to": level, "t": seconds}; empty while
        # dynamics are off or the level has never moved.
        self.consciousness_history = []
        # Consecutive seconds the vitals-only risk has held in one
        # direction; each resets when the risk flips.
        self._deteriorating_s = 0.0
        self._recovering_s = 0.0
        self.meds = MedSchedule(bed_id, meds)
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
        # Early warning: a deterioration episode is open while the
        # patient's NEWS2 risk is medium or high — raised at the
        # first tick the risk leaves low, closed when it returns.
        # ``escalation_history`` keeps every closed episode; the
        # open one lives in ``_escalation`` until it closes.
        self._escalation = None
        self.escalation_history = []
        self.escalations_raised = 0

    @staticmethod
    def _validate_on_oxygen(on_oxygen):
        if not isinstance(on_oxygen, bool):
            raise ValueError(
                "on_oxygen must be a bool, got %r" % (on_oxygen,)
            )
        return on_oxygen

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
        self.pending_since = round(
            expected_t if expected_t is not None else now, 3
        )

    def begin_cleaning(self, now, cleaning_s):
        """Start turnover for a discharged (or queued) patient.

        Records the discharge delay when one was queued: how long
        past the expected discharge time cleaning actually started.
        """
        if self.pending_since is not None:
            delay = round(now - self.pending_since, 3)
            if delay > 0:
                self.discharge_delays += 1
                self.total_discharge_delay_s = round(
                    self.total_discharge_delay_s + delay, 3
                )
            self.pending_since = None
        self.awaiting_cleaning = False
        self._record_alarm_transitions([], now)
        self._close_escalation(now)
        self.meds.reset()
        self.admitted = False
        self.cleaning_until = round(now + cleaning_s, 3)
        self.turnovers += 1
        self.total_cleaning_s = round(self.total_cleaning_s + cleaning_s, 3)

    def admit(self, now, profile=None, acuity=None, on_oxygen=None, consciousness=None):
        """Admit a new patient to an available bed.

        Raises ValueError if the bed is occupied or still cleaning.
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
            "awaiting_cleaning": self.awaiting_cleaning,
            "discharge_expected_t": self.pending_since,
            "discharge_delays": self.discharge_delays,
            "total_discharge_delay_s": self.total_discharge_delay_s,
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

        # Consciousness moves before the reading is built, so this
        # tick's reading, score, and snapshot already carry the new
        # level — a change is visible in the same tick it happens.
        self._step_consciousness(now, dt)

        self.seq += 1
        reading = {
            "bed": self.bed_id,
            "seq": self.seq,
            "t": round(now, 3),
            "profile": self.profile,
            "acuity": self.acuity,
            "on_oxygen": self.on_oxygen,
            "consciousness": self.consciousness,
            "stale": False,
            **self.signs,
        }
        reading["alarms"] = alarms_for(self.signs)
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
            self.signs, self.on_oxygen, self.consciousness
        )
        self._track_escalation(reading["news2"], now)
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

    def _track_escalation(self, news2, now):
        """Open, update, or close the deterioration episode.

        An episode is open while the NEWS2 risk is medium or high.
        It records when it raised, the worst score it reached, and
        the worst risk — so a client can see a patient who was
        briefly high and came back, not just where they are now.
        """
        now = round(now, 3)
        risk = news2["risk"]
        if risk == "low":
            self._close_escalation(now)
            return
        if self._escalation is None:
            self._escalation = {
                "raised_t": now,
                "cleared_t": None,
                "peak_score": news2["score"],
                "peak_risk": risk,
            }
            self.escalations_raised += 1
        else:
            if news2["score"] > self._escalation["peak_score"]:
                self._escalation["peak_score"] = news2["score"]
            if risk == "high":
                self._escalation["peak_risk"] = "high"

    def _close_escalation(self, now):
        """Close the open deterioration episode, if any."""
        if self._escalation is not None:
            self._escalation["cleared_t"] = round(now, 3)
            self.escalation_history.append(self._escalation)
            self._escalation = None

    def news2_detail(self, now):
        """Per-patient early-warning state as plain JSON data.

        ``news2`` is the score from the latest reading (None while
        the bed has no patient or no reading yet); ``escalated`` is
        True while a deterioration episode is open; ``episodes``
        lists the closed ones with raised/cleared times and peak
        score and risk.
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
        }

    def stable_s(self, now):
        """Seconds the patient's vitals have been alarm-free: since
        the last alarm cleared, or since admission when they never
        alarmed; 0 while an alarm is active."""
        if self._stable_since is None:
            return 0.0
        return round(now - self._stable_since, 3)

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
        stay (one that is still planned or delayed)."""
        return {
            "bed": self.bed_id,
            "ready": self.discharge_ready(now, stable_window_s),
            "admitted": self.admitted,
            "stable_s": self.stable_s(now) if self.admitted else 0.0,
            "stable_window_s": float(stable_window_s),
            "orders_complete": self.meds.orders_complete(),
            "auto_ordered": bool(auto_ordered),
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
