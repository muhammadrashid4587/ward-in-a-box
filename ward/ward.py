"""The ward: N beds, each with a monitor, plus a chaos engine."""

import random
import threading
import time

from .alert_hooks import (
    validate_bed_blocker_hooks,
    validate_boarding_full_hooks,
    validate_discharge_priority_hooks,
    validate_discharge_readiness_hooks,
    validate_escalation_hooks,
    validate_missed_dose_hooks,
    validate_oxygen_outage_hooks,
    validate_power_outage_hooks,
    validate_sensor_drift_hooks,
    validate_trend_alert_hooks,
    validate_wait_time_hooks,
)
from .chaos import ChaosEngine
from .discharge import DischargePlanner
from .escalation import (
    StaffingEscalation,
    validate_news2_staffing_escalation,
)
from .monitor import Monitor
from .news2 import (
    validate_consciousness,
    validate_consciousness_dynamics,
    validate_news2_trend,
)
from .readiness import DischargeReadiness
from .staffing import Staffing
from .timeline import Timeline
from .triage import TriageQueue
from .vitals import PROFILES, validate_acuity

# Default patient mix when a scenario does not name profiles.
DEFAULT_PROFILES = ["stable", "stable", "stable", "tachycardic", "hypoxic", "febrile"]


class Ward:
    """A simulated ward.

    ``tick()`` advances every monitor one step and returns the snapshot:
    ``{"t": elapsed, "beds": [reading, ...], "staffing": {...}}``. Offline
    beds appear as ``{"bed": n, "online": false, "t": elapsed}``. ``start()`` runs the
    same loop in a background thread at ``rate_hz``. Pass a
    ``recorder`` (a ``ward.recorder.Recorder``) to persist every tick
    snapshot to a JSONL file.
    """

    def __init__(self, beds=12, rate_hz=1.0, profiles=None, chaos=None, seed=None, timeline=None, recorder=None, acuities=None, staffing=None, cleaning_s=30.0, meds=None, triage=None, discharge_planning=None, cleaning_capacity=None, discharge_readiness=None, oxygen=None, consciousness=None, consciousness_dynamics=None, news2_trend=None, trend_alert_hooks=None, escalation_hooks=None, boarding_full_hooks=None, wait_time_hooks=None, missed_dose_hooks=None, discharge_readiness_hooks=None, discharge_priority_hooks=None, oxygen_outage_hooks=None, power_outage_hooks=None, sensor_drift_hooks=None, bed_blocker_hooks=None, news2_staffing_escalation=None):
        if not 1 <= beds <= 64:
            raise ValueError("beds must be 1..64")
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        if cleaning_s <= 0:
            raise ValueError("cleaning_s must be > 0")
        if cleaning_capacity is not None and (
            not isinstance(cleaning_capacity, int)
            or isinstance(cleaning_capacity, bool)
            or cleaning_capacity < 1
        ):
            raise ValueError(
                "cleaning_capacity must be an int >= 1, got %r"
                % (cleaning_capacity,)
            )
        self.cleaning_capacity = cleaning_capacity
        if discharge_readiness is None:
            self.readiness = DischargeReadiness()
        elif isinstance(discharge_readiness, DischargeReadiness):
            self.readiness = discharge_readiness
        elif isinstance(discharge_readiness, dict):
            self.readiness = DischargeReadiness(**discharge_readiness)
        else:
            raise ValueError(
                "discharge_readiness must be a DischargeReadiness or a dict, got %r"
                % (discharge_readiness,)
            )
        self.news2_trend = validate_news2_trend(news2_trend)
        # NEWS2-driven staffing escalation: when enough
        # deterioration episodes are open at high risk, the ward
        # calls in extra nurses. Off by default — a ward with no
        # config never escalates and behaves exactly as before.
        _escalation_cfg = validate_news2_staffing_escalation(
            news2_staffing_escalation
        )
        if _escalation_cfg is None:
            self.staffing_escalation = StaffingEscalation(enabled=False)
        else:
            self.staffing_escalation = StaffingEscalation(**_escalation_cfg)
        # Trend-alert hooks: one shared object for the whole ward
        # (a webhook URL and/or an in-process callback), invoked
        # when a NEWS2 trend alert fires. Off by default.
        self.trend_alert_hooks = validate_trend_alert_hooks(trend_alert_hooks)
        # Escalation hooks: one shared object for the whole ward
        # (a webhook URL and/or an in-process callback), invoked
        # when a deterioration episode opens. Off by default.
        self.escalation_hooks = validate_escalation_hooks(escalation_hooks)
        # Boarding-full hooks: one shared object for the whole ward
        # (a webhook URL and/or an in-process callback), invoked
        # when a boarding-full alarm episode raises. Off by default.
        # The triage queue owns the object from here on: a queue
        # built with its own hooks keeps them unless this ward
        # configures its own, and ``self.boarding_full_hooks``
        # always points at the queue's object.
        self.boarding_full_hooks = validate_boarding_full_hooks(
            boarding_full_hooks
        )
        # Wait-time hooks: one shared object for the whole ward
        # (a webhook URL and/or an in-process callback), invoked
        # when a wait-time alarm raises. Off by default. The
        # triage queue owns the object from here on, exactly like
        # the boarding-full hooks above.
        self.wait_time_hooks = validate_wait_time_hooks(wait_time_hooks)
        # Missed-dose hooks: one shared object for the whole ward
        # (a webhook URL and/or an in-process callback), invoked
        # when a dose is missed or skipped for a stockout. Off by
        # default. Each monitor's schedule owns the object from
        # here on, exactly like the trend-alert hooks above.
        self.missed_dose_hooks = validate_missed_dose_hooks(
            missed_dose_hooks
        )
        # Discharge-readiness hooks: one shared object for the
        # whole ward (a webhook URL and/or an in-process
        # callback), invoked when a patient's readiness flag
        # flips — to ready or back to not-ready. Off by default.
        # Each monitor owns the object from here on, exactly
        # like the trend-alert hooks above.
        self.discharge_readiness_hooks = (
            validate_discharge_readiness_hooks(discharge_readiness_hooks)
        )
        # Discharge-priority hooks: one shared object for the
        # whole ward (a webhook URL and/or an in-process
        # callback), invoked when the discharge-priority order
        # changes — a bed becoming ready, a ready bed leaving,
        # or the ranking reordering. Off by default. The ward
        # owns the object: the order is a ward-level view, not
        # a per-bed one.
        self.discharge_priority_hooks = (
            validate_discharge_priority_hooks(discharge_priority_hooks)
        )
        # Oxygen-outage hooks: one shared object for the whole
        # ward (a webhook URL and/or an in-process callback),
        # invoked when an oxygen outage starts or ends, and when
        # a patient on oxygen first hits the outage's SpO2 floor.
        # Off by default. The ward owns the object: an outage is
        # a ward-level event, not a per-bed one.
        self.oxygen_outage_hooks = validate_oxygen_outage_hooks(
            oxygen_outage_hooks
        )
        # Every oxygen-outage hook event this run — the audit
        # trail for the oxygen-outage hooks — and the beds whose
        # floor hit has already been fired for the current
        # deficit episode, so a patient sitting on the floor
        # fires once, not once a tick. A bed leaves the set when
        # its deficit recovers to zero, so a later outage can
        # fire for it again.
        self.oxygen_outage_events = []
        self._oxygen_floor_fired = set()
        # Power-outage hooks: one shared object for the whole
        # ward (a webhook URL and/or an in-process callback),
        # invoked when a power outage starts or ends. Off by
        # default. The ward owns the object: an outage is a
        # ward-level event, not a per-bed one.
        self.power_outage_hooks = validate_power_outage_hooks(
            power_outage_hooks
        )
        # Every power-outage hook event this run — the audit
        # trail for the power-outage hooks.
        self.power_outage_events = []
        # Sensor-drift hooks: one shared object for the whole
        # ward (a webhook URL and/or an in-process callback),
        # invoked when a sensor-drift event starts or ends.
        # Off by default. The ward owns the object: a drift
        # event is a ward-level event, not a per-bed one.
        self.sensor_drift_hooks = validate_sensor_drift_hooks(
            sensor_drift_hooks
        )
        # Every sensor-drift hook event this run — the audit
        # trail for the sensor-drift hooks.
        self.sensor_drift_events = []
        # Bed-blocker hooks: one shared object for the whole
        # ward (a webhook URL and/or an in-process callback),
        # invoked when a bed-blocker event starts or ends.
        # Off by default. The ward owns the object: a block
        # is a ward-level event, not a per-bed one.
        self.bed_blocker_hooks = validate_bed_blocker_hooks(
            bed_blocker_hooks
        )
        # Every bed-blocker hook event this run — the audit
        # trail for the bed-blocker hooks.
        self.bed_blocker_events = []
        # The priority order last sent to the hooks (empty
        # before the first ready bed), and every order change
        # this run — the audit trail for the priority hooks.
        # ``_last_priority_rank`` maps each bed currently in
        # the order to its 1-based rank, and
        # ``_last_priority_flip`` maps it to the NEWS2 at the
        # flip that made it ready; together they let
        # :meth:`_check_priority_order` record, per bed, how
        # its rank changed. ``priority_rank_history`` is the
        # per-bed history: one entry per rank change, with the
        # new rank (null when the bed leaves the order), the
        # NEWS2 at the flip, and when it changed.
        self._last_priority_order = []
        self._last_priority_rank = {}
        self._last_priority_flip = {}
        self.priority_events = []
        self.priority_rank_history = []
        if discharge_planning is None:
            self.discharge_planner = DischargePlanner()
        elif isinstance(discharge_planning, DischargePlanner):
            self.discharge_planner = discharge_planning
        elif isinstance(discharge_planning, dict):
            self.discharge_planner = DischargePlanner(**discharge_planning)
        else:
            raise ValueError(
                "discharge_planning must be a DischargePlanner or a dict, got %r"
                % (discharge_planning,)
            )
        for order in self.discharge_planner.scheduled_orders:
            if not 0 <= order["bed"] < beds:
                raise ValueError(
                    "discharge order targets bed %r, ward has %d beds"
                    % (order["bed"], beds)
                )
        self.rng = random.Random(seed)
        if triage is None:
            self.triage = TriageQueue(
                rng=self.rng,
                boarding_full_hooks=self.boarding_full_hooks,
                wait_time_hooks=self.wait_time_hooks,
            )
        elif isinstance(triage, TriageQueue):
            self.triage = triage
            # A triage queue built elsewhere (e.g. by load_scenario)
            # gets this ward's seeded RNG for its walk-in arrivals,
            # so a seeded run reproduces the same arrivals.
            if self.triage.poisson is not None:
                self.triage.poisson.rng = self.rng
            if self.boarding_full_hooks.configured:
                self.triage.boarding_full_hooks = self.boarding_full_hooks
            if self.wait_time_hooks.configured:
                self.triage.wait_time_hooks = self.wait_time_hooks
        elif isinstance(triage, dict):
            kwargs = dict(triage)
            kwargs.setdefault("rng", self.rng)
            kwargs.setdefault(
                "boarding_full_hooks", self.boarding_full_hooks
            )
            kwargs.setdefault("wait_time_hooks", self.wait_time_hooks)
            self.triage = TriageQueue(**kwargs)
        else:
            raise ValueError("triage must be a TriageQueue or a dict, got %r" % (triage,))
        self.boarding_full_hooks = self.triage.boarding_full_hooks
        self.wait_time_hooks = self.triage.wait_time_hooks
        self.cleaning_s = float(cleaning_s)
        profiles = profiles or []
        for p in profiles:
            if p not in PROFILES:
                raise ValueError("unknown profile: %r" % p)
        acuities = acuities or []
        for a in acuities:
            validate_acuity(a)
        consciousness = consciousness or []
        for c in consciousness:
            validate_consciousness(c)
        consciousness_dynamics = consciousness_dynamics or []
        for d in consciousness_dynamics:
            validate_consciousness_dynamics(d)
        if staffing is None:
            self.staffing = Staffing()
        elif isinstance(staffing, Staffing):
            self.staffing = staffing
        elif isinstance(staffing, dict):
            self.staffing = Staffing(**staffing)
        else:
            raise ValueError("staffing must be a Staffing or a dict, got %r" % (staffing,))
        self.beds = beds
        self.rate_hz = rate_hz
        self.partitioned = False
        self.elapsed = 0.0
        self.timeline = timeline
        if timeline is not None and timeline.max_beds > beds:
            raise ValueError(
                "timeline targets %d beds, ward has %d beds" % (timeline.max_beds, beds)
            )
        self.monitors = []
        for i in range(beds):
            profile = profiles[i] if i < len(profiles) else DEFAULT_PROFILES[i % len(DEFAULT_PROFILES)]
            acuity = acuities[i] if i < len(acuities) else 3
            bed_meds = meds[i] if meds is not None and i < len(meds) else None
            on_oxygen = oxygen[i] if oxygen is not None and i < len(oxygen) else False
            bed_consciousness = (
                consciousness[i]
                if consciousness is not None and i < len(consciousness)
                else "alert"
            )
            bed_dynamics = (
                consciousness_dynamics[i]
                if consciousness_dynamics is not None
                and i < len(consciousness_dynamics)
                else None
            )
            monitor = Monitor(i, profile, rng=self.rng, acuity=acuity, meds=bed_meds, on_oxygen=on_oxygen, consciousness=bed_consciousness, consciousness_dynamics=bed_dynamics, news2_trend=self.news2_trend, trend_alert_hooks=self.trend_alert_hooks, escalation_hooks=self.escalation_hooks, missed_dose_hooks=self.missed_dose_hooks, discharge_readiness_hooks=self.discharge_readiness_hooks)
            # With a timeline, beds start unadmitted and come online as
            # the timeline admits them; without one, all beds start open.
            if timeline is not None:
                monitor.admitted = False
            self.monitors.append(monitor)
        self.chaos = ChaosEngine(chaos or [])
        # Discharges held by a bed-blocker event this run: a
        # discharge attempted while its bed was blocked counts
        # once, when it is first held — a second attempt during
        # the same block does not count again.
        self.bed_blocker_held = 0
        for ev in self.chaos.events:
            for b in ev.targets(self):
                if not 0 <= b < beds:
                    raise ValueError("chaos event targets bed %r, ward has %d beds" % (b, beds))
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest = None
        self.recorder = recorder

    def tick(self, dt=None):
        dt = dt if dt is not None else 1.0 / self.rate_hz
        self.elapsed = round(self.elapsed + dt, 3)
        self.chaos.apply(self, self.elapsed, dt)
        # Staff shortage: staff lost to an active shortage event
        # come off the effective roster before anything in this
        # tick reads the load factor, so alarms raised and checks
        # scheduled now already feel the sick call.
        self.staffing.shortage = self.chaos.shortage(self, self.elapsed)
        # Bed blocker: a discharge held by a block goes through
        # the tick its block ends, so the patient leaves as soon
        # as the bed is usable again instead of waiting for a new
        # order. Runs before the discharge planner, so a planned
        # order due on the same tick sees the bed unblocked.
        self._release_blocked_discharges()
        # Discharge planning: orders due now are placed, and orders
        # whose expected time has arrived are discharged — or queued
        # behind a full cleaning roster until a slot frees up.
        self.discharge_planner.tick(self.elapsed, self)
        # Cleaning slots freed by now (a bed's turnover ends when
        # ``now`` reaches its cleaning time, by the state check, not
        # by the monitor loop below) go to queued discharges, in
        # discharge-priority order — before the monitor loop runs,
        # so the snapshot is consistent within the tick.
        self._start_queued_cleanings()
        if self.timeline is not None:
            for i, m in enumerate(self.monitors):
                if not m.admitted:
                    admitted_at = self.timeline.admission_time(i)
                    # A bed still being cleaned is not re-admitted, even
                    # if its timeline admission time has passed; it comes
                    # back once turnover finishes.
                    if (
                        admitted_at is not None
                        and admitted_at <= self.elapsed
                        and m.state(self.elapsed) == "available"
                    ):
                        m.admitted = True
        # Triage: scheduled arrivals join the queue, walk-ins arrive
        # by the Poisson process at the ward clock hour's rate, then
        # waiting patients take any bed that has become available (a
        # discharge plus its cleaning leaves the bed available; the
        # timeline above already had first claim on it this tick).
        hour = (self.staffing.start_hour + self.elapsed / 3600.0) % 24
        self.triage.process_arrivals(self.elapsed, hour)
        self.triage.process_poisson_arrivals(self.elapsed, dt, hour)
        self.triage.admit_available(self.elapsed, self)
        # After admissions: a patient still waiting past their
        # acuity's threshold raises a wait-time alarm.
        self.triage.check_wait_alarms(self.elapsed)
        readings = []
        online = []
        occupied = sum(1 for m in self.monitors if m.admitted)
        load_factor = self.staffing.load_factor(occupied, self.elapsed)
        stockouts = self.chaos.stockouts(self, self.elapsed)
        for m in self.monitors:
            # Medication first, so a missed dose raises its alarm in
            # this tick's reading rather than the next one.
            m.meds_tick(self.elapsed, load_factor, stockouts.get(m.bed_id, ()))
            r = m.tick(self.elapsed, dt, load_factor=load_factor)
            if r is None:
                entry = {"bed": m.bed_id, "online": False, "t": self.elapsed}
                if m.power_dark(self.elapsed):
                    entry["power_dark"] = True
                if not m.admitted:
                    entry["admitted"] = False
                    entry["state"] = m.state(self.elapsed)
                readings.append(entry)
            else:
                r["online"] = True
                r["discharge_ready"] = m.discharge_ready(
                    self.elapsed, self.readiness.stable_window_s
                )
                readings.append(r)
                online.append(m.bed_id)
            # Discharge-readiness flips: one event (and one hook
            # delivery) per transition, in the tick it happens —
            # so a bed that goes quiet while its monitor is
            # offline still flips on schedule, and the flag in
            # the summary and the flip history never disagree.
            if m.admitted:
                m.note_readiness(
                    self.elapsed,
                    m.discharge_ready(
                        self.elapsed, self.readiness.stable_window_s
                    ),
                )
        # Oxygen-outage floor hits: a patient on oxygen whose
        # deficit reached the floor fires one hook event, in the
        # tick the floor is reached — after the monitor loop, so
        # this tick's deficit is already in.
        self._check_oxygen_floor_hits()
        # NEWS2-driven staffing escalation: with this tick's
        # scores in, count the deterioration episodes open at
        # high risk — at the threshold the ward calls in extra
        # nurses, so the load factor (and the ack, check, and
        # response clocks that run on it) improves from this tick
        # on, and every open high episode is flagged as covered.
        # Runs before the staffing tick, so the checks scheduled
        # now already use the improved factor.
        open_high = self.open_high_episodes()
        self.staffing_escalation.tick(self.elapsed, open_high)
        self.staffing.escalation_nurses = (
            self.staffing_escalation.extra_nurses_now
        )
        if self.staffing_escalation.active:
            for bed_id in open_high:
                self.monitors[bed_id].mark_staffing_escalated()
        self.staffing.tick(self.elapsed, occupied, online)
        for m in self.monitors:
            for alarm in m.raised_this_tick:
                self.staffing.alarm_raised(m.bed_id, alarm, self.elapsed, occupied)
        # Discharge paperwork timing: a bed flagged ready starts
        # its paperwork clock, and one that stops being ready (or
        # leaves) drops it until it flags ready again — so the
        # planner can measure ready -> order and ready -> discharge.
        for m in self.monitors:
            if (
                m.admitted
                and m.discharge_ready(
                    self.elapsed, self.readiness.stable_window_s
                )
            ):
                self.discharge_planner.note_ready(m.bed_id, self.elapsed)
            else:
                self.discharge_planner.note_not_ready(m.bed_id)
        # Discharge-readiness-driven workflow: when auto-ordering is
        # on, a bed flagged discharge-ready gets a discharge-planning
        # order at once — one per stay, never a second while a live
        # order already covers the bed — and a bed that stops being
        # ready has its live auto order cancelled, so the ward does
        # not send home a patient who just deteriorated. Manual and
        # scheduled orders are never cancelled here: they are a
        # doctor's order, not the ward's. A cancelled auto order
        # leaves the bed free to get a fresh one when it flags
        # ready again. Beds that flag ready on the same tick raise
        # their orders in discharge-priority order — safest first —
        # not bed order, so the safest patient is first in line
        # when a cleaning slot is scarce. Placed after the monitor
        # loop so this tick's readiness already reflects the latest
        # vitals and doses, and before the snapshot so the order is
        # in it.
        if self.readiness.auto_order:
            # The priority ranking for this tick's newly ready beds:
            # the same NEWS2-at-flip order the discharge-priority
            # view serves. A bed flagging ready this tick already
            # has its flip in the history (the monitor loop above
            # recorded it), so it is in the ranking; the fallback
            # only guards against a bed the ranking missed, which
            # goes last in bed order.
            priority = {
                e["bed"]: i
                for i, e in enumerate(self._priority_entries())
            }
            newly_ready = []
            for m in self.monitors:
                if not m.admitted:
                    continue
                if m.discharge_ready(
                    self.elapsed, self.readiness.stable_window_s
                ):
                    if not self.discharge_planner.has_live_order(m.bed_id):
                        newly_ready.append(m.bed_id)
                else:
                    self.discharge_planner.cancel_auto_orders(
                        m.bed_id, self.elapsed
                    )
            newly_ready.sort(key=lambda bed: (priority.get(bed, 1e9), bed))
            for bed in newly_ready:
                self.discharge_planner.place_order(
                    bed,
                    self.elapsed,
                    lead_time_s=self.readiness.auto_order_lead_time_s,
                    source="auto",
                )
        # Discharge-priority order: a change since the last
        # emitted order — a bed becoming ready, a ready bed
        # leaving, or the ranking reordering — fires the
        # priority hooks once, with the new order. Checked
        # after the monitor loop and the auto-order workflow,
        # so this tick's readiness and discharges are already
        # in, and before the snapshot, so the event is in it.
        # A tick where the order is unchanged fires nothing.
        self._check_priority_order()
        snap = {"t": self.elapsed, "beds": readings, "staffing": self.staffing_summary(), "turnover": self.turnover_summary(), "meds": self.meds_summary(), "triage": self.triage_summary(), "arrivals": self.arrivals_summary(), "discharge": self.discharge_summary(), "discharge_readiness": self.readiness_summary(), "discharge_priority": self.discharge_priority_summary(), "early_warning": self.early_warning_summary(), "sensor_drift": self.sensor_drift_summary(), "bed_blocker": self.bed_blocker_summary(), "oxygen_outage": self.oxygen_outage_summary(), "power_outage": self.power_outage_summary(), "network_outage": self.network_outage_summary()}
        with self._lock:
            self.latest = snap
        if self.recorder is not None:
            self.recorder.record(snap)
        return snap

    def snapshot(self):
        """Most recent snapshot, or None before the first tick."""
        with self._lock:
            return self.latest

    def sensor_drift_summary(self):
        """Per-bed sensor-drift state and ward totals.

        ``beds`` carries each bed's accumulated bias per vital
        (empty while its sensor is clean); ``beds_drifting``
        counts the beds with any bias, and ``max_bias`` is the
        largest absolute bias on any bed per vital, so a client
        can see at a glance how far the ward's sensors have
        wandered. The bias is a property of the sensor, not the
        patient: it stays after the drift event ends and is
        cleared only when a new patient is admitted to the bed.
        ``event_count`` counts every sensor-drift hook event this
        run — drift starts and drift ends — and ``events`` lists
        them, each with the tick, the drift window, and whether
        the ward's hooks were invoked for it (``hook_invoked``)
        and the retries and final delivery outcome
        (``hook_retries``, ``hook_final_status``);
        ``sensor_drift_hooks`` carries the hook totals: whether
        any hook is configured, how many hook deliveries were
        invoked, how many failed, how many retries were spent,
        and how many deliveries eventually landed.
        """
        beds = [m.sensor_drift_detail() for m in self.monitors]
        max_bias = {}
        for b in beds:
            for vital, bias in b["drift"].items():
                if abs(bias) > abs(max_bias.get(vital, 0.0)):
                    max_bias[vital] = bias
        return {
            "t": self.elapsed,
            "beds": beds,
            "beds_drifting": sum(1 for b in beds if b["drifting"]),
            "max_bias": max_bias,
            "event_count": len(self.sensor_drift_events),
            "events": [dict(e) for e in self.sensor_drift_events],
            "sensor_drift_hooks": (
                self.sensor_drift_hooks.summary()
            ),
        }

    def _sensor_drift_started(self, event, now):
        """Fire the start hook for one sensor-drift event.

        Called by the chaos engine on the first tick inside the
        event's window (see :meth:`chaos.ChaosEngine.apply`).
        """
        self._fire_sensor_drift_event(
            "sensor_drift_started", event, now
        )

    def _sensor_drift_ended(self, event, now):
        """Fire the end hook for one sensor-drift event.

        Called by the chaos engine on the first tick after the
        event's window closes.
        """
        self._fire_sensor_drift_event(
            "sensor_drift_ended", event, now
        )

    def _fire_sensor_drift_event(self, kind, event, now):
        """Fire one ward-level sensor-drift hook event.

        The payload carries the event kind, the tick, the
        drift window (``at``, ``ends``, ``duration``), the beds
        it targets, and the drift rates — so a hook can tell a
        ward-wide miscalibration from a single-bed one and know
        how fast the readings are creeping. The event is
        recorded on the audit trail with whether the hook was
        invoked for it, any synchronous error, and the retries
        and final delivery outcome.
        """
        payload = {
            "event": kind,
            "t": now,
            "at": event.at,
            "ends": event.end,
            "duration": event.duration,
            "beds": event.targets(self),
            "drift": dict(event.drift),
        }
        self.sensor_drift_events.append(payload)
        result = self.sensor_drift_hooks.fire(
            dict(payload),
            record=payload,
        )
        payload["hook_invoked"] = result["invoked"]
        payload["hook_errors"] = result["errors"]

    def _bed_blocker_started(self, event, now):
        """Fire the start hook for one bed-blocker event.

        Called by the chaos engine on the first tick inside the
        event's window (see :meth:`chaos.ChaosEngine.apply`).
        """
        self._fire_bed_blocker_event(
            "bed_blocker_started", event, now
        )

    def _bed_blocker_ended(self, event, now):
        """Fire the end hook for one bed-blocker event.

        Called by the chaos engine on the first tick after the
        event's window closes.
        """
        self._fire_bed_blocker_event(
            "bed_blocker_ended", event, now
        )

    def _fire_bed_blocker_event(self, kind, event, now):
        """Fire one ward-level bed-blocker hook event.

        The payload carries the event kind, the tick, the
        block window (``at``, ``ends``, ``duration``), the beds
        it targets, and the block reason — so a hook can tell
        a bed waiting for a porter from an isolation clean
        and know how long the bed is out of use. The event is
        recorded on the audit trail with whether the hook was
        invoked for it, any synchronous error, and the retries
        and final delivery outcome.
        """
        payload = {
            "event": kind,
            "t": now,
            "at": event.at,
            "ends": event.end,
            "duration": event.duration,
            "beds": event.targets(self),
            "reason": event.reason,
        }
        self.bed_blocker_events.append(payload)
        result = self.bed_blocker_hooks.fire(
            dict(payload),
            record=payload,
        )
        payload["hook_invoked"] = result["invoked"]
        payload["hook_errors"] = result["errors"]

    def _oxygen_outage_started(self, event, now):
        """Fire the start hook for one oxygen-outage event.

        Called by the chaos engine on the first tick inside the
        event's window (see :meth:`chaos.ChaosEngine.apply`).
        """
        self._fire_oxygen_outage_event(
            "oxygen_outage_started", event, now
        )

    def _oxygen_outage_ended(self, event, now):
        """Fire the end hook for one oxygen-outage event.

        Called by the chaos engine on the first tick after the
        event's window closes.
        """
        self._fire_oxygen_outage_event(
            "oxygen_outage_ended", event, now
        )

    def _power_outage_started(self, event, now):
        """Fire the start hook for one power-outage event.

        Called by the chaos engine on the first tick inside the
        event's window (see :meth:`chaos.ChaosEngine.apply`).
        """
        self._fire_power_outage_event(
            "power_outage_started", event, now
        )

    def _power_outage_ended(self, event, now):
        """Fire the end hook for one power-outage event.

        Called by the chaos engine on the first tick after the
        event's window closes.
        """
        self._fire_power_outage_event(
            "power_outage_ended", event, now
        )

    def _fire_power_outage_event(self, kind, event, now):
        """Fire one ward-level power-outage hook event.

        The payload carries the event kind, the tick, the
        outage window (``at``, ``ends``, ``duration``), the beds
        it targets, and the outage's battery and reboot times —
        so a hook can tell a ward-wide mains failure from a
        single-bed one and know how long the monitors run on
        battery before going dark. The event is recorded on the
        audit trail with whether the hook was invoked for it,
        any synchronous error, and the retries and final
        delivery outcome.
        """
        payload = {
            "event": kind,
            "t": now,
            "at": event.at,
            "ends": event.end,
            "duration": event.duration,
            "beds": event.targets(self),
            "power": dict(event.power),
        }
        self.power_outage_events.append(payload)
        result = self.power_outage_hooks.fire(
            dict(payload),
            record=payload,
        )
        payload["hook_invoked"] = result["invoked"]
        payload["hook_errors"] = result["errors"]

    def _fire_oxygen_outage_event(self, kind, event, now):
        """Fire one ward-level oxygen-outage hook event.

        The payload carries the event kind, the tick, the
        outage window (``at``, ``ends``, ``duration``), the beds
        it targets, and the outage's rate, floor, and recovery
        rate — so a hook can tell a ward-wide supply failure
        from a single-bed one and know how far patients will
        fall. The event is recorded on the audit trail with
        whether the hook was invoked for it, any synchronous
        error, and the retries and final delivery outcome.
        """
        payload = {
            "event": kind,
            "t": now,
            "at": event.at,
            "ends": event.end,
            "duration": event.duration,
            "beds": event.targets(self),
            "oxygen": dict(event.oxygen),
        }
        self.oxygen_outage_events.append(payload)
        result = self.oxygen_outage_hooks.fire(
            dict(payload),
            record=payload,
        )
        payload["hook_invoked"] = result["invoked"]
        payload["hook_errors"] = result["errors"]

    def _check_oxygen_floor_hits(self):
        """Fire one floor-hit hook per patient per outage.

        A monitor flags the tick its deficit first reaches the
        outage's floor; the ward fires one
        ``oxygen_spo2_floor_hit`` event for that bed — with the
        bed, the tick, the SpO2 the patient is sitting on, the
        floor, and the deficit that got them there — and
        remembers the bed until its deficit recovers to zero, so
        a patient sitting on the floor fires once, not once a
        tick, and a later outage can fire for them again. A
        callback can use the event to escalate: the patient has
        hit the lowest saturation the outage can push them to,
        and the low_spo2 alarm and NEWS2 climb from here are
        the outage doing its worst.
        """
        for m in self.monitors:
            if m.oxygen_deficit <= 0:
                self._oxygen_floor_fired.discard(m.bed_id)
                continue
            if not m.oxygen_floor_hit:
                continue
            if m.bed_id in self._oxygen_floor_fired:
                continue
            self._oxygen_floor_fired.add(m.bed_id)
            payload = {
                "event": "oxygen_spo2_floor_hit",
                "t": self.elapsed,
                "bed": m.bed_id,
                "spo2": m.reported_signs()["spo2"],
                "floor": (
                    m.oxygen_outage_cfg["floor"]
                    if m.oxygen_outage_cfg is not None
                    else None
                ),
                "deficit": m.oxygen_deficit,
            }
            self.oxygen_outage_events.append(payload)
            result = self.oxygen_outage_hooks.fire(
                dict(payload),
                record=payload,
            )
            payload["hook_invoked"] = result["invoked"]
            payload["hook_errors"] = result["errors"]

    def oxygen_outage_summary(self):
        """Per-bed oxygen-outage state and ward totals.

        ``beds`` carries each bed's outage state — whether the
        supply is down, when it returns, and how many SpO2
        points its patient has lost to it — ``beds_on_oxygen``
        counts the patients the supply feeds, ``beds_desaturating``
        counts the ones currently below their walked saturation
        (including a patient still recovering after the supply
        returned), and ``max_deficit`` is the largest SpO2 loss
        on any bed, so a client can see at a glance how far the
        ward's oxygen-dependent patients have fallen.
        ``event_count`` counts every oxygen-outage hook event this
        run — outage starts, outage ends, and per-patient SpO2
        floor hits — and ``events`` lists them, each with the
        tick, the outage window or the bed and floor, and whether
        the ward's hooks were invoked for it (``hook_invoked``)
        and the retries and final delivery outcome
        (``hook_retries``, ``hook_final_status``);
        ``oxygen_outage_hooks`` carries the hook totals: whether
        any hook is configured, how many hook deliveries were
        invoked, how many failed, how many retries were spent,
        and how many deliveries eventually landed.
        """
        beds = [m.oxygen_outage_detail(self.elapsed) for m in self.monitors]
        return {
            "t": self.elapsed,
            "beds": beds,
            "beds_on_oxygen": sum(1 for b in beds if b["on_oxygen"]),
            "beds_desaturating": sum(
                1 for b in beds if b["desaturating"]
            ),
            "max_deficit": max((b["deficit"] for b in beds), default=0.0),
            "event_count": len(self.oxygen_outage_events),
            "events": [dict(e) for e in self.oxygen_outage_events],
            "oxygen_outage_hooks": (
                self.oxygen_outage_hooks.summary()
            ),
        }

    def power_outage_summary(self):
        """Per-bed power-outage state and ward totals.

        ``beds`` carries each bed's outage state — whether the
        mains is down, when it returns, whether the monitor is
        on battery, dark, or rebooting, and how many ticks this
        stay it has been dark — ``beds_on_battery`` counts the
        monitors still running on backup, ``beds_dark`` counts
        the ones that have gone dark (battery dead or still
        rebooting), ``beds_rebooting`` counts the ones waiting
        out a reboot after power returned, and ``dark_ticks``
        counts every dark tick across the ward, so a client can
        see at a glance how much of the ward the outage has
        blinded and for how long.
        ``event_count`` counts every power-outage hook event this
        run — outage starts and outage ends — and ``events``
        lists them, each with the tick, the outage window, and
        whether the ward's hooks were invoked for it
        (``hook_invoked``) and the retries and final delivery
        outcome (``hook_retries``, ``hook_final_status``);
        ``power_outage_hooks`` carries the hook totals: whether
        any hook is configured, how many hook deliveries were
        invoked, how many failed, how many retries were spent,
        and how many deliveries eventually landed.
        """
        beds = [m.power_outage_detail(self.elapsed) for m in self.monitors]
        return {
            "t": self.elapsed,
            "beds": beds,
            "beds_on_battery": sum(
                1 for b in beds if b["on_battery"]
            ),
            "beds_dark": sum(1 for b in beds if b["dark"]),
            "beds_rebooting": sum(
                1 for b in beds if b["rebooting"]
            ),
            "dark_ticks": sum(b["dark_ticks"] for b in beds),
            "event_count": len(self.power_outage_events),
            "events": [dict(e) for e in self.power_outage_events],
            "power_outage_hooks": (
                self.power_outage_hooks.summary()
            ),
        }

    def network_outage_summary(self):
        """Per-bed network-outage state and ward totals.

        ``beds`` carries each bed's outage state — whether the
        link to the central station is down, when it returns,
        how many readings are buffered locally, and this stay's
        buffered / synced / dropped counts and undelivered
        alarms — ``beds_down`` counts the beds whose link is
        down, ``backlog`` is the buffered readings across the
        ward, and ``buffered`` / ``synced`` / ``dropped`` /
        ``alarms_undelivered`` / ``alarms_synced`` are the ward
        totals, so a client can see at a glance how much of the
        ward the outage has cut off, how much data is waiting to
        sync, and how much the outage has already cost.
        """
        beds = [m.network_outage_detail(self.elapsed) for m in self.monitors]
        return {
            "t": self.elapsed,
            "beds": beds,
            "beds_down": sum(1 for b in beds if b["outage_active"]),
            "backlog": sum(b["backlog"] for b in beds),
            "buffered": sum(b["buffered"] for b in beds),
            "synced": sum(b["synced"] for b in beds),
            "dropped": sum(b["dropped"] for b in beds),
            "alarms_undelivered": sum(
                b["alarms_undelivered"] for b in beds
            ),
            "alarms_synced": sum(b["alarms_synced"] for b in beds),
        }

    def bed_blocker_summary(self):
        """Per-bed bed-blocker state and ward totals.

        ``beds`` carries each bed's block state — whether it is
        blocked, why, and when the block ends — and whether a
        discharge attempted during the block is still waiting
        for it to clear; ``beds_blocked`` counts the beds
        currently blocked, ``discharges_pending`` counts the
        discharges still held by a block, and ``discharges_held``
        counts every discharge the run has held this way, so a
        client can see both the beds out of use now and how
        often a block has held a patient back.
        ``event_count`` counts every bed-blocker hook event this
        run — block starts and block ends — and ``events`` lists
        them, each with the tick, the block window, and whether
        the ward's hooks were invoked for it (``hook_invoked``)
        and the retries and final delivery outcome
        (``hook_retries``, ``hook_final_status``);
        ``bed_blocker_hooks`` carries the hook totals: whether
        any hook is configured, how many hook deliveries were
        invoked, how many failed, how many retries were spent,
        and how many deliveries eventually landed.
        """
        beds = [m.bed_blocker_detail(self.elapsed) for m in self.monitors]
        return {
            "t": self.elapsed,
            "beds": beds,
            "beds_blocked": sum(1 for b in beds if b["blocked"]),
            "discharges_pending": sum(
                1 for b in beds if b["discharge_pending"]
            ),
            "discharges_held": self.bed_blocker_held,
            "event_count": len(self.bed_blocker_events),
            "events": [dict(e) for e in self.bed_blocker_events],
            "bed_blocker_hooks": (
                self.bed_blocker_hooks.summary()
            ),
        }

    def alarm_summary(self, window_s=60.0):
        """Per-bed alarm history and rate metrics for the whole ward."""
        beds = [m.alarm_summary(self.elapsed, window_s) for m in self.monitors]
        return {
            "t": self.elapsed,
            "beds": beds,
            "total_raised": sum(b["total_raised"] for b in beds),
            "rate_per_min": round(sum(b["rate_per_min"] for b in beds), 3),
        }

    def cleaning_active_count(self, now):
        """Beds being cleaned at elapsed time ``now``."""
        return sum(1 for m in self.monitors if m.state(now) == "cleaning")

    def _release_blocked_discharges(self):
        """Discharge the patients whose discharge a bed blocker
        was holding, now that their block has cleared.

        A discharge attempted while its bed was blocked is held
        until the block ends (see :meth:`discharge`); this runs
        every tick after the chaos engine, so the patient leaves
        on the very tick the bed becomes usable again, through
        the normal discharge path — including a full cleaning
        roster, which queues the discharge like any other. A
        delayed discharge-planning order is completed when the
        discharge actually starts cleaning, not when it queues
        behind the roster — the queued case is completed by
        :meth:`_start_queued_cleanings` when its slot frees up.
        """
        for m in self.monitors:
            if not m.blocked_discharge_pending:
                continue
            if not m.admitted or m.is_blocked(self.elapsed):
                continue
            self.discharge(m.bed_id)
            if not m.awaiting_cleaning:
                self.discharge_planner.complete(m.bed_id, self.elapsed)

    def discharge(self, bed_id):
        """Discharge the patient in ``bed_id``.

        The bed goes into cleaning for ``cleaning_s`` seconds and
        cannot take a new admission until turnover finishes. When
        ``cleaning_capacity`` is set and every cleaning slot is busy,
        the discharge is queued instead: the patient stays in the bed,
        still monitored, until a slot frees up (see
        :meth:`_start_queued_cleanings`). When the bed is blocked by
        a bed-blocker event, the discharge is held instead: the
        patient stays in the bed, still monitored, until the block
        clears, when :meth:`_release_blocked_discharges` puts it
        through. A bed whose discharge is already queued or held is
        left alone.
        """
        m = self._monitor(bed_id)
        if m.awaiting_cleaning:
            return
        if not m.admitted:
            raise ValueError("bed %d is not occupied" % bed_id)
        if m.is_blocked(self.elapsed):
            if m.mark_blocked_discharge(self.elapsed):
                self.bed_blocker_held += 1
            return
        if (
            self.cleaning_capacity is not None
            and self.cleaning_active_count(self.elapsed)
            >= self.cleaning_capacity
        ):
            m.queue_discharge(self.elapsed)
            return
        m.discharge(self.elapsed, self.cleaning_s)

    def discharge_planned(self, bed_id, expected_t):
        """Discharge a planned discharge due at ``expected_t``.

        Returns ``"discharged"`` when the bed went into cleaning at
        once, ``"queued"`` when cleaning is backed up — or the bed
        is blocked by a bed-blocker event — and the patient stays
        in the bed, or ``"cancelled"`` when the patient already left
        (a chaos ``discharge`` event fired first, say). A discharge
        held by a block goes through the tick the block ends (see
        :meth:`_release_blocked_discharges`), which completes the
        delayed order with the wait recorded as its delay.
        """
        m = self._monitor(bed_id)
        if m.awaiting_cleaning:
            return "queued"
        if not m.admitted:
            return "cancelled"
        if m.is_blocked(self.elapsed):
            if m.mark_blocked_discharge(self.elapsed, expected_t):
                self.bed_blocker_held += 1
            return "queued"
        if (
            self.cleaning_capacity is not None
            and self.cleaning_active_count(self.elapsed)
            >= self.cleaning_capacity
        ):
            m.queue_discharge(self.elapsed, expected_t)
            return "queued"
        m.discharge(self.elapsed, self.cleaning_s)
        return "discharged"

    def _last_ready_flip(self, monitor):
        """The monitor's latest flip to discharge-ready, or None.

        The flip carries the NEWS2 score at the moment the bed was
        flagged ready — the stable, recorded value the
        discharge-priority ranking uses, not the score drifting
        tick by tick.
        """
        flips = [e for e in monitor.readiness_history if e["ready"]]
        return flips[-1] if flips else None

    def _queued_cleaning_monitors(self):
        """Beds queued for cleaning, in the order cleaning starts.

        Discharge-priority order — the same ranking the
        discharge-priority view serves: lowest NEWS2 at the flip
        that made the bed ready first, ties by earliest flip then
        bed — so when several beds queue behind a full roster the
        safest patient gets the next slot, not the one whose
        discharge was due first. A bed that never flagged ready
        has no NEWS2 at a flip and sorts after every ready bed,
        in expected-discharge then bed order, exactly as before.
        """
        queued = [m for m in self.monitors if m.awaiting_cleaning]

        def key(m):
            flip = self._last_ready_flip(m)
            news2 = flip["news2"] if flip is not None else None
            return (
                news2["score"] if news2 is not None else 1e9,
                flip["ready_t"] if flip is not None else 1e9,
                m.pending_since if m.pending_since is not None else 0,
                m.bed_id,
            )

        queued.sort(key=key)
        return queued

    def cleaning_queue(self):
        """Beds awaiting a cleaning slot, in the order cleaning
        will start them — see :meth:`_queued_cleaning_monitors`.

        Each entry carries the bed, the NEWS2 at the flip that
        made it ready (null when it never flagged ready), when it
        flagged ready, and when its discharge was expected.
        """
        queue = []
        for m in self._queued_cleaning_monitors():
            flip = self._last_ready_flip(m)
            news2 = flip["news2"] if flip is not None else None
            queue.append(
                {
                    "bed": m.bed_id,
                    "news2_at_flip": (
                        news2["score"] if news2 is not None else None
                    ),
                    "ready_t": flip["ready_t"] if flip is not None else None,
                    "expected_t": m.pending_since,
                }
            )
        return queue

    def _start_queued_cleanings(self):
        """Start cleaning for discharges queued behind a full roster.

        In discharge-priority order (see
        :meth:`_queued_cleaning_monitors`); each one frees its
        patient from the bed and tells the discharge planner the
        order is complete, with its delay.
        """
        for m in self._queued_cleaning_monitors():
            if (
                self.cleaning_capacity is not None
                and self.cleaning_active_count(self.elapsed)
                >= self.cleaning_capacity
            ):
                break
            m.begin_cleaning(self.elapsed, self.cleaning_s)
            self.discharge_planner.complete(m.bed_id, self.elapsed)

    def readiness_summary(self):
        """Per-patient discharge-readiness flags and ward totals.

        ``readiness_flips`` counts every readiness flip this
        run — to ready and back to not-ready — across all beds,
        split into ``flips_to_ready`` and
        ``flips_to_not_ready``; ``discharge_readiness_hooks``
        carries the hook totals: whether any hook is configured,
        how many hook deliveries were invoked, how many failed,
        how many retries were spent, and how many deliveries
        eventually landed.
        """
        beds = [
            m.readiness_detail(
                self.elapsed,
                self.readiness.stable_window_s,
                auto_ordered=self.discharge_planner.has_auto_order(m.bed_id),
            )
            for m in self.monitors
        ]
        flips = [e for b in beds for e in b["readiness_events"]]
        return {
            "t": self.elapsed,
            "stable_window_s": self.readiness.stable_window_s,
            "auto_order": self.readiness.auto_order,
            "auto_order_lead_time_s": self.readiness.auto_order_lead_time_s,
            "beds": beds,
            "occupied": sum(1 for b in beds if b["admitted"]),
            "ready": sum(1 for b in beds if b["ready"]),
            "not_ready": sum(
                1 for b in beds if b["admitted"] and not b["ready"]
            ),
            "readiness_flips": len(flips),
            "flips_to_ready": sum(1 for e in flips if e["ready"]),
            "flips_to_not_ready": sum(1 for e in flips if not e["ready"]),
            "discharge_readiness_hooks": (
                self.discharge_readiness_hooks.summary()
            ),
        }

    def _priority_entries(self):
        """The discharge-priority entries: ready beds ranked by
        NEWS2 at the flip that made them ready, lowest first.

        See :meth:`discharge_priority_summary` for the ranking
        rules. Returned as plain dicts, in rank order.
        """
        entries = []
        for m in self.monitors:
            if not m.discharge_ready(
                self.elapsed, self.readiness.stable_window_s
            ):
                continue
            flips = [e for e in m.readiness_history if e["ready"]]
            flip = flips[-1] if flips else None
            flip_news2 = flip["news2"] if flip is not None else None
            current = (
                m.last_reading.get("news2")
                if m.last_reading is not None and m.admitted
                else None
            )
            entries.append(
                {
                    "bed": m.bed_id,
                    "news2_at_flip": (
                        flip_news2["score"]
                        if flip_news2 is not None
                        else None
                    ),
                    "risk_at_flip": (
                        flip_news2["risk"]
                        if flip_news2 is not None
                        else None
                    ),
                    "news2": (
                        current["score"] if current is not None else None
                    ),
                    "ready_t": flip["ready_t"] if flip is not None else None,
                    "time_ready_s": (
                        round(self.elapsed - flip["ready_t"], 3)
                        if flip is not None
                        else None
                    ),
                }
            )
        entries.sort(
            key=lambda e: (
                (
                    e["news2_at_flip"]
                    if e["news2_at_flip"] is not None
                    else 1e9
                ),
                e["ready_t"] if e["ready_t"] is not None else 1e9,
                e["bed"],
            )
        )
        return entries

    def _check_priority_order(self):
        """Fire the discharge-priority hooks when the ranked
        order changed since the last emitted one.

        A bed becoming ready, a ready bed leaving, or the
        NEWS2-at-flip ranking reordering the list all change
        the order; a tick where the order is unchanged fires
        nothing, so a ward that stays ready for an hour sends
        one notification, not one a tick. The event records the
        new order — the same entries ``GET /discharge-priority``
        serves — and the order it replaces, whether the hook
        was invoked (``hook_invoked``, False with no hooks) and
        any synchronous error (``hook_errors``), with the
        retries and final delivery outcome (``hook_retries``,
        ``hook_final_status``) updated when a webhook delivery
        settles — the same convention as the other hooks. A
        hook failure is logged and counted, never fatal to the
        tick. Called from :meth:`tick` only: a summary read
        (the endpoint, a snapshot) never fires a hook.
        """
        entries = self._priority_entries()
        order = [e["bed"] for e in entries]
        if order == self._last_priority_order:
            return
        now = round(self.elapsed, 3)
        # Per-bed rank history: one entry per bed whose rank
        # changed in this tick — newly ready (rank 1..n), moved
        # up or down as the order re-sorted, or dropped out of
        # the order (rank null, keeping the NEWS2 at the flip
        # that had made it ready). A bed whose rank did not
        # change records nothing, so the history stays a
        # change log, not a per-tick dump.
        new_rank = {e["bed"]: i + 1 for i, e in enumerate(entries)}
        new_flip = {e["bed"]: e["news2_at_flip"] for e in entries}
        for bed in sorted(
            set(self._last_priority_rank) | set(new_rank)
        ):
            old = self._last_priority_rank.get(bed)
            new = new_rank.get(bed)
            if old == new:
                continue
            self.priority_rank_history.append(
                {
                    "bed": bed,
                    "t": now,
                    "rank": new,
                    "news2_at_flip": (
                        new_flip.get(bed)
                        if new is not None
                        else self._last_priority_flip.get(bed)
                    ),
                }
            )
        self._last_priority_rank = new_rank
        self._last_priority_flip = new_flip
        event = {
            "event": "discharge_priority_changed",
            "t": now,
            "beds": [dict(e) for e in entries],
            "previous_beds": list(self._last_priority_order),
        }
        self._last_priority_order = order
        self.priority_events.append(event)
        result = self.discharge_priority_hooks.fire(
            dict(event),
            record=event,
        )
        event["hook_invoked"] = result["invoked"]
        event["hook_errors"] = result["errors"]

    def discharge_priority_summary(self):
        """Discharge-ready beds ranked by NEWS2, safest first.

        Only beds currently flagged discharge-ready appear, in
        the order the ward should work through them: lowest NEWS2
        first, so the safest patient to send home is at the top.
        The ranking uses the NEWS2 score recorded at the flip
        that made the bed ready (``news2_at_flip``) — the stable,
        recorded value, not the score drifting tick by tick — so
        a bed that flagged ready at NEWS2 6 still ranks, last,
        exactly as the readiness flag says it should: readiness
        itself has no NEWS2 gate, and this view does not add one.
        Ties break by the earliest flip, then bed number. Each
        entry also carries the current NEWS2 score (``news2``),
        when the bed flagged ready (``ready_t``), and how long
        it has been ready (``time_ready_s``). A bed with no score
        at its flip — no reading yet — sorts last.

        ``priority_changes`` counts every order change this run
        and ``changes`` lists them, each with the new order, the
        order it replaced, and whether the ward's hooks were
        invoked for it (``hook_invoked``) and the retries and
        final delivery outcome (``hook_retries``,
        ``hook_final_status``); ``rank_changes`` counts every
        per-bed rank change this run and ``rank_history`` is
        that history grouped per bed — each entry the tick it
        changed, the new 1-based rank (null when the bed left
        the order), and the NEWS2 at the flip that had made it
        ready — so a bed manager can see how a bed moved
        through the queue, not just where it sits now;
        ``discharge_priority_hooks``
        carries the hook totals: whether any hook is configured,
        how many hook deliveries were invoked, how many failed,
        how many retries were spent, and how many deliveries
        eventually landed.
        """
        entries = self._priority_entries()
        rank_history = {}
        for change in self.priority_rank_history:
            rank_history.setdefault(str(change["bed"]), []).append(
                {
                    "t": change["t"],
                    "rank": change["rank"],
                    "news2_at_flip": change["news2_at_flip"],
                }
            )
        return {
            "t": self.elapsed,
            "beds": entries,
            "ready": len(entries),
            "priority_changes": len(self.priority_events),
            "changes": [dict(e) for e in self.priority_events],
            "rank_changes": len(self.priority_rank_history),
            "rank_history": rank_history,
            "discharge_priority_hooks": (
                self.discharge_priority_hooks.summary()
            ),
        }

    def early_warning_summary(self):
        """Per-patient NEWS2 scores, deterioration episodes, and
        response timing.

        ``responses`` aggregates every episode this run: how many
        got a response, how many closed before anyone responded,
        how many are still waiting for one, and the mean and
        longest time from an escalation raising to the response
        landing (null before the first response).
        ``trend_alert_hooks`` carries the trend-alert hook totals:
        whether any hook is configured, how many hook deliveries
        were invoked, how many failed, how many retries were spent,
        and how many deliveries eventually landed.
        ``escalation_hooks`` carries the same totals for the
        escalation hooks.
        """
        beds = [m.news2_detail(self.elapsed) for m in self.monitors]
        scored = [b for b in beds if b["news2"] is not None]
        episodes = [e for b in beds for e in b["episodes"]]
        responded = [e for e in episodes if e["responded_t"] is not None]
        times = [e["time_to_respond_s"] for e in responded]
        return {
            "t": self.elapsed,
            "beds": beds,
            "on_oxygen": sum(1 for b in beds if b["on_oxygen"]),
            "not_alert": sum(
                1 for b in beds if b["consciousness"] != "alert"
            ),
            "consciousness_changes": sum(
                b["consciousness_changes"] for b in beds
            ),
            "high_risk": sum(
                1 for b in scored if b["news2"]["risk"] == "high"
            ),
            "medium_risk": sum(
                1 for b in scored if b["news2"]["risk"] == "medium"
            ),
            "low_risk": sum(
                1 for b in scored if b["news2"]["risk"] == "low"
            ),
            "escalated": sum(1 for b in beds if b["escalated"]),
            "escalations_raised": sum(
                b["escalations_raised"] for b in beds
            ),
            "responses": {
                "responded": len(responded),
                "unresponded": sum(
                    1
                    for e in episodes
                    if e["responded_t"] is None
                    and e["cleared_t"] is not None
                ),
                "pending": sum(
                    1
                    for e in episodes
                    if e["responded_t"] is None
                    and e["cleared_t"] is None
                ),
                "mean_time_to_respond_s": (
                    round(sum(times) / len(times), 3) if times else None
                ),
                "max_time_to_respond_s": max(times) if times else None,
            },
            "trend_alerts": sum(
                1 for b in beds if b["trend_alert_active"]
            ),
            "trend_alerts_raised": sum(
                b["trend_alerts_raised"] for b in beds
            ),
            "trend_alert_hooks": self.trend_alert_hooks.summary(),
            "escalation_hooks": self.escalation_hooks.summary(),
        }

    def news2_history_summary(self, bed_id=None):
        """NEWS2 score history, for one bed or the whole ward.

        ``bed_id`` picks one bed (``ValueError`` when the ward has
        no such bed); None returns every bed. Each bed's entry is
        its score history with peak score and risk and the latest
        score, as returned by
        :meth:`Monitor.news2_history_detail`.
        """
        if bed_id is None:
            return {
                "t": self.elapsed,
                "beds": [
                    m.news2_history_detail(self.elapsed)
                    for m in self.monitors
                ],
            }
        return self._monitor(bed_id).news2_history_detail(self.elapsed)

    def discharge_summary(self):
        """Planned discharges, cleaning capacity, and queued beds.

        ``cleaning_queue`` lists the beds awaiting a cleaning slot
        in the order cleaning will start them — discharge-priority
        order, safest first (see :meth:`cleaning_queue`) — so a
        client can see who is next for a slot, not just how many
        are waiting.
        """
        summary = self.discharge_planner.summary(self.elapsed)
        summary["cleaning_capacity"] = self.cleaning_capacity
        summary["cleaning_active"] = self.cleaning_active_count(self.elapsed)
        summary["awaiting_cleaning"] = sum(
            1 for m in self.monitors if m.awaiting_cleaning
        )
        summary["cleaning_queue"] = self.cleaning_queue()
        return summary

    def admit(self, bed_id, profile=None, acuity=None, on_oxygen=None, consciousness=None):
        """Admit a new patient to an available bed.

        Raises ValueError if the bed is occupied or still cleaning.
        """
        self._monitor(bed_id).admit(self.elapsed, profile=profile, acuity=acuity, on_oxygen=on_oxygen, consciousness=consciousness)

    def _monitor(self, bed_id):
        if not 0 <= bed_id < self.beds:
            raise ValueError("no bed %r, ward has %d beds" % (bed_id, self.beds))
        return self.monitors[bed_id]

    def turnover_summary(self):
        """Per-bed turnover state and ward totals."""
        beds = [m.turnover_summary(self.elapsed) for m in self.monitors]
        return {
            "t": self.elapsed,
            "beds": beds,
            "occupied": sum(1 for b in beds if b["state"] == "occupied"),
            "cleaning": sum(1 for b in beds if b["state"] == "cleaning"),
            "blocked": sum(1 for b in beds if b["state"] == "blocked"),
            "available": sum(1 for b in beds if b["state"] == "available"),
            "awaiting_cleaning": sum(
                1 for b in beds if b["awaiting_cleaning"]
            ),
            "total_turnovers": sum(b["turnovers"] for b in beds),
            "total_cleaning_s": round(sum(b["total_cleaning_s"] for b in beds), 3),
            "total_discharge_delays": sum(b["discharge_delays"] for b in beds),
            "total_discharge_delay_s": round(
                sum(b["total_discharge_delay_s"] for b in beds), 3
            ),
        }

    def meds_summary(self):
        """Per-bed medication state and ward dose totals.

        ``missed_dose_hooks`` carries the missed-dose hook totals:
        whether any hook is configured, how many hook deliveries
        were invoked, how many failed, how many retries were spent,
        and how many deliveries eventually landed.
        """
        beds = [m.meds.summary(self.elapsed) for m in self.monitors]
        return {
            "t": self.elapsed,
            "beds": beds,
            "total_given": sum(b["doses"]["given"] for b in beds),
            "total_late": sum(b["doses"]["late"] for b in beds),
            "total_missed": sum(b["doses"]["missed"] for b in beds),
            "total_stockout": sum(b["doses"]["stockout"] for b in beds),
            "total_warnings": sum(len(b["warnings"]) for b in beds),
            "missed_dose_hooks": self.missed_dose_hooks.summary(),
        }

    def open_high_episodes(self):
        """Beds with a deterioration episode open at high NEWS2
        risk — the count a staffing escalation is decided on."""
        return [
            m.bed_id for m in self.monitors if m.open_high_episode()
        ]

    def staffing_summary(self):
        """Current staffing state and care metrics.

        ``escalation`` carries the NEWS2 staffing-escalation
        state: whether one is configured and active, the
        threshold and extra nurses in force, how many high-risk
        episodes are open, and every escalation raised so far.
        """
        occupied = sum(1 for m in self.monitors if m.admitted)
        summary = self.staffing.summary(self.elapsed, occupied)
        summary["escalation"] = self.staffing_escalation.summary(
            self.elapsed, self.open_high_episodes()
        )
        return summary

    def triage_summary(self):
        """Current intake queue state and wait metrics."""
        return self.triage.summary(self.elapsed)

    def arrivals_summary(self):
        """Live arrivals view: rate, last-hour arrivals, per-hour
        counts, and diversion status."""
        return self.triage.arrivals_view(self.elapsed)

    def triage_arrive(self, acuity, profile="stable"):
        """Add a patient arriving now to the intake queue.

        Returns True if they joined the queue, False if they were
        diverted because it is full.
        """
        hour = (self.staffing.start_hour + self.elapsed / 3600.0) % 24
        return self.triage.arrive(self.elapsed, acuity, profile, hour=hour)

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        interval = 1.0 / self.rate_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            self.tick()
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
            self._thread = None
