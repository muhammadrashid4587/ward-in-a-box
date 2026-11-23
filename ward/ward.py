"""The ward: N beds, each with a monitor, plus a chaos engine."""

import random
import threading
import time

from .chaos import ChaosEngine
from .discharge import DischargePlanner
from .monitor import Monitor
from .news2 import validate_consciousness, validate_consciousness_dynamics
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

    def __init__(self, beds=12, rate_hz=1.0, profiles=None, chaos=None, seed=None, timeline=None, recorder=None, acuities=None, staffing=None, cleaning_s=30.0, meds=None, triage=None, discharge_planning=None, cleaning_capacity=None, discharge_readiness=None, oxygen=None, consciousness=None, consciousness_dynamics=None):
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
            self.triage = TriageQueue(rng=self.rng)
        elif isinstance(triage, TriageQueue):
            self.triage = triage
            # A triage queue built elsewhere (e.g. by load_scenario)
            # gets this ward's seeded RNG for its walk-in arrivals,
            # so a seeded run reproduces the same arrivals.
            if self.triage.poisson is not None:
                self.triage.poisson.rng = self.rng
        elif isinstance(triage, dict):
            kwargs = dict(triage)
            kwargs.setdefault("rng", self.rng)
            self.triage = TriageQueue(**kwargs)
        else:
            raise ValueError("triage must be a TriageQueue or a dict, got %r" % (triage,))
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
            monitor = Monitor(i, profile, rng=self.rng, acuity=acuity, meds=bed_meds, on_oxygen=on_oxygen, consciousness=bed_consciousness, consciousness_dynamics=bed_dynamics)
            # With a timeline, beds start unadmitted and come online as
            # the timeline admits them; without one, all beds start open.
            if timeline is not None:
                monitor.admitted = False
            self.monitors.append(monitor)
        self.chaos = ChaosEngine(chaos or [])
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
        self.chaos.apply(self, self.elapsed)
        # Discharge planning: orders due now are placed, and orders
        # whose expected time has arrived are discharged — or queued
        # behind a full cleaning roster until a slot frees up.
        self.discharge_planner.tick(self.elapsed, self)
        # Cleaning slots freed by now (a bed's turnover ends when
        # ``now`` reaches its cleaning time, by the state check, not
        # by the monitor loop below) go to queued discharges,
        # earliest expected discharge first — before the monitor loop
        # runs, so the snapshot is consistent within the tick.
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
            r = m.tick(self.elapsed, dt)
            if r is None:
                entry = {"bed": m.bed_id, "online": False, "t": self.elapsed}
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
        # order already covers the bed. Placed after the monitor loop
        # so this tick's readiness already reflects the latest vitals
        # and doses, and before the snapshot so the order is in it.
        if self.readiness.auto_order:
            for m in self.monitors:
                if (
                    m.admitted
                    and m.discharge_ready(
                        self.elapsed, self.readiness.stable_window_s
                    )
                    and not self.discharge_planner.has_live_order(m.bed_id)
                ):
                    self.discharge_planner.place_order(
                        m.bed_id,
                        self.elapsed,
                        lead_time_s=self.readiness.auto_order_lead_time_s,
                        source="auto",
                    )
        snap = {"t": self.elapsed, "beds": readings, "staffing": self.staffing.summary(self.elapsed, occupied), "turnover": self.turnover_summary(), "meds": self.meds_summary(), "triage": self.triage_summary(), "arrivals": self.arrivals_summary(), "discharge": self.discharge_summary(), "discharge_readiness": self.readiness_summary(), "early_warning": self.early_warning_summary()}
        with self._lock:
            self.latest = snap
        if self.recorder is not None:
            self.recorder.record(snap)
        return snap

    def snapshot(self):
        """Most recent snapshot, or None before the first tick."""
        with self._lock:
            return self.latest

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

    def discharge(self, bed_id):
        """Discharge the patient in ``bed_id``.

        The bed goes into cleaning for ``cleaning_s`` seconds and
        cannot take a new admission until turnover finishes. When
        ``cleaning_capacity`` is set and every cleaning slot is busy,
        the discharge is queued instead: the patient stays in the bed,
        still monitored, until a slot frees up (see
        :meth:`_start_queued_cleanings`). A bed whose discharge is
        already queued is left alone.
        """
        m = self._monitor(bed_id)
        if m.awaiting_cleaning:
            return
        if not m.admitted:
            raise ValueError("bed %d is not occupied" % bed_id)
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
        once, ``"queued"`` when cleaning is backed up and the patient
        stays in the bed, or ``"cancelled"`` when the patient already
        left (a chaos discharge fired first, say).
        """
        m = self._monitor(bed_id)
        if m.awaiting_cleaning:
            return "queued"
        if not m.admitted:
            return "cancelled"
        if (
            self.cleaning_capacity is not None
            and self.cleaning_active_count(self.elapsed)
            >= self.cleaning_capacity
        ):
            m.queue_discharge(self.elapsed, expected_t)
            return "queued"
        m.discharge(self.elapsed, self.cleaning_s)
        return "discharged"

    def _start_queued_cleanings(self):
        """Start cleaning for discharges queued behind a full roster.

        Earliest expected discharge first; each one frees its patient
        from the bed and tells the discharge planner the order is
        complete, with its delay.
        """
        queued = [m for m in self.monitors if m.awaiting_cleaning]
        queued.sort(
            key=lambda m: (
                m.pending_since if m.pending_since is not None else 0,
                m.bed_id,
            )
        )
        for m in queued:
            if (
                self.cleaning_capacity is not None
                and self.cleaning_active_count(self.elapsed)
                >= self.cleaning_capacity
            ):
                break
            m.begin_cleaning(self.elapsed, self.cleaning_s)
            self.discharge_planner.complete(m.bed_id, self.elapsed)

    def readiness_summary(self):
        """Per-patient discharge-readiness flags and ward totals."""
        beds = [
            m.readiness_detail(
                self.elapsed,
                self.readiness.stable_window_s,
                auto_ordered=self.discharge_planner.has_auto_order(m.bed_id),
            )
            for m in self.monitors
        ]
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
        }

    def early_warning_summary(self):
        """Per-patient NEWS2 scores and deterioration episodes."""
        beds = [m.news2_detail(self.elapsed) for m in self.monitors]
        scored = [b for b in beds if b["news2"] is not None]
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
        }

    def discharge_summary(self):
        """Planned discharges, cleaning capacity, and queued beds."""
        summary = self.discharge_planner.summary(self.elapsed)
        summary["cleaning_capacity"] = self.cleaning_capacity
        summary["cleaning_active"] = self.cleaning_active_count(self.elapsed)
        summary["awaiting_cleaning"] = sum(
            1 for m in self.monitors if m.awaiting_cleaning
        )
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
        """Per-bed medication state and ward dose totals."""
        beds = [m.meds.summary(self.elapsed) for m in self.monitors]
        return {
            "t": self.elapsed,
            "beds": beds,
            "total_given": sum(b["doses"]["given"] for b in beds),
            "total_late": sum(b["doses"]["late"] for b in beds),
            "total_missed": sum(b["doses"]["missed"] for b in beds),
            "total_stockout": sum(b["doses"]["stockout"] for b in beds),
            "total_warnings": sum(len(b["warnings"]) for b in beds),
        }

    def staffing_summary(self):
        """Current staffing state and care metrics."""
        occupied = sum(1 for m in self.monitors if m.admitted)
        return self.staffing.summary(self.elapsed, occupied)

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
