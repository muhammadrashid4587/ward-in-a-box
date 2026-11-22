"""The ward: N beds, each with a monitor, plus a chaos engine."""

import random
import threading
import time

from .chaos import ChaosEngine
from .monitor import Monitor
from .staffing import Staffing
from .timeline import Timeline
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

    def __init__(self, beds=12, rate_hz=1.0, profiles=None, chaos=None, seed=None, timeline=None, recorder=None, acuities=None, staffing=None):
        if not 1 <= beds <= 64:
            raise ValueError("beds must be 1..64")
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        profiles = profiles or []
        for p in profiles:
            if p not in PROFILES:
                raise ValueError("unknown profile: %r" % p)
        acuities = acuities or []
        for a in acuities:
            validate_acuity(a)
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
        self.rng = random.Random(seed)
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
            monitor = Monitor(i, profile, rng=self.rng, acuity=acuity)
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
        if self.timeline is not None:
            for i, m in enumerate(self.monitors):
                if not m.admitted:
                    admitted_at = self.timeline.admission_time(i)
                    if admitted_at is not None and admitted_at <= self.elapsed:
                        m.admitted = True
        readings = []
        online = []
        for m in self.monitors:
            r = m.tick(self.elapsed, dt)
            if r is None:
                entry = {"bed": m.bed_id, "online": False, "t": self.elapsed}
                if not m.admitted:
                    entry["admitted"] = False
                readings.append(entry)
            else:
                r["online"] = True
                readings.append(r)
                online.append(m.bed_id)
        occupied = sum(1 for m in self.monitors if m.admitted)
        self.staffing.tick(self.elapsed, occupied, online)
        for m in self.monitors:
            for alarm in m.raised_this_tick:
                self.staffing.alarm_raised(m.bed_id, alarm, self.elapsed, occupied)
        snap = {"t": self.elapsed, "beds": readings, "staffing": self.staffing.summary(self.elapsed, occupied)}
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

    def staffing_summary(self):
        """Current staffing state and care metrics."""
        occupied = sum(1 for m in self.monitors if m.admitted)
        return self.staffing.summary(self.elapsed, occupied)

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
