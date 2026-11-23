"""ED triage / intake queue: patients arrive, wait, and take beds as they free up.

A patient arrives at triage with an acuity (1 critical to 4 low, the
same scale as the ward's patient acuity) and waits in the intake
queue. When a bed becomes available — after a discharge and its
cleaning — the highest-priority waiting patient is admitted: lowest
acuity number first, earliest arrival first within an acuity. If the
queue is full when a patient arrives, that patient is diverted
(counted, not queued): the ward cannot take them.

Arrivals can be scheduled in a scenario (``{"at": t, "acuity": n,
"profile": ...}``) or added live with :meth:`TriageQueue.arrive`.
"""

from .arrivals import ArrivalModel
from .vitals import PROFILES, validate_acuity

# Default intake queue capacity when a scenario does not set one.
DEFAULT_CAPACITY = 8

# Default wait-time alarm thresholds, seconds per acuity: a waiting
# patient raises a wait-time alarm once they have waited this long.
# Critical patients get five minutes, low-acuity patients an hour —
# the same shape as real ED triage targets.
DEFAULT_WAIT_THRESHOLDS = {1: 300.0, 2: 900.0, 3: 1800.0, 4: 3600.0}


class TriageQueue:
    """The ED intake queue for one ward."""

    def __init__(self, capacity=DEFAULT_CAPACITY, arrivals=None, wait_thresholds=None, poisson=None, rng=None, seed=None):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("triage capacity must be an int >= 1, got %r" % (capacity,))
        self.capacity = capacity
        self.wait_thresholds = self._validate_thresholds(wait_thresholds)
        # Optional walk-in arrival model (a Poisson process): a dict
        # configures one, an ArrivalModel is used as-is. ``rng`` is
        # the random source for the model — the ward passes its own
        # seeded RNG so a seeded run reproduces the same arrivals.
        self.poisson = None
        if poisson is not None:
            if isinstance(poisson, ArrivalModel):
                self.poisson = poisson
            elif isinstance(poisson, dict):
                self.poisson = ArrivalModel(rng=rng, seed=seed, **poisson)
            else:
                raise ValueError(
                    "poisson must be an ArrivalModel or a dict, got %r"
                    % (poisson,)
                )
        # The arrival rate at the ward clock hour of the last tick
        # (None before the first tick with a model).
        self._current_rate = None
        # Waiting patients, in arrival order: {"arrival_t", "acuity",
        # "profile", "wait_alarm_t"} — "wait_alarm_t" is the time the
        # wait-time alarm was raised, or None while under threshold.
        self.waiting = []
        # Scheduled arrivals not yet due, sorted by time.
        self._scheduled = []
        for i, a in enumerate(arrivals or []):
            if not isinstance(a, dict):
                raise ValueError("triage arrival %d: must be an object" % i)
            try:
                at = float(a["at"])
            except KeyError:
                raise ValueError("triage arrival %d: missing 'at'" % i)
            if at < 0:
                raise ValueError("triage arrival %d: 'at' must be >= 0" % i)
            try:
                acuity = validate_acuity(a.get("acuity", 3))
            except ValueError:
                raise ValueError(
                    "triage arrival %d: acuity must be an int in 1..4, got %r"
                    % (i, a.get("acuity"))
                )
            profile = a.get("profile", "stable")
            if profile not in PROFILES:
                raise ValueError(
                    "triage arrival %d: unknown profile: %r" % (i, profile)
                )
            self._scheduled.append(
                {"at": round(at, 3), "acuity": acuity, "profile": profile}
            )
        self._scheduled.sort(key=lambda a: a["at"])
        self._next_scheduled = 0
        self.admitted = 0
        self.diverted = 0
        # Wait times (seconds) of patients admitted so far.
        self._waits = []
        # Wait-time alarm events: {"acuity", "profile", "arrival_t",
        # "raised_t", "cleared_t"} — cleared when the patient is
        # admitted (or None while the patient still waits).
        self.wait_alarm_history = []
        self.wait_alarms_raised = 0

    @staticmethod
    def _validate_thresholds(wait_thresholds):
        """Per-acuity wait thresholds, seconds, with defaults kept.

        Accepts acuity keys as ints or strings (JSON object keys are
        strings); a partial mapping overrides only the acuities it
        names. Every threshold must be a positive number.
        """
        thresholds = dict(DEFAULT_WAIT_THRESHOLDS)
        if wait_thresholds is None:
            return thresholds
        if not isinstance(wait_thresholds, dict):
            raise ValueError(
                "wait_thresholds must be an object, got %r" % (wait_thresholds,)
            )
        for key, value in wait_thresholds.items():
            if isinstance(key, bool):
                raise ValueError(
                    "wait_thresholds: unknown acuity key %r" % (key,)
                )
            if isinstance(key, int):
                acuity_key = key
            elif isinstance(key, str) and key.isdigit():
                acuity_key = int(key)
            else:
                raise ValueError(
                    "wait_thresholds: unknown acuity key %r" % (key,)
                )
            try:
                acuity = validate_acuity(acuity_key)
            except ValueError:
                raise ValueError(
                    "wait_thresholds: unknown acuity key %r" % (key,)
                )
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    "wait_thresholds: acuity %d threshold must be > 0, got %r"
                    % (acuity, value)
                )
            thresholds[acuity] = float(value)
        return thresholds

    def arrive(self, now, acuity, profile="stable"):
        """Add a patient arriving at ``now`` to the queue.

        Returns True if the patient joined the queue, False if they
        were diverted because the queue is full.
        """
        acuity = validate_acuity(acuity)
        if profile not in PROFILES:
            raise ValueError("unknown profile: %r" % profile)
        if len(self.waiting) >= self.capacity:
            self.diverted += 1
            return False
        self.waiting.append(
            {
                "arrival_t": round(now, 3),
                "acuity": acuity,
                "profile": profile,
                "wait_alarm_t": None,
            }
        )
        return True

    def process_arrivals(self, now):
        """Move scheduled arrivals due at ``now`` into the queue."""
        while self._next_scheduled < len(self._scheduled):
            arrival = self._scheduled[self._next_scheduled]
            if arrival["at"] > now:
                break
            self._next_scheduled += 1
            self.arrive(arrival["at"], arrival["acuity"], arrival["profile"])

    def process_poisson_arrivals(self, now, dt, hour):
        """Generate this tick's walk-in arrivals and queue them.

        ``dt`` is the tick length in seconds, ``hour`` the ward clock
        hour (0-23) the arrival rate is read at. Each generated
        arrival joins the queue like any other — or is diverted when
        the queue is full. Returns the generated arrivals. With no
        Poisson model configured this is a no-op.
        """
        if self.poisson is None:
            return []
        self._current_rate = self.poisson.rate_at(hour)
        arrivals = self.poisson.generate(now, dt, hour)
        for arrival in arrivals:
            self.arrive(arrival["arrival_t"], arrival["acuity"], arrival["profile"])
        return arrivals

    def check_wait_alarms(self, now):
        """Raise a wait-time alarm for any patient past their threshold.

        A patient's alarm raises at exactly ``arrival_t + threshold``
        for their acuity, once — later ticks do not re-raise it. The
        alarm clears when the patient is admitted (see
        :meth:`admit_available`). Returns the patients whose alarm
        was raised by this call.
        """
        now = round(now, 3)
        raised = []
        for patient in self.waiting:
            if patient["wait_alarm_t"] is not None:
                continue
            threshold = self.wait_thresholds[patient["acuity"]]
            if now - patient["arrival_t"] >= threshold:
                raised_t = round(patient["arrival_t"] + threshold, 3)
                patient["wait_alarm_t"] = raised_t
                self.wait_alarm_history.append(
                    {
                        "acuity": patient["acuity"],
                        "profile": patient["profile"],
                        "arrival_t": patient["arrival_t"],
                        "raised_t": raised_t,
                        "cleared_t": None,
                    }
                )
                self.wait_alarms_raised += 1
                raised.append(patient)
        return raised

    def _priority_order(self):
        """Waiting patients, highest triage priority first.

        Lowest acuity number first (1 is critical), earliest arrival
        first within an acuity.
        """
        return sorted(
            self.waiting, key=lambda p: (p["acuity"], p["arrival_t"])
        )

    def admit_available(self, now, ward):
        """Admit waiting patients into the ward's available beds.

        Returns the list of bed ids filled this call. A patient is
        admitted only to a bed in state ``"available"`` — occupied
        beds stay occupied and cleaning beds stay cleaning.
        """
        filled = []
        while self.waiting:
            patient = self._priority_order()[0]
            bed_id = None
            for m in ward.monitors:
                # Only a bed that has freed up takes a triage patient:
                # a bed the scenario timeline has not admitted yet is
                # reserved for the timeline, not the intake queue.
                if m.state(now) == "available" and m.turnovers > 0:
                    bed_id = m.bed_id
                    break
            if bed_id is None:
                break
            self.waiting.remove(patient)
            if patient["wait_alarm_t"] is not None:
                # The patient's wait-time alarm clears on admission.
                for event in reversed(self.wait_alarm_history):
                    if (
                        event["cleared_t"] is None
                        and event["arrival_t"] == patient["arrival_t"]
                        and event["acuity"] == patient["acuity"]
                    ):
                        event["cleared_t"] = round(now, 3)
                        break
            ward.admit(bed_id, profile=patient["profile"], acuity=patient["acuity"])
            self._waits.append(round(now - patient["arrival_t"], 3))
            self.admitted += 1
            filled.append(bed_id)
        return filled

    def summary(self, now):
        """Queue state and wait metrics as plain JSON data."""
        now = round(now, 3)
        waits = self._waits
        return {
            "t": now,
            "waiting": [
                {
                    "acuity": p["acuity"],
                    "profile": p["profile"],
                    "arrival_t": p["arrival_t"],
                    "wait_s": round(now - p["arrival_t"], 3),
                    "wait_threshold_s": self.wait_thresholds[p["acuity"]],
                    "wait_alarm": p["wait_alarm_t"] is not None,
                    "wait_alarm_t": p["wait_alarm_t"],
                }
                for p in self._priority_order()
            ],
            "waiting_by_acuity": {
                str(a): sum(1 for p in self.waiting if p["acuity"] == a)
                for a in (1, 2, 3, 4)
            },
            "waiting_total": len(self.waiting),
            "capacity": self.capacity,
            "admitted": self.admitted,
            "diverted": self.diverted,
            "wait_thresholds_s": {
                str(a): self.wait_thresholds[a] for a in (1, 2, 3, 4)
            },
            "wait_alarms_raised": self.wait_alarms_raised,
            "wait_alarms_active": sum(
                1 for p in self.waiting if p["wait_alarm_t"] is not None
            ),
            "wait_alarm_history": [dict(e) for e in self.wait_alarm_history],
            "arrival_rate_per_hour": self._current_rate,
            "arrivals_generated": (
                self.poisson.generated if self.poisson is not None else 0
            ),
            "arrival_hourly_rates": (
                list(self.poisson.hourly_rates)
                if self.poisson is not None
                and self.poisson.hourly_rates is not None
                else None
            ),
            "mean_wait_s": round(sum(waits) / len(waits), 3) if waits else 0.0,
            "max_wait_s": round(max(waits), 3) if waits else 0.0,
            "longest_wait_s": round(
                now - min(p["arrival_t"] for p in self.waiting), 3
            )
            if self.waiting
            else 0.0,
        }
