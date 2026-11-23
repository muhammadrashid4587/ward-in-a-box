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

from .vitals import PROFILES, validate_acuity

# Default intake queue capacity when a scenario does not set one.
DEFAULT_CAPACITY = 8


class TriageQueue:
    """The ED intake queue for one ward."""

    def __init__(self, capacity=DEFAULT_CAPACITY, arrivals=None):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("triage capacity must be an int >= 1, got %r" % (capacity,))
        self.capacity = capacity
        # Waiting patients, in arrival order: {"arrival_t", "acuity", "profile"}.
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
            {"arrival_t": round(now, 3), "acuity": acuity, "profile": profile}
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
            "mean_wait_s": round(sum(waits) / len(waits), 3) if waits else 0.0,
            "max_wait_s": round(max(waits), 3) if waits else 0.0,
            "longest_wait_s": round(
                now - min(p["arrival_t"] for p in self.waiting), 3
            )
            if self.waiting
            else 0.0,
        }
