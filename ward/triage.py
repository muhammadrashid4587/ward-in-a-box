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

Boarding: when the queue is full, a patient who would be diverted
boards instead — they wait in the ED for an inpatient bed — up to
``boarding_capacity`` (0 by default, so boarding is off unless a
scenario turns it on). A boarded patient is already admitted to the
hospital, so when a bed frees up they take it before a patient still
waiting in the intake queue; within each group the usual triage
priority applies. Boarded patients raise the same wait-time alarms
as queued ones, and their waits are tracked separately.

Boarding-full alarm: when boarding is full and a new arrival is
diverted because both the queue and boarding are full, a
boarding-full alarm raises — one episode per continuous full period.
Further diversions while the alarm is active count against the
episode instead of raising a new one, and the alarm clears when a
boarded patient is admitted to a bed and a boarding spot frees up.
Boarding off means no alarm: a diversion with no boarding capacity
is a plain diversion, not a boarding failure.
"""

from .arrivals import ArrivalModel
from .vitals import PROFILES, validate_acuity

# Default intake queue capacity when a scenario does not set one.
DEFAULT_CAPACITY = 8

# Default boarding capacity: 0, boarding off unless a scenario
# enables it. Boarding is an overflow for the intake queue, not a
# second queue — it only ever holds patients the queue could not.
DEFAULT_BOARDING_CAPACITY = 0

# Default wait-time alarm thresholds, seconds per acuity: a waiting
# patient raises a wait-time alarm once they have waited this long.
# Critical patients get five minutes, low-acuity patients an hour —
# the same shape as real ED triage targets.
DEFAULT_WAIT_THRESHOLDS = {1: 300.0, 2: 900.0, 3: 1800.0, 4: 3600.0}


class TriageQueue:
    """The ED intake queue for one ward."""

    def __init__(self, capacity=DEFAULT_CAPACITY, boarding_capacity=DEFAULT_BOARDING_CAPACITY, arrivals=None, wait_thresholds=None, poisson=None, rng=None, seed=None):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("triage capacity must be an int >= 1, got %r" % (capacity,))
        if (
            not isinstance(boarding_capacity, int)
            or isinstance(boarding_capacity, bool)
            or boarding_capacity < 0
        ):
            raise ValueError(
                "triage boarding_capacity must be an int >= 0, got %r"
                % (boarding_capacity,)
            )
        self.capacity = capacity
        self.boarding_capacity = boarding_capacity
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
        # Arrivals per ward clock hour (0-23): every patient who
        # showed up in that hour, whether they joined the queue or
        # were diverted — the ED load shape over the day.
        self.arrivals_by_hour = {}
        # Arrival times (ward elapsed seconds) of every patient who
        # showed up, and of the ones diverted — the live arrivals
        # view counts the trailing hour from these.
        self._arrival_times = []
        self._diverted_times = []
        # Waiting patients, in arrival order: {"arrival_t", "acuity",
        # "profile", "wait_alarm_t"} — "wait_alarm_t" is the time the
        # wait-time alarm was raised, or None while under threshold.
        self.waiting = []
        # Boarded patients, same shape as waiting: they arrived when
        # the queue was full and wait in the ED for an inpatient bed.
        self.boarding = []
        self.boarded = 0
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
        # Wait times (seconds) of patients admitted so far, split by
        # where they waited: the intake queue, or boarding.
        self._waits = []
        self._boarding_waits = []
        # Wait-time alarm events: {"acuity", "profile", "arrival_t",
        # "raised_t", "cleared_t"} — cleared when the patient is
        # admitted (or None while the patient still waits).
        self.wait_alarm_history = []
        self.wait_alarms_raised = 0
        # Boarding-full alarm episodes: {"raised_t", "cleared_t",
        # "diverted"} — raised when a patient is diverted while
        # boarding is full, cleared when a boarded patient is
        # admitted and a boarding spot frees up; "diverted" counts
        # every diversion during the episode, including the one that
        # raised it. "cleared_t" is None while the episode is active.
        self.boarding_full_alarm_history = []
        self.boarding_full_alarms_raised = 0

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

    def arrive(self, now, acuity, profile="stable", hour=None):
        """Add a patient arriving at ``now`` to the queue.

        Returns True if the patient joined the queue (or boarded when
        the queue was full and boarding had room), False if they were
        diverted because both the queue and boarding are full.
        ``hour`` is the ward clock hour (0-23) the arrival falls in;
        when given, the arrival counts toward that hour's total even
        if diverted.
        """
        acuity = validate_acuity(acuity)
        if profile not in PROFILES:
            raise ValueError("unknown profile: %r" % profile)
        now = round(now, 3)
        self._arrival_times.append(now)
        if hour is not None:
            h = int(hour) % 24
            self.arrivals_by_hour[h] = self.arrivals_by_hour.get(h, 0) + 1
        if len(self.waiting) >= self.capacity:
            if len(self.boarding) < self.boarding_capacity:
                self.boarding.append(
                    {
                        "arrival_t": now,
                        "acuity": acuity,
                        "profile": profile,
                        "wait_alarm_t": None,
                    }
                )
                self.boarded += 1
                return True
            self.diverted += 1
            self._diverted_times.append(now)
            self._boarding_full_diverted(now)
            return False
        self.waiting.append(
            {
                "arrival_t": now,
                "acuity": acuity,
                "profile": profile,
                "wait_alarm_t": None,
            }
        )
        return True

    def _boarding_full_alarm(self):
        """The active boarding-full alarm episode, or None.

        An episode is active from the first diversion while boarding
        is full until a boarded patient is admitted and a boarding
        spot frees up; its "cleared_t" is None while it is active.
        """
        if self.boarding_full_alarm_history:
            last = self.boarding_full_alarm_history[-1]
            if last["cleared_t"] is None:
                return last
        return None

    def _boarding_full_diverted(self, now):
        """Record a diversion that happened while boarding is full.

        Raises a boarding-full alarm episode when none is active;
        when one is active the diversion counts against it instead
        of raising a second alarm. Boarding off (capacity 0) means
        no alarm: the diversion is a plain diversion.
        """
        if self.boarding_capacity <= 0:
            return
        if len(self.boarding) < self.boarding_capacity:
            return
        active = self._boarding_full_alarm()
        if active is not None:
            active["diverted"] += 1
            return
        self.boarding_full_alarm_history.append(
            {"raised_t": now, "cleared_t": None, "diverted": 1}
        )
        self.boarding_full_alarms_raised += 1

    def _clear_boarding_full_alarm(self, now):
        """Clear the active boarding-full alarm, if any.

        Called when a boarded patient is admitted: a boarding spot
        has freed up, so the ED is no longer turning patients away
        for lack of boarding room.
        """
        active = self._boarding_full_alarm()
        if active is not None:
            active["cleared_t"] = round(now, 3)

    def process_arrivals(self, now, hour=None):
        """Move scheduled arrivals due at ``now`` into the queue.

        ``hour`` is the ward clock hour (0-23) the arrivals fall in,
        for the per-hour arrival counts.
        """
        while self._next_scheduled < len(self._scheduled):
            arrival = self._scheduled[self._next_scheduled]
            if arrival["at"] > now:
                break
            self._next_scheduled += 1
            self.arrive(
                arrival["at"], arrival["acuity"], arrival["profile"], hour=hour
            )

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
            self.arrive(
                arrival["arrival_t"],
                arrival["acuity"],
                arrival["profile"],
                hour=hour,
            )
        return arrivals

    def check_wait_alarms(self, now):
        """Raise a wait-time alarm for any patient past their threshold.

        A patient's alarm raises at exactly ``arrival_t + threshold``
        for their acuity, once — later ticks do not re-raise it. The
        alarm clears when the patient is admitted (see
        :meth:`admit_available`). Boarded patients raise the same
        alarms as queued ones; their history events are marked
        ``"boarding": true``. Returns the patients whose alarm was
        raised by this call.
        """
        now = round(now, 3)
        raised = []
        for patient, boarding in [(p, False) for p in self.waiting] + [
            (p, True) for p in self.boarding
        ]:
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
                        "boarding": boarding,
                    }
                )
                self.wait_alarms_raised += 1
                raised.append(patient)
        return raised

    def _priority_order(self, patients=None):
        """Waiting patients, highest triage priority first.

        Lowest acuity number first (1 is critical), earliest arrival
        first within an acuity. Defaults to the intake queue; pass
        the boarding list to order boarded patients the same way.
        """
        return sorted(
            self.waiting if patients is None else patients,
            key=lambda p: (p["acuity"], p["arrival_t"]),
        )

    def admit_available(self, now, ward):
        """Admit waiting patients into the ward's available beds.

        Returns the list of bed ids filled this call. A patient is
        admitted only to a bed in state ``"available"`` — occupied
        beds stay occupied and cleaning beds stay cleaning. Boarded
        patients take freed beds before queued ones, in triage
        priority order within each group.
        """
        filled = []
        while True:
            boarding = self._priority_order(self.boarding)
            waiting = self._priority_order()
            if boarding:
                patient = boarding[0]
                from_boarding = True
            elif waiting:
                patient = waiting[0]
                from_boarding = False
            else:
                break
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
            if from_boarding:
                self.boarding.remove(patient)
                # A boarding spot just freed up: a boarding-full
                # alarm episode, if one is active, clears now.
                self._clear_boarding_full_alarm(now)
            else:
                self.waiting.remove(patient)
            if patient["wait_alarm_t"] is not None:
                # The patient's wait-time alarm clears on admission.
                for event in reversed(self.wait_alarm_history):
                    if (
                        event["cleared_t"] is None
                        and event["arrival_t"] == patient["arrival_t"]
                        and event["acuity"] == patient["acuity"]
                        and event.get("boarding", False) == from_boarding
                    ):
                        event["cleared_t"] = round(now, 3)
                        break
            ward.admit(bed_id, profile=patient["profile"], acuity=patient["acuity"])
            wait = round(now - patient["arrival_t"], 3)
            if from_boarding:
                self._boarding_waits.append(wait)
            else:
                self._waits.append(wait)
            self.admitted += 1
            filled.append(bed_id)
        return filled

    def arrivals_view(self, now):
        """Live arrivals view for the dashboard.

        The current walk-in rate, arrivals in the last hour of ward
        time, arrivals per ward clock hour so far, and diversion
        status — whether new arrivals are being turned away, which
        now means the queue is full and boarding is full too. Boarding
        counts ride along: how many patients are boarding now, how
        many have boarded in total, and how many boarding spots the
        ED has. ``arrivals_total`` counts every patient who showed up
        (scheduled, walk-in, or added live), including diverted ones;
        ``arrivals_by_hour`` only counts arrivals given a ward clock
        hour. The last-hour window is ``(now - 3600, now]``.
        """
        now = round(now, 3)
        window_start = now - 3600.0
        recent = [t for t in self._arrival_times if t > window_start]
        diverted_recent = [
            t for t in self._diverted_times if t > window_start
        ]
        return {
            "t": now,
            "arrival_rate_per_hour": self._current_rate,
            "arrivals_last_hour": len(recent),
            "arrivals_total": len(self._arrival_times),
            "arrivals_by_hour": {
                str(h): n for h, n in sorted(self.arrivals_by_hour.items())
            },
            "waiting_total": len(self.waiting),
            "capacity": self.capacity,
            "boarding_total": len(self.boarding),
            "boarding_capacity": self.boarding_capacity,
            "boarded": self.boarded,
            "admitted": self.admitted,
            "diverted": self.diverted,
            "diverted_last_hour": len(diverted_recent),
            "diverting": len(self.waiting) >= self.capacity
            and len(self.boarding) >= self.boarding_capacity,
            # Boarding-full alarm: active while an episode raised by
            # a diversion with boarding full has not yet cleared.
            "boarding_full_alarm": self._boarding_full_alarm() is not None,
            "boarding_full_alarms_raised": self.boarding_full_alarms_raised,
        }

    def summary(self, now):
        """Queue state and wait metrics as plain JSON data."""
        now = round(now, 3)
        waits = self._waits
        boarding_waits = self._boarding_waits
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
            "boarding": [
                {
                    "acuity": p["acuity"],
                    "profile": p["profile"],
                    "arrival_t": p["arrival_t"],
                    "wait_s": round(now - p["arrival_t"], 3),
                    "wait_threshold_s": self.wait_thresholds[p["acuity"]],
                    "wait_alarm": p["wait_alarm_t"] is not None,
                    "wait_alarm_t": p["wait_alarm_t"],
                }
                for p in self._priority_order(self.boarding)
            ],
            "boarding_total": len(self.boarding),
            "boarding_capacity": self.boarding_capacity,
            "boarded": self.boarded,
            "admitted": self.admitted,
            "diverted": self.diverted,
            "wait_thresholds_s": {
                str(a): self.wait_thresholds[a] for a in (1, 2, 3, 4)
            },
            "wait_alarms_raised": self.wait_alarms_raised,
            "wait_alarms_active": sum(
                1
                for p in list(self.waiting) + list(self.boarding)
                if p["wait_alarm_t"] is not None
            ),
            "wait_alarm_history": [dict(e) for e in self.wait_alarm_history],
            # Boarding-full alarm: episodes raised when a patient was
            # diverted while boarding was full, each with how many
            # diversions happened during it and when it cleared.
            "boarding_full_alarms_raised": self.boarding_full_alarms_raised,
            "boarding_full_alarm_active": (
                self._boarding_full_alarm() is not None
            ),
            "boarding_full_alarm_history": [
                dict(e) for e in self.boarding_full_alarm_history
            ],
            "boarding_full_diverted": sum(
                e["diverted"] for e in self.boarding_full_alarm_history
            ),
            "arrival_rate_per_hour": self._current_rate,
            "arrivals_generated": (
                self.poisson.generated if self.poisson is not None else 0
            ),
            "arrivals_by_hour": {
                str(h): n for h, n in sorted(self.arrivals_by_hour.items())
            },
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
            "mean_boarding_wait_s": (
                round(sum(boarding_waits) / len(boarding_waits), 3)
                if boarding_waits
                else 0.0
            ),
            "max_boarding_wait_s": (
                round(max(boarding_waits), 3) if boarding_waits else 0.0
            ),
            "longest_boarding_wait_s": round(
                now - min(p["arrival_t"] for p in self.boarding), 3
            )
            if self.boarding
            else 0.0,
        }
