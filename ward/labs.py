"""Lab results: scheduled results, delayed results, overdue alarms.

Each patient can have a list of lab results they are waiting on,
set in the scenario per patient:

    {"bed": 0, "profile": "stable",
     "labs": [{"name": "bloods", "due_at": 30}]}

A result is expected at ``due_at`` (elapsed seconds) and lands then
— unless a ``lab_delay`` chaos event pushes it out: a result due
inside the event's window lands ``delay_s`` seconds late instead.
While a result is past its due time and still not back, the bed
raises a ``lab_overdue`` alarm, which clears the tick the result
lands. A result still pending — due or overdue, it has not landed
— is an outstanding order, so the patient is not discharge-ready
until every result is back: a ward does not send a patient home
while their bloods are still in the lab.

The model is deterministic: given the schedule and the delay
events, a seeded run shows the lab delay and nothing else. The
lab is a property of the patient, not the monitor: the result
lands whether or not the monitor is online — the lab does not
care if the bedside monitor is dark — but the overdue alarm only
shows in a reading the monitor actually emits, so a monitor that
is offline when a result goes overdue reports it when it comes
back, like any other alarm.
"""

# How many lab events are kept per bed is unbounded: a lab list is
# a fixed schedule, one entry per ordered result, so there is no
# history to cap.


def validate_lab(lab):
    """Validate one lab result dict, returning a normalised copy.

    ``name`` is a non-empty string — the test or panel the result
    is for, for example ``"bloods"`` or ``"troponin"`` — and
    ``due_at`` is the elapsed seconds the result is expected
    (>= 0).
    """
    if not isinstance(lab, dict):
        raise ValueError("lab must be an object, got %r" % (lab,))
    name = lab.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("lab name must be a non-empty string, got %r" % (name,))
    due_at = lab.get("due_at")
    if (
        not isinstance(due_at, (int, float))
        or isinstance(due_at, bool)
        or due_at != due_at
        or due_at in (float("inf"), float("-inf"))
        or due_at < 0
    ):
        raise ValueError(
            "lab %s: due_at must be >= 0, got %r" % (name, due_at)
        )
    return {"name": name.strip(), "due_at": float(due_at)}


class LabSchedule:
    """One bed's lab results: due, landed, delayed, overdue.

    ``tick(now)`` advances the schedule: a result whose landing
    time has arrived lands — recorded with when it landed, and
    clearing its overdue alarm if one had raised — and a result
    still waiting past its due time goes overdue, returning the
    overdue transitions this tick so the ward can fire its
    lab-overdue hooks on them. ``apply_delay``
    pushes a result's landing time out when a ``lab_delay`` event
    covers it. ``overdue`` holds the names of the results
    currently overdue, so the monitor can turn them into a
    ``lab_overdue`` alarm; ``orders_complete`` is True when no
    result is still pending, so a patient with results outstanding
    is never flagged discharge-ready.
    """

    def __init__(self, bed_id, labs=None):
        self.bed_id = bed_id
        self.labs = []
        for lab in [validate_lab(l) for l in (labs or [])]:
            entry = {
                "name": lab["name"],
                "due_at": lab["due_at"],
                # When the result actually lands: the due time
                # until a delay pushes it out.
                "land_t": lab["due_at"],
                "result_t": None,
                "delayed_s": 0.0,
                # When the overdue alarm raised and cleared for
                # this result (None while it never raised, or
                # while it is still overdue).
                "overdue_raised_t": None,
                "overdue_cleared_t": None,
                # Which delay events already pushed this result
                # out, so one event delays it once, not once a
                # tick while its window is open.
                "delayed_by": set(),
            }
            self.labs.append(entry)
        names = [lab["name"].lower() for lab in self.labs]
        if len(set(names)) != len(names):
            raise ValueError("bed %d: duplicate lab names" % bed_id)
        # Results currently overdue (past due, not landed), by
        # name — maintained by tick, so the monitor can read it
        # without recomputing.
        self._overdue = set()
        # How many overdue alarms this stay has raised, and how
        # many results have been delayed at least once, with the
        # total seconds of delay applied — the ward totals.
        self.overdue_raised = 0
        self.delayed = 0
        self.total_delayed_s = 0.0

    def reset(self, labs=None):
        """Clear all schedule state (a new patient takes the bed).

        The new patient's labs, if any, replace the old schedule
        entirely: a result ordered for the departed patient is
        not the new patient's to wait on.
        """
        fresh = LabSchedule(self.bed_id, labs)
        self.labs = fresh.labs
        self._overdue = fresh._overdue
        self.overdue_raised = fresh.overdue_raised
        self.delayed = fresh.delayed
        self.total_delayed_s = fresh.total_delayed_s

    @property
    def overdue(self):
        """Names of the results currently overdue, sorted."""
        return sorted(self._overdue)

    @property
    def pending(self):
        """Names of the results still waiting to land, sorted."""
        return sorted(
            lab["name"] for lab in self.labs if lab["result_t"] is None
        )

    def tick(self, now):
        """Advance the schedule to elapsed time ``now``.

        A result whose landing time has arrived lands first — so
        a result landing exactly on its due time never goes
        overdue — and then any result still waiting past its due
        time goes overdue. Landing and going overdue are both
        recorded with the tick time, so a seeded run reproduces
        them exactly. Returns the overdue transitions this tick
        — one dict per result that went overdue or landed after
        being overdue — so the ward can fire its lab-overdue
        hooks on them: a result goes overdue once, not once a
        tick while it waits, and landing after being overdue
        ends the episode, so a later result going overdue fires
        again (the transitions are per result, not per run).
        """
        now = round(now, 3)
        transitions = []
        for lab in self.labs:
            if lab["result_t"] is None and lab["land_t"] <= now:
                lab["result_t"] = now
                if lab["name"] in self._overdue:
                    self._overdue.discard(lab["name"])
                    lab["overdue_cleared_t"] = now
                    transitions.append(
                        {
                            "event": "lab_overdue_ended",
                            "name": lab["name"],
                            "due_at": lab["due_at"],
                            "land_t": lab["land_t"],
                            "delayed_s": lab["delayed_s"],
                            "result_t": lab["result_t"],
                            "overdue_raised_t": lab[
                                "overdue_raised_t"
                            ],
                            "overdue_cleared_t": lab[
                                "overdue_cleared_t"
                            ],
                        }
                    )
        for lab in self.labs:
            if (
                lab["result_t"] is None
                and lab["due_at"] <= now
                and lab["overdue_raised_t"] is None
            ):
                lab["overdue_raised_t"] = now
                self._overdue.add(lab["name"])
                self.overdue_raised += 1
                transitions.append(
                    {
                        "event": "lab_overdue_started",
                        "name": lab["name"],
                        "due_at": lab["due_at"],
                        "land_t": lab["land_t"],
                        "delayed_s": lab["delayed_s"],
                        "result_t": None,
                        "overdue_raised_t": lab[
                            "overdue_raised_t"
                        ],
                        "overdue_cleared_t": None,
                    }
                )
        return transitions

    def apply_delay(self, event):
        """Push out the results a lab-delay event covers.

        A result due inside the event's window — ``at`` to
        ``at + duration`` — and still waiting to land is delayed
        by the event's ``delay_s``: its landing time moves out
        by that much, and the delay is counted. One event delays
        a result once, however long its window runs, and a result
        that has already landed, or is due outside the window,
        is untouched — the lab cannot delay a result that is
        already back, and a result due after the lab is running
        normally again was never in the slow queue.
        """
        cfg = event.lab
        for lab in self.labs:
            if lab["result_t"] is not None:
                continue
            if lab["name"].lower() != cfg["name"]:
                continue
            if not event.at <= lab["due_at"] < event.end:
                continue
            if id(event) in lab["delayed_by"]:
                continue
            lab["delayed_by"].add(id(event))
            first_delay = lab["delayed_s"] == 0.0
            lab["land_t"] = round(lab["land_t"] + cfg["delay_s"], 3)
            lab["delayed_s"] = round(
                lab["delayed_s"] + cfg["delay_s"], 3
            )
            self.total_delayed_s = round(
                self.total_delayed_s + cfg["delay_s"], 3
            )
            if first_delay:
                self.delayed += 1

    def orders_complete(self):
        """True when no lab result is still outstanding: every
        ordered result has landed. A result still waiting — due
        or overdue — keeps the orders incomplete until it lands,
        so a patient whose bloods are still in the lab is never
        flagged discharge-ready."""
        return all(lab["result_t"] is not None for lab in self.labs)

    def summary(self, now):
        """Per-bed lab state as plain JSON data.

        ``labs`` lists every ordered result — its name, when it
        was due, when it is now expected to land, when it landed
        (null while it is still waiting), how much it has been
        delayed, and when its overdue alarm raised and cleared —
        ``pending`` / ``overdue`` name the results still waiting
        and the ones past due, and the counts say how many have
        landed, been delayed, and gone overdue this stay.
        """
        return {
            "bed": self.bed_id,
            "labs": [
                {
                    "name": lab["name"],
                    "due_at": lab["due_at"],
                    "land_t": lab["land_t"],
                    "result_t": lab["result_t"],
                    "delayed_s": lab["delayed_s"],
                    "overdue_raised_t": lab["overdue_raised_t"],
                    "overdue_cleared_t": lab["overdue_cleared_t"],
                }
                for lab in self.labs
            ],
            "pending": self.pending,
            "overdue": self.overdue,
            "landed": sum(
                1 for lab in self.labs if lab["result_t"] is not None
            ),
            "delayed": self.delayed,
            "total_delayed_s": self.total_delayed_s,
            "overdue_raised": self.overdue_raised,
        }
