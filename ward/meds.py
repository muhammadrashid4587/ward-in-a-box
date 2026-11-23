"""Medication administration: scheduled doses, late and missed doses,
and interaction warnings.

Each patient can have a list of scheduled medications, set in the
scenario per patient:

    {"bed": 0, "profile": "stable",
     "meds": [{"name": "metoprolol", "every_s": 3600,
               "first_at": 0, "grace_s": 600}]}

A dose due at ``t`` is administered ``BASE_ADMIN_S * load_factor``
seconds later — the same staffing load factor that slows alarm
acknowledgement, so a thin night shift delays medication too. A dose
given more than ``LATE_AFTER_S`` after it was due counts as late; a
dose whose administration would land past ``due + grace_s`` is missed
instead, and the miss raises a ``missed_dose`` alarm on the bed until
that medication's next dose is given. The model is deterministic:
given the roster and the schedule, a seeded run shows the staffing
effect and nothing else.

Interaction warnings come from a small table of known interacting
pairs. They are warnings only — the doses still happen — because a
simulator should surface the risk, not decide the prescription.
"""

# Seconds for a nurse to administer a due dose with a full roster.
BASE_ADMIN_S = 30.0
# A dose given this long after it was due counts as late.
LATE_AFTER_S = 300.0
# Default window after the due time in which a dose can still be given.
DEFAULT_GRACE_S = 600.0

# Known interacting pairs (lower-case names) and what they risk.
# A deliberately small table of well-known interactions, not a formulary.
INTERACTIONS = {
    frozenset({"warfarin", "aspirin"}): "bleeding risk",
    frozenset({"metoprolol", "verapamil"}): "bradycardia risk",
    frozenset({"insulin", "metoprolol"}): "masks hypoglycaemia",
    frozenset({"lisinopril", "spironolactone"}): "hyperkalaemia risk",
    frozenset({"digoxin", "furosemide"}): "arrhythmia risk",
}


def validate_med(med):
    """Validate one medication dict, returning a normalised copy.

    ``name`` is a non-empty string, ``every_s`` the interval between
    doses in seconds (> 0), ``first_at`` the elapsed seconds of the
    first dose (>= 0, default 0), ``grace_s`` the window after the due
    time in which the dose can still be given (> 0, default 600).
    """
    if not isinstance(med, dict):
        raise ValueError("med must be an object, got %r" % (med,))
    name = med.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("med name must be a non-empty string, got %r" % (name,))
    every_s = med.get("every_s")
    if not isinstance(every_s, (int, float)) or isinstance(every_s, bool) or every_s <= 0:
        raise ValueError("med %s: every_s must be > 0, got %r" % (name, every_s))
    first_at = med.get("first_at", 0)
    if not isinstance(first_at, (int, float)) or isinstance(first_at, bool) or first_at < 0:
        raise ValueError("med %s: first_at must be >= 0, got %r" % (name, first_at))
    grace_s = med.get("grace_s", DEFAULT_GRACE_S)
    if not isinstance(grace_s, (int, float)) or isinstance(grace_s, bool) or grace_s <= 0:
        raise ValueError("med %s: grace_s must be > 0, got %r" % (name, grace_s))
    return {
        "name": name.strip(),
        "every_s": float(every_s),
        "first_at": float(first_at),
        "grace_s": float(grace_s),
    }


def interaction_warnings(meds):
    """Warnings for interacting pairs in a bed's medication list.

    Each warning is ``{"meds": [name, name], "warning": text}``, sorted
    by the pair. Names are matched case-insensitively.
    """
    names = [m["name"].lower() for m in meds]
    warnings = []
    for pair, text in INTERACTIONS.items():
        a, b = sorted(pair)
        if a in names and b in names:
            warnings.append({"meds": [a, b], "warning": text})
    return sorted(warnings, key=lambda w: w["meds"])


class MedSchedule:
    """One bed's medication schedule: due doses, given/late/missed.

    ``tick(now, load_factor)`` advances the schedule: doses whose due
    time has passed become pending, and pending doses resolve when
    their administration time (or the end of their grace window)
    arrives. ``missed_active`` holds the names of medications with a
    missed dose not yet followed by a given one — the monitor turns
    that into a ``missed_dose`` alarm. A dose that falls due while its
    medication is stocked out (``stockout_meds`` passed to ``tick``)
    is skipped at once: it is recorded in the history with status
    ``stockout``, counted separately, and also flagged in
    ``missed_active``, because a skipped dose is a missed dose from
    the patient's point of view.
    """

    def __init__(self, bed_id, meds=None):
        self.bed_id = bed_id
        self.meds = [validate_med(m) for m in (meds or [])]
        names = [m["name"].lower() for m in self.meds]
        if len(set(names)) != len(names):
            raise ValueError("bed %d: duplicate medication names" % bed_id)
        self.warnings = interaction_warnings(self.meds)
        self._next_due = {m["name"]: m["first_at"] for m in self.meds}
        self._grace = {m["name"]: m["grace_s"] for m in self.meds}
        self._every = {m["name"]: m["every_s"] for m in self.meds}
        self._pending = []  # {"med", "due_t", "resolve_t", "outcome", "delay_s"}
        self.history = []  # {"med", "due_t", "admin_t", "status", "delay_s"}
        self.given = 0
        self.late = 0
        self.missed = 0
        self.stockout = 0
        self.missed_active = set()
        # Medications currently stocked out (set by tick's caller).
        self.stockout_active = set()

    def reset(self):
        """Clear all schedule state (a new patient takes the bed)."""
        self._next_due = {m["name"]: m["first_at"] for m in self.meds}
        self._pending = []
        self.history = []
        self.given = 0
        self.late = 0
        self.missed = 0
        self.stockout = 0
        self.missed_active = set()
        self.stockout_active = set()

    def tick(self, now, load_factor, stockout_meds=()):
        """Advance doses due by ``now`` under this staffing load.

        ``stockout_meds`` is the set of medication names (lower-case)
        the pharmacy currently has none of: doses of those that fall
        due are skipped, not queued.
        """
        now = round(now, 3)
        canonical = {m["name"].lower(): m["name"] for m in self.meds}
        self.stockout_active = {
            canonical[m] for m in (s.lower() for s in stockout_meds)
            if m in canonical
        }
        for med in self.meds:
            name = med["name"]
            while self._next_due[name] <= now:
                due_t = round(self._next_due[name], 3)
                self._next_due[name] = round(due_t + self._every[name], 3)
                if name in self.stockout_active:
                    self.history.append(
                        {"med": name, "due_t": due_t, "admin_t": None,
                         "status": "stockout", "delay_s": None}
                    )
                    self.stockout += 1
                    self.missed_active.add(name)
                    continue
                delay = round(BASE_ADMIN_S * load_factor, 3)
                admin_t = round(due_t + delay, 3)
                if admin_t <= round(due_t + self._grace[name], 3):
                    self._pending.append(
                        {"med": name, "due_t": due_t, "resolve_t": admin_t,
                         "outcome": "given", "delay_s": delay}
                    )
                else:
                    self._pending.append(
                        {"med": name, "due_t": due_t,
                         "resolve_t": round(due_t + self._grace[name], 3),
                         "outcome": "missed", "delay_s": None}
                    )
        for pending in list(self._pending):
            if pending["resolve_t"] <= now:
                self._pending.remove(pending)
                if pending["outcome"] == "given":
                    status = "late" if pending["delay_s"] > LATE_AFTER_S else "given"
                    self.history.append(
                        {"med": pending["med"], "due_t": pending["due_t"],
                         "admin_t": pending["resolve_t"], "status": status,
                         "delay_s": pending["delay_s"]}
                    )
                    self.given += 1
                    if status == "late":
                        self.late += 1
                    self.missed_active.discard(pending["med"])
                else:
                    self.history.append(
                        {"med": pending["med"], "due_t": pending["due_t"],
                         "admin_t": None, "status": "missed", "delay_s": None}
                    )
                    self.missed += 1
                    self.missed_active.add(pending["med"])

    def summary(self, now):
        """Per-bed medication state and dose counts as plain JSON data."""
        return {
            "bed": self.bed_id,
            "meds": [m["name"] for m in self.meds],
            "warnings": [dict(w) for w in self.warnings],
            "doses": {
                "given": self.given,
                "late": self.late,
                "missed": self.missed,
                "stockout": self.stockout,
                "pending": len(self._pending),
            },
            "missed_active": sorted(self.missed_active),
            "stockouts": sorted(self.stockout_active),
            "history": [dict(h) for h in self.history],
        }
