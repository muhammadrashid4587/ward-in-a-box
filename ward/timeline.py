"""Scenario timelines: admissions over time with ramp-up.

A timeline is a list of phases. Each phase names a target number of
occupied beds and a time to reach it: starting at ``at``, beds are
admitted one by one over ``ramp_s`` seconds until ``beds`` beds are
occupied (``ramp_s`` 0 admits them all at once, at ``at``). Between
phases the occupancy stays at the last target. Targets only go up:
this is admissions, not discharges.

Bed ``i`` (0-based) is admitted when the occupancy first reaches
``i + 1``, so during a ramp beds come online spread evenly over the
ramp window instead of all at once.
"""


class TimelinePhase:
    """One admission phase: reach ``beds`` occupied beds from ``at``.

    ``ramp_s`` is how long the ramp takes; 0 means instant at ``at``.
    """

    def __init__(self, at, beds, ramp_s=0):
        if at < 0:
            raise ValueError("timeline phase start time must be >= 0")
        if beds < 0:
            raise ValueError("timeline phase beds must be >= 0")
        if ramp_s < 0:
            raise ValueError("timeline phase ramp_s must be >= 0")
        self.at = float(at)
        self.beds = int(beds)
        self.ramp_s = float(ramp_s)

    @property
    def end(self):
        return self.at + self.ramp_s


class Timeline:
    """An ordered list of admission phases."""

    def __init__(self, phases):
        self.phases = sorted((list(phases)), key=lambda p: p.at)
        for prev, cur in zip(self.phases, self.phases[1:]):
            if cur.beds < prev.beds:
                raise ValueError(
                    "timeline targets must not decrease: %d beds at t=%s "
                    "after %d beds at t=%s"
                    % (cur.beds, cur.at, prev.beds, prev.at)
                )

    @property
    def max_beds(self):
        """Highest occupancy any phase targets (0 with no phases)."""
        return self.phases[-1].beds if self.phases else 0

    def occupied_at(self, now):
        """Beds that should be occupied at elapsed time ``now``."""
        occupied = 0
        for phase in self.phases:
            if now < phase.at:
                break
            if phase.ramp_s == 0 or now >= phase.end:
                occupied = phase.beds
            else:
                prev = occupied
                frac = (now - phase.at) / phase.ramp_s
                occupied = int(prev + (phase.beds - prev) * frac)
        return occupied

    def admission_time(self, bed_index):
        """When bed ``bed_index`` (0-based) is admitted, or None if never."""
        target = bed_index + 1
        prev = 0
        for phase in self.phases:
            if prev < target <= phase.beds:
                if phase.ramp_s == 0:
                    return round(phase.at, 3)
                frac = (target - prev) / (phase.beds - prev)
                return round(phase.at + frac * phase.ramp_s, 3)
            prev = phase.beds
        return None
