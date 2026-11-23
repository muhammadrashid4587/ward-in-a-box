"""Record a ward run to a JSONL file, for replay and summary.

Format: the first line is a header object describing the run::

    {"type": "header", "beds": 12, "rate_hz": 1.0, "seed": 42}

followed by one line per tick, the snapshot as returned by
``Ward.tick`` with ``"type": "tick"`` added. Plain JSONL, so any
tool that reads JSON lines can consume a recording without this
package. ``summarize`` and ``iter_ticks`` read a recording back.
"""

import json


class Recorder:
    """Appends tick snapshots to a JSONL recording file.

    Use as a context manager, or call ``close()`` explicitly — the
    header is written on open, so an empty run still records how it
    was configured.
    """

    def __init__(self, path, beds, rate_hz, seed=None):
        self.path = path
        self.ticks = 0
        self._f = open(path, "w")
        self._write(
            {"type": "header", "beds": beds, "rate_hz": rate_hz, "seed": seed}
        )

    def _write(self, obj):
        self._f.write(json.dumps(obj) + "\n")

    def record(self, snapshot):
        """Append one tick snapshot (as returned by ``Ward.tick``)."""
        self._write({"type": "tick", **snapshot})
        self.ticks += 1

    def close(self):
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def iter_ticks(path):
    """Yield the tick snapshots of a recording, in order.

    The header line is skipped; each yielded dict is the snapshot
    exactly as the ward emitted it (with ``"type": "tick"`` removed).
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "tick":
                continue
            obj.pop("type", None)
            yield obj


def summarize(path):
    """Summarise a recording as plain JSON data.

    Counts ticks and the run's time span, per-bed readings, offline
    and stale ticks, and alarms raised (an alarm counts once when it
    first appears in a bed's reading, matching the monitor's
    raise/clear transitions), plus intake totals from the last tick —
    patients admitted and diverted, walk-ins generated, and the
    arrival rate at that hour, with arrivals per ward clock hour so
    the run's load shape over the day is visible, and boarding-full
    alarm episodes — and discharge
    totals from the last tick: planned discharges, how many landed,
    how many were delayed by a backed-up cleaning roster, and the
    total delay — and discharge-readiness totals from the last tick:
    how many occupied beds are flagged ready and how many are not.
    Returns ``None`` fields for a recording with no ticks.
    """
    header = None
    ticks = 0
    first_t = last_t = None
    readings = {}
    offline = {}
    stale = {}
    alarms_raised = 0
    active = {}
    last_triage = None
    last_discharge = None
    last_readiness = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "header":
                header = obj
                continue
            if obj.get("type") != "tick":
                continue
            ticks += 1
            t = obj["t"]
            if first_t is None:
                first_t = t
            last_t = t
            if isinstance(obj.get("triage"), dict):
                last_triage = obj["triage"]
            if isinstance(obj.get("discharge"), dict):
                last_discharge = obj["discharge"]
            if isinstance(obj.get("discharge_readiness"), dict):
                last_readiness = obj["discharge_readiness"]
            for bed in obj["beds"]:
                b = bed["bed"]
                if not bed.get("online", True):
                    offline[b] = offline.get(b, 0) + 1
                    continue
                readings[b] = readings.get(b, 0) + 1
                if bed.get("stale"):
                    stale[b] = stale.get(b, 0) + 1
                current = set(bed.get("alarms") or [])
                alarms_raised += len(current - active.get(b, set()))
                active[b] = current
    return {
        "beds": header["beds"] if header else None,
        "rate_hz": header["rate_hz"] if header else None,
        "seed": header["seed"] if header else None,
        "ticks": ticks,
        "first_t": first_t,
        "last_t": last_t,
        "duration_s": round(last_t - first_t, 3)
        if first_t is not None
        else None,
        "readings_per_bed": readings,
        "offline_ticks_per_bed": offline,
        "stale_ticks_per_bed": stale,
        "alarms_raised": alarms_raised,
        # Intake totals from the last tick: how many patients the
        # queue admitted and diverted, how many walk-ins the Poisson
        # process generated, and the arrival rate at that hour —
        # plus arrivals per ward clock hour and the configured
        # hourly rates, so the run's load shape is visible — and
        # boarding-full alarm episodes: how many raised, how many
        # diversions happened during them, and whether one is still
        # active at the end of the run.
        "triage": (
            {
                "admitted": last_triage["admitted"],
                "diverted": last_triage["diverted"],
                "arrivals_generated": last_triage.get("arrivals_generated", 0),
                "arrival_rate_per_hour": last_triage.get(
                    "arrival_rate_per_hour"
                ),
                "arrivals_by_hour": last_triage.get("arrivals_by_hour", {}),
                "arrival_hourly_rates": last_triage.get(
                    "arrival_hourly_rates"
                ),
                "boarding_full_alarms_raised": last_triage.get(
                    "boarding_full_alarms_raised", 0
                ),
                "boarding_full_alarm_active": last_triage.get(
                    "boarding_full_alarm_active", False
                ),
                "boarding_full_diverted": last_triage.get(
                    "boarding_full_diverted", 0
                ),
            }
            if last_triage is not None
            else None
        ),
        # Discharge totals from the last tick: planned discharges,
        # how many landed, how many were delayed waiting for a
        # cleaning slot, and the total delay.
        "discharge": (
            {
                "planned": last_discharge["planned"],
                "discharged": last_discharge["discharged"],
                "delayed": last_discharge["delayed"],
                "cancelled": last_discharge["cancelled"],
                "total_delay_s": last_discharge["total_delay_s"],
                "mean_delay_s": last_discharge["mean_delay_s"],
            }
            if last_discharge is not None
            else None
        ),
        # Discharge-readiness totals from the last tick: how many
        # occupied beds are flagged ready, how many are not, and the
        # stability window in force.
        "discharge_readiness": (
            {
                "ready": last_readiness["ready"],
                "not_ready": last_readiness["not_ready"],
                "occupied": last_readiness["occupied"],
                "stable_window_s": last_readiness["stable_window_s"],
            }
            if last_readiness is not None
            else None
        ),
    }
