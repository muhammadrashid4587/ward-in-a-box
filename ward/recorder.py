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
    the run's load shape over the day is visible. Returns ``None``
    fields for a recording with no ticks.
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
        # hourly rates, so the run's load shape is visible.
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
            }
            if last_triage is not None
            else None
        ),
    }
