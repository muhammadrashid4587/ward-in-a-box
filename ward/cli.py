"""CLI: run a ward scenario file, or an ad-hoc ward."""

import argparse
import json
import sys

from .chaos import ChaosEvent
from .recorder import Recorder, iter_ticks, summarize
from .server import serve
from .timeline import Timeline, TimelinePhase
from .vitals import PROFILES, validate_acuity
from .ward import Ward


def load_scenario(path):
    """Load and validate a scenario JSON file."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    beds = data.get("beds", 12)
    rate = data.get("rate_hz", 1.0)
    profiles = []
    acuities = []
    for i, p in enumerate(data.get("patients", [])):
        profile = p.get("profile", "stable") if isinstance(p, dict) else p
        if profile not in PROFILES:
            raise ValueError("patient %d: unknown profile: %r" % (i, profile))
        profiles.append(profile)
        acuity = p.get("acuity", 3) if isinstance(p, dict) else 3
        try:
            validate_acuity(acuity)
        except ValueError:
            raise ValueError("patient %d: acuity must be an int in 1..4, got %r" % (i, acuity))
        acuities.append(acuity)
    chaos = []
    for i, ev in enumerate(data.get("chaos", [])):
        if not isinstance(ev, dict):
            raise ValueError("chaos event %d: must be an object" % i)
        try:
            chaos.append(
                ChaosEvent(
                    ev["type"],
                    ev["at"],
                    ev["duration"],
                    bed=ev.get("bed"),
                    beds=ev.get("beds"),
                )
            )
        except KeyError as e:
            raise ValueError("chaos event %d: missing %s" % (i, e))
    timeline = None
    if "timeline" in data:
        phases = []
        for i, ph in enumerate(data.get("timeline") or []):
            if not isinstance(ph, dict):
                raise ValueError("timeline phase %d: must be an object" % i)
            try:
                phases.append(
                    TimelinePhase(ph["at"], ph["beds"], ramp_s=ph.get("ramp_s", 0))
                )
            except KeyError as e:
                raise ValueError("timeline phase %d: missing %s" % (i, e))
        timeline = Timeline(phases)
    return {
        "beds": beds,
        "rate_hz": rate,
        "profiles": profiles,
        "acuities": acuities,
        "chaos": chaos,
        "timeline": timeline,
        "host": data.get("host", "127.0.0.1"),
        "port": data.get("port", 8080),
    }


def build_ward(cfg, seed=None, recorder=None):
    return Ward(
        beds=cfg["beds"],
        rate_hz=cfg["rate_hz"],
        profiles=cfg["profiles"],
        acuities=cfg.get("acuities"),
        chaos=cfg["chaos"],
        seed=seed,
        timeline=cfg.get("timeline"),
        recorder=recorder,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ward-in-a-box",
        description="Simulated hospital ward: a load/chaos generator for testing health software.",
    )
    ap.add_argument("scenario", nargs="?", help="scenario JSON file")
    ap.add_argument("--beds", type=int, default=12, help="beds (no scenario)")
    ap.add_argument("--rate", type=float, default=1.0, dest="rate_hz", help="ticks per second (no scenario)")
    ap.add_argument("--host", default=None, help="override scenario host")
    ap.add_argument("--port", type=int, default=None, help="override scenario port")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible runs")
    ap.add_argument("--once", action="store_true", help="print one snapshot as JSON and exit")
    ap.add_argument("--record", metavar="FILE", default=None,
                    help="record every tick to a JSONL file")
    ap.add_argument("--summary", metavar="FILE", default=None,
                    help="print a summary of a recorded run and exit")
    ap.add_argument("--replay", metavar="FILE", default=None,
                    help="print a recorded run's ticks as JSON lines and exit")
    args = ap.parse_args(argv)

    if args.summary is not None:
        print(json.dumps(summarize(args.summary)))
        return 0
    if args.replay is not None:
        for snap in iter_ticks(args.replay):
            print(json.dumps(snap))
        return 0

    if args.scenario:
        cfg = load_scenario(args.scenario)
        if args.host is not None:
            cfg["host"] = args.host
        if args.port is not None:
            cfg["port"] = args.port
    else:
        cfg = {
            "beds": args.beds,
            "rate_hz": args.rate_hz,
            "profiles": [],
            "chaos": [],
            "timeline": None,
            "host": args.host or "127.0.0.1",
            "port": args.port if args.port is not None else 8080,
        }

    recorder = None
    if args.record is not None:
        recorder = Recorder(args.record, cfg["beds"], cfg["rate_hz"], seed=args.seed)
    ward = build_ward(cfg, seed=args.seed, recorder=recorder)
    try:
        if args.once:
            print(json.dumps(ward.tick()))
            return 0

        ward.start()
        print(
            "ward-in-a-box: %d beds at %s Hz on http://%s:%d (Ctrl-C to stop)"
            % (ward.beds, ward.rate_hz, cfg["host"], cfg["port"]),
            file=sys.stderr,
        )
        serve(ward, cfg["host"], cfg["port"])
    finally:
        if recorder is not None:
            recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
