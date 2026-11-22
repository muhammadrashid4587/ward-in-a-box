"""CLI: run a ward scenario file, or an ad-hoc ward."""

import argparse
import json
import sys

from .chaos import ChaosEvent
from .server import serve
from .vitals import PROFILES
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
    for i, p in enumerate(data.get("patients", [])):
        profile = p.get("profile", "stable") if isinstance(p, dict) else p
        if profile not in PROFILES:
            raise ValueError("patient %d: unknown profile: %r" % (i, profile))
        profiles.append(profile)
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
    return {
        "beds": beds,
        "rate_hz": rate,
        "profiles": profiles,
        "chaos": chaos,
        "host": data.get("host", "127.0.0.1"),
        "port": data.get("port", 8080),
    }


def build_ward(cfg, seed=None):
    return Ward(
        beds=cfg["beds"],
        rate_hz=cfg["rate_hz"],
        profiles=cfg["profiles"],
        chaos=cfg["chaos"],
        seed=seed,
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
    args = ap.parse_args(argv)

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
            "host": args.host or "127.0.0.1",
            "port": args.port if args.port is not None else 8080,
        }

    ward = build_ward(cfg, seed=args.seed)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
