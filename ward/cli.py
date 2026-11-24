"""CLI: run a ward scenario file, or an ad-hoc ward."""

import argparse
import json
import sys

from .alert_hooks import (
    validate_boarding_full_hooks,
    validate_discharge_readiness_hooks,
    validate_escalation_hooks,
    validate_missed_dose_hooks,
    validate_trend_alert_hooks,
    validate_wait_time_hooks,
)
from .chaos import ChaosEvent
from .discharge import DischargePlanner
from .escalation import validate_news2_staffing_escalation
from .news2 import (
    validate_consciousness,
    validate_consciousness_dynamics,
    validate_news2_trend,
)
from .readiness import DischargeReadiness
from .recorder import Recorder, iter_ticks, summarize
from .server import serve
from .staffing import Staffing
from .timeline import Timeline, TimelinePhase
from .triage import TriageQueue
from .meds import validate_med
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
    oxygen = []
    consciousness = []
    consciousness_dynamics = []
    meds = []
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
        on_oxygen = p.get("on_oxygen", False) if isinstance(p, dict) else False
        if not isinstance(on_oxygen, bool):
            raise ValueError(
                "patient %d: on_oxygen must be a bool, got %r" % (i, on_oxygen)
            )
        oxygen.append(on_oxygen)
        level = p.get("consciousness", "alert") if isinstance(p, dict) else "alert"
        try:
            validate_consciousness(level)
        except ValueError:
            raise ValueError(
                "patient %d: consciousness must be one of alert, confusion, voice, pain, unresponsive, got %r"
                % (i, level)
            )
        consciousness.append(level)
        dynamics = p.get("consciousness_dynamics") if isinstance(p, dict) else None
        try:
            consciousness_dynamics.append(
                validate_consciousness_dynamics(dynamics)
            )
        except ValueError as e:
            raise ValueError("patient %d: %s" % (i, e))
        bed_meds = p.get("meds", []) if isinstance(p, dict) else []
        if not isinstance(bed_meds, list):
            raise ValueError("patient %d: meds must be a list" % i)
        try:
            meds.append([validate_med(m) for m in bed_meds])
        except ValueError as e:
            raise ValueError("patient %d: %s" % (i, e))
        names = [m["name"].lower() for m in meds[-1]]
        if len(set(names)) != len(names):
            raise ValueError("patient %d: duplicate medication names" % i)
    chaos = []
    for i, ev in enumerate(data.get("chaos", [])):
        if not isinstance(ev, dict):
            raise ValueError("chaos event %d: must be an object" % i)
        try:
            chaos.append(
                ChaosEvent(
                    ev["type"],
                    ev["at"],
                    ev.get("duration"),
                    bed=ev.get("bed"),
                    beds=ev.get("beds"),
                    med=ev.get("med"),
                )
            )
        except KeyError as e:
            raise ValueError("chaos event %d: missing %s" % (i, e))
        except ValueError as e:
            raise ValueError("chaos event %d: %s" % (i, e))
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
    staffing = None
    if "staffing" in data:
        if not isinstance(data["staffing"], dict):
            raise ValueError("staffing must be an object")
        try:
            staffing = Staffing(**data["staffing"])
        except (TypeError, ValueError) as e:
            raise ValueError("staffing: %s" % e)
    cleaning_s = data.get("cleaning_s", 30.0)
    if not isinstance(cleaning_s, (int, float)) or cleaning_s <= 0:
        raise ValueError("cleaning_s must be > 0")
    cleaning_capacity = data.get("cleaning_capacity")
    if cleaning_capacity is not None and (
        not isinstance(cleaning_capacity, int)
        or isinstance(cleaning_capacity, bool)
        or cleaning_capacity < 1
    ):
        raise ValueError(
            "cleaning_capacity must be an int >= 1, got %r"
            % (cleaning_capacity,)
        )
    triage = None
    if "triage" in data:
        if not isinstance(data["triage"], dict):
            raise ValueError("triage must be an object")
        try:
            triage = TriageQueue(**data["triage"])
        except (TypeError, ValueError) as e:
            raise ValueError("triage: %s" % e)
    discharge_planning = None
    if "discharge_planning" in data:
        if not isinstance(data["discharge_planning"], dict):
            raise ValueError("discharge_planning must be an object")
        try:
            discharge_planning = DischargePlanner(**data["discharge_planning"])
        except (TypeError, ValueError) as e:
            raise ValueError("discharge_planning: %s" % e)
    discharge_readiness = None
    if "discharge_readiness" in data:
        if not isinstance(data["discharge_readiness"], dict):
            raise ValueError("discharge_readiness must be an object")
        try:
            discharge_readiness = DischargeReadiness(
                **data["discharge_readiness"]
            )
        except (TypeError, ValueError) as e:
            raise ValueError("discharge_readiness: %s" % e)
    news2_trend = None
    if "news2_trend" in data:
        if not isinstance(data["news2_trend"], dict):
            raise ValueError("news2_trend must be an object")
        try:
            news2_trend = validate_news2_trend(data["news2_trend"])
        except ValueError as e:
            raise ValueError("news2_trend: %s" % e)
    news2_staffing_escalation = None
    if "news2_staffing_escalation" in data:
        if not isinstance(data["news2_staffing_escalation"], dict):
            raise ValueError("news2_staffing_escalation must be an object")
        try:
            news2_staffing_escalation = (
                validate_news2_staffing_escalation(
                    data["news2_staffing_escalation"]
                )
            )
        except ValueError as e:
            raise ValueError("news2_staffing_escalation: %s" % e)
    trend_alert_hooks = None
    if "trend_alert_hooks" in data:
        if not isinstance(data["trend_alert_hooks"], dict):
            raise ValueError("trend_alert_hooks must be an object")
        try:
            trend_alert_hooks = validate_trend_alert_hooks(
                data["trend_alert_hooks"]
            )
        except ValueError as e:
            raise ValueError("trend_alert_hooks: %s" % e)
    escalation_hooks = None
    if "escalation_hooks" in data:
        if not isinstance(data["escalation_hooks"], dict):
            raise ValueError("escalation_hooks must be an object")
        try:
            escalation_hooks = validate_escalation_hooks(
                data["escalation_hooks"]
            )
        except ValueError as e:
            raise ValueError("escalation_hooks: %s" % e)
    boarding_full_hooks = None
    if "boarding_full_hooks" in data:
        if not isinstance(data["boarding_full_hooks"], dict):
            raise ValueError("boarding_full_hooks must be an object")
        try:
            boarding_full_hooks = validate_boarding_full_hooks(
                data["boarding_full_hooks"]
            )
        except ValueError as e:
            raise ValueError("boarding_full_hooks: %s" % e)
    wait_time_hooks = None
    if "wait_time_hooks" in data:
        if not isinstance(data["wait_time_hooks"], dict):
            raise ValueError("wait_time_hooks must be an object")
        try:
            wait_time_hooks = validate_wait_time_hooks(
                data["wait_time_hooks"]
            )
        except ValueError as e:
            raise ValueError("wait_time_hooks: %s" % e)
    missed_dose_hooks = None
    if "missed_dose_hooks" in data:
        if not isinstance(data["missed_dose_hooks"], dict):
            raise ValueError("missed_dose_hooks must be an object")
        try:
            missed_dose_hooks = validate_missed_dose_hooks(
                data["missed_dose_hooks"]
            )
        except ValueError as e:
            raise ValueError("missed_dose_hooks: %s" % e)
    discharge_readiness_hooks = None
    if "discharge_readiness_hooks" in data:
        if not isinstance(data["discharge_readiness_hooks"], dict):
            raise ValueError("discharge_readiness_hooks must be an object")
        try:
            discharge_readiness_hooks = (
                validate_discharge_readiness_hooks(
                    data["discharge_readiness_hooks"]
                )
            )
        except ValueError as e:
            raise ValueError("discharge_readiness_hooks: %s" % e)
    return {
        "beds": beds,
        "rate_hz": rate,
        "profiles": profiles,
        "acuities": acuities,
        "oxygen": oxygen,
        "consciousness": consciousness,
        "consciousness_dynamics": consciousness_dynamics,
        "meds": meds,
        "chaos": chaos,
        "timeline": timeline,
        "staffing": staffing,
        "cleaning_s": float(cleaning_s),
        "cleaning_capacity": cleaning_capacity,
        "discharge_planning": discharge_planning,
        "discharge_readiness": discharge_readiness,
        "news2_trend": news2_trend,
        "news2_staffing_escalation": news2_staffing_escalation,
        "trend_alert_hooks": trend_alert_hooks,
        "escalation_hooks": escalation_hooks,
        "boarding_full_hooks": boarding_full_hooks,
        "wait_time_hooks": wait_time_hooks,
        "missed_dose_hooks": missed_dose_hooks,
        "discharge_readiness_hooks": discharge_readiness_hooks,
        "triage": triage,
        "host": data.get("host", "127.0.0.1"),
        "port": data.get("port", 8080),
    }


def build_ward(cfg, seed=None, recorder=None):
    return Ward(
        beds=cfg["beds"],
        rate_hz=cfg["rate_hz"],
        profiles=cfg["profiles"],
        acuities=cfg.get("acuities"),
        oxygen=cfg.get("oxygen"),
        consciousness=cfg.get("consciousness"),
        consciousness_dynamics=cfg.get("consciousness_dynamics"),
        meds=cfg.get("meds"),
        chaos=cfg["chaos"],
        seed=seed,
        timeline=cfg.get("timeline"),
        staffing=cfg.get("staffing"),
        cleaning_s=cfg.get("cleaning_s", 30.0),
        cleaning_capacity=cfg.get("cleaning_capacity"),
        discharge_planning=cfg.get("discharge_planning"),
        discharge_readiness=cfg.get("discharge_readiness"),
        news2_trend=cfg.get("news2_trend"),
        news2_staffing_escalation=cfg.get("news2_staffing_escalation"),
        trend_alert_hooks=cfg.get("trend_alert_hooks"),
        escalation_hooks=cfg.get("escalation_hooks"),
        boarding_full_hooks=cfg.get("boarding_full_hooks"),
        wait_time_hooks=cfg.get("wait_time_hooks"),
        missed_dose_hooks=cfg.get("missed_dose_hooks"),
        discharge_readiness_hooks=cfg.get("discharge_readiness_hooks"),
        triage=cfg.get("triage"),
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
            "meds": [],
            "chaos": [],
            "timeline": None,
            "staffing": None,
            "cleaning_capacity": None,
            "discharge_planning": None,
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
