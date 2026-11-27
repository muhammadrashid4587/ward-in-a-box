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
    how many were delayed by a backed-up cleaning roster, how many
    auto orders were cancelled when a bed stopped being ready,
    and the total delay — and discharge-readiness totals from the last tick:
    how many occupied beds are flagged ready and how many are not,
    how many times the readiness flag flipped this run, and whether
    the discharge-readiness hooks are configured and how many hook
    deliveries were invoked and failed — and the discharge-priority
    order from the last tick: the ready beds ranked by their NEWS2
    at the flip that made them ready, with when each flipped and
    how long it has been ready — and early-warning totals from the last tick: how many beds are
    at high, medium, and low NEWS2 risk, how many deterioration
    episodes are open, and how many have raised, and how many
    NEWS2 trend alerts are open and have raised — and whether the
    trend-alert hooks are configured and how many hook deliveries
    were invoked and failed — and how the ward
    responded to its escalations: how many got a response, how many
    closed before anyone responded, how many are still waiting, and
    the mean and longest time from an escalation raising to the
    response landing — and NEWS2 history
    totals per bed: ticks scored, peak score and risk, and ticks at
    each risk level — and sensor-drift totals from the last tick:
    how many beds have a drifted sensor, each drifting bed's
    accumulated bias per vital, and the largest absolute bias per
    vital — plus how many sensor-drift hook events fired this run,
    and whether the sensor-drift hooks are configured and how many
    hook deliveries were invoked — and bed-blocker totals from the last tick: how many
    beds are blocked and why, how many discharges are still held
    by a block, and how many discharges the run has held this
    way — plus how many bed-blocker hook events fired this run,
    and whether the bed-blocker hooks are configured and how many
    hook deliveries were invoked — and oxygen-outage totals from the last tick: how many
    patients are on oxygen, how many are desaturating from a
    failed supply, each desaturating bed's SpO2 loss, and the
    largest loss on any bed — and power-outage totals from the
    last tick: how many monitors are on battery, how many have
    gone dark, how many are rebooting, each dark bed's dark
    ticks, and the ward's total dark ticks — and network-outage
    totals from the last tick: how many beds have their link
    down, how many readings are buffered locally waiting to
    sync, how many were buffered, synced, and dropped, and how
    many alarms raised while down are still undelivered and how
    many have since synced — and medication-stockout totals from
    the last tick: how many beds have a medication stocked out,
    how many are drifting off their drug, which medications are
    out, the largest drift per vital, and each drifting bed's
    drift — and medication totals from the last tick:
    doses given, late, missed, and skipped for a stockout, and
    whether the missed-dose hooks are configured and how many hook
    deliveries were invoked and failed — and staffing totals from
    the last tick: the load factor, how many extra nurses a NEWS2
    staffing escalation has called in, and the escalation state.
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
    news2_hist = {}
    last_triage = None
    last_discharge = None
    last_readiness = None
    last_priority = None
    last_early_warning = None
    last_meds = None
    last_staffing = None
    last_sensor_drift = None
    last_bed_blocker = None
    last_oxygen_outage = None
    last_power_outage = None
    last_network_outage = None
    last_medication_stockout = None
    last_pump_failure = None
    last_lab_delay = None
    last_ventilator_failure = None
    last_dialysis_failure = None
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
            if isinstance(obj.get("meds"), dict):
                last_meds = obj["meds"]
            if isinstance(obj.get("discharge"), dict):
                last_discharge = obj["discharge"]
            if isinstance(obj.get("discharge_readiness"), dict):
                last_readiness = obj["discharge_readiness"]
            if isinstance(obj.get("discharge_priority"), dict):
                last_priority = obj["discharge_priority"]
            if isinstance(obj.get("early_warning"), dict):
                last_early_warning = obj["early_warning"]
            if isinstance(obj.get("staffing"), dict):
                last_staffing = obj["staffing"]
            if isinstance(obj.get("sensor_drift"), dict):
                last_sensor_drift = obj["sensor_drift"]
            if isinstance(obj.get("bed_blocker"), dict):
                last_bed_blocker = obj["bed_blocker"]
                # NEWS2 history totals per bed, from the score on
                # each tick's reading: ticks scored, peak score and
                # the risk at that score, and ticks at each risk
                # level, with the first and last scored times — so
                # a run's escalation shape is visible without
                # replaying every tick.
                for b in obj["early_warning"].get("beds", []):
                    n2 = b.get("news2")
                    if n2 is None:
                        continue
                    entry = news2_hist.setdefault(
                        b["bed"],
                        {
                            "ticks": 0,
                            "peak_score": 0,
                            "peak_risk": "low",
                            "ticks_low": 0,
                            "ticks_medium": 0,
                            "ticks_high": 0,
                            "first_t": t,
                            "last_t": t,
                        },
                    )
                    entry["ticks"] += 1
                    entry["last_t"] = t
                    if n2["score"] > entry["peak_score"]:
                        entry["peak_score"] = n2["score"]
                        entry["peak_risk"] = n2["risk"]
                    entry["ticks_" + n2["risk"]] += 1
            if isinstance(obj.get("oxygen_outage"), dict):
                last_oxygen_outage = obj["oxygen_outage"]
            if isinstance(obj.get("power_outage"), dict):
                last_power_outage = obj["power_outage"]
            if isinstance(obj.get("network_outage"), dict):
                last_network_outage = obj["network_outage"]
            if isinstance(obj.get("medication_stockout"), dict):
                last_medication_stockout = obj["medication_stockout"]
            if isinstance(obj.get("pump_failure"), dict):
                last_pump_failure = obj["pump_failure"]
            if isinstance(obj.get("lab_delay"), dict):
                last_lab_delay = obj["lab_delay"]
            if isinstance(obj.get("ventilator_failure"), dict):
                last_ventilator_failure = obj["ventilator_failure"]
            if isinstance(obj.get("dialysis_failure"), dict):
                last_dialysis_failure = obj["dialysis_failure"]
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
                "arrival_ramp": last_triage.get("arrival_ramp"),
                "boarding_full_alarms_raised": last_triage.get(
                    "boarding_full_alarms_raised", 0
                ),
                "boarding_full_alarm_active": last_triage.get(
                    "boarding_full_alarm_active", False
                ),
                "boarding_full_diverted": last_triage.get(
                    "boarding_full_diverted", 0
                ),
                "boarding_full_hooks": last_triage.get(
                    "boarding_full_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
                "wait_time_hooks": last_triage.get(
                    "wait_time_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
            }
            if last_triage is not None
            else None
        ),
        # Discharge totals from the last tick: planned discharges,
        # how many landed, how many were delayed waiting for a
        # cleaning slot, and the total delay — and how many of the
        # planned orders the ward raised automatically when a bed
        # was flagged discharge-ready, and how many of those auto
        # orders were cancelled when a bed stopped being ready —
        # and the paperwork timing: how long, on average, from a
        # bed being flagged ready to the order and to the patient
        # leaving.
        "discharge": (
            {
                "planned": last_discharge["planned"],
                "auto_planned": last_discharge.get("auto_planned", 0),
                "discharged": last_discharge["discharged"],
                "delayed": last_discharge["delayed"],
                "cancelled": last_discharge["cancelled"],
                "auto_cancelled": last_discharge.get("auto_cancelled", 0),
                "total_delay_s": last_discharge["total_delay_s"],
                "mean_delay_s": last_discharge["mean_delay_s"],
                "cleaning_queue": last_discharge.get(
                    "cleaning_queue", []
                ),
                "timing": last_discharge.get(
                    "timing",
                    {
                        "measured": 0,
                        "mean_ready_to_order_s": 0.0,
                        "mean_ready_to_discharge_s": 0.0,
                        "max_ready_to_discharge_s": 0.0,
                    },
                ),
            }
            if last_discharge is not None
            else None
        ),
        # Medication totals from the last tick: doses given, late,
        # missed, and skipped for a stockout — and whether the
        # missed-dose hooks are configured and how many hook
        # deliveries were invoked, failed, retried, and delivered.
        "meds": (
            {
                "total_given": last_meds["total_given"],
                "total_late": last_meds["total_late"],
                "total_missed": last_meds["total_missed"],
                "total_stockout": last_meds["total_stockout"],
                "missed_dose_hooks": last_meds.get(
                    "missed_dose_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
            }
            if last_meds is not None
            else None
        ),
        # Discharge-readiness totals from the last tick: how many
        # occupied beds are flagged ready, how many are not, and the
        # stability window in force — plus the auto-order lead-time
        # override, when one is set — and how many times the
        # readiness flag flipped this run, to ready and back, and
        # whether the discharge-readiness hooks are configured and
        # how many hook deliveries were invoked, failed, retried,
        # and delivered.
        "discharge_readiness": (
            {
                "ready": last_readiness["ready"],
                "not_ready": last_readiness["not_ready"],
                "occupied": last_readiness["occupied"],
                "stable_window_s": last_readiness["stable_window_s"],
                "auto_order_lead_time_s": last_readiness.get(
                    "auto_order_lead_time_s"
                ),
                "readiness_flips": last_readiness.get(
                    "readiness_flips", 0
                ),
                "flips_to_ready": last_readiness.get(
                    "flips_to_ready", 0
                ),
                "flips_to_not_ready": last_readiness.get(
                    "flips_to_not_ready", 0
                ),
                "discharge_readiness_hooks": last_readiness.get(
                    "discharge_readiness_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
            }
            if last_readiness is not None
            else None
        ),
        # Discharge-priority totals from the last tick: how many
        # beds are flagged ready, in NEWS2 order — bed, NEWS2 at
        # the flip that made it ready, when it flipped, and how
        # long it has been ready — so the run's discharge order is
        # visible without replaying every tick — plus how many
        # times the order changed this run, and the per-bed rank
        # history — how each bed's rank moved through the queue,
        # with the NEWS2 at the flip and the tick of each change
        # — and whether the discharge-priority hooks are
        # configured and how many hook deliveries were invoked,
        # failed, retried, and delivered.
        "discharge_priority": (
            {
                "ready": last_priority["ready"],
                "beds": last_priority["beds"],
                "priority_changes": last_priority.get(
                    "priority_changes", 0
                ),
                "rank_changes": last_priority.get("rank_changes", 0),
                "rank_history": last_priority.get("rank_history", {}),
                "discharge_priority_hooks": last_priority.get(
                    "discharge_priority_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
            }
            if last_priority is not None
            else None
        ),
        # Early-warning totals from the last tick: beds at each
        # NEWS2 risk level, open deterioration episodes, and how
        # many episodes have raised over the run — and how many
        # patients are not Alert on the ACVPU scale — and the
        # response timing: how many escalations got a response,
        # how many closed before anyone responded, how many are
        # still waiting, and the mean and longest time from an
        # escalation raising to the response landing.
        "early_warning": (
            {
                "high_risk": last_early_warning["high_risk"],
                "medium_risk": last_early_warning["medium_risk"],
                "low_risk": last_early_warning["low_risk"],
                "escalated": last_early_warning["escalated"],
                "escalations_raised": last_early_warning[
                    "escalations_raised"
                ],
                "responses": last_early_warning.get(
                    "responses",
                    {
                        "responded": 0,
                        "unresponded": 0,
                        "pending": 0,
                        "mean_time_to_respond_s": None,
                        "max_time_to_respond_s": None,
                    },
                ),
                "trend_alerts": last_early_warning.get(
                    "trend_alerts", 0
                ),
                "trend_alerts_raised": last_early_warning.get(
                    "trend_alerts_raised", 0
                ),
                "trend_alert_hooks": last_early_warning.get(
                    "trend_alert_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
                "escalation_hooks": last_early_warning.get(
                    "escalation_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
                "not_alert": last_early_warning.get("not_alert", 0),
                "consciousness_changes": last_early_warning.get(
                    "consciousness_changes", 0
                ),
            }
            if last_early_warning is not None
            else None
        ),
        # NEWS2 history totals per bed: ticks scored, peak score
        # and the risk at that score, ticks at each risk level, and
        # the first and last scored times.
        "news2_history": news2_hist,
        # Sensor-drift totals from the last tick: how many beds
        # have a drifted sensor, each bed's accumulated bias per
        # vital, and the largest absolute bias per vital — so a
        # run's slow miscalibration is visible without replaying
        # every tick.
        # Sensor-drift totals from the last tick: how many beds
        # have a drifted sensor, each drifting bed's accumulated
        # bias per vital, and the largest absolute bias per
        # vital — so a run's sensor miscalibration is visible
        # without replaying every tick — plus how many
        # sensor-drift hook events fired this run, and whether
        # the sensor-drift hooks are configured and how many
        # hook deliveries were invoked, failed, retried, and
        # delivered.
        "sensor_drift": (
            {
                "beds_drifting": last_sensor_drift["beds_drifting"],
                "beds": {
                    str(b["bed"]): b["drift"]
                    for b in last_sensor_drift["beds"]
                    if b["drifting"]
                },
                "max_bias": last_sensor_drift.get("max_bias", {}),
                "event_count": last_sensor_drift.get(
                    "event_count", 0
                ),
                "sensor_drift_hooks": last_sensor_drift.get(
                    "sensor_drift_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
            }
            if last_sensor_drift is not None
            else None
        ),
        # Bed-blocker totals from the last tick: how many beds
        # are blocked and why, how many discharges are still
        # held by a block, and how many discharges the run has
        # held this way — so a run's beds out of use are visible
        # without replaying every tick — plus how many
        # bed-blocker hook events fired this run, and whether
        # the bed-blocker hooks are configured and how many
        # hook deliveries were invoked, failed, retried, and
        # delivered.
        "bed_blocker": (
            {
                "beds_blocked": last_bed_blocker["beds_blocked"],
                "beds": {
                    str(b["bed"]): b["reason"]
                    for b in last_bed_blocker["beds"]
                    if b["blocked"]
                },
                "discharges_pending": last_bed_blocker.get(
                    "discharges_pending", 0
                ),
                "discharges_held": last_bed_blocker.get(
                    "discharges_held", 0
                ),
                "event_count": last_bed_blocker.get(
                    "event_count", 0
                ),
                "bed_blocker_hooks": last_bed_blocker.get(
                    "bed_blocker_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
            }
            if last_bed_blocker is not None
            else None
        ),
        # Oxygen-outage totals from the last tick: how many
        # patients are on oxygen, how many are desaturating from
        # a failed supply, each desaturating bed's SpO2 loss, and
        # the largest loss on any bed — so a run's oxygen failure
        # is visible without replaying every tick — plus how many
        # oxygen-outage hook events fired this run, and whether
        # the oxygen-outage hooks are configured and how many
        # hook deliveries were invoked, failed, retried, and
        # delivered.
        "oxygen_outage": (
            {
                "beds_on_oxygen": last_oxygen_outage[
                    "beds_on_oxygen"
                ],
                "beds_desaturating": last_oxygen_outage[
                    "beds_desaturating"
                ],
                "beds": {
                    str(b["bed"]): b["deficit"]
                    for b in last_oxygen_outage["beds"]
                    if b["desaturating"]
                },
                "max_deficit": last_oxygen_outage.get(
                    "max_deficit", 0.0
                ),
                "event_count": last_oxygen_outage.get(
                    "event_count", 0
                ),
                "oxygen_outage_hooks": last_oxygen_outage.get(
                    "oxygen_outage_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
            }
            if last_oxygen_outage is not None
            else None
        ),
        # Power-outage totals from the last tick: how many
        # monitors are on battery, how many have gone dark, how
        # many are rebooting, each dark bed's dark ticks, and
        # the ward's total dark ticks — so a run's mains failure
        # is visible without replaying every tick — plus how many
        # power-outage hook events fired this run, and whether
        # the power-outage hooks are configured and how many
        # hook deliveries were invoked, failed, retried, and
        # delivered.
        "power_outage": (
            {
                "beds_on_battery": last_power_outage[
                    "beds_on_battery"
                ],
                "beds_dark": last_power_outage["beds_dark"],
                "beds_rebooting": last_power_outage[
                    "beds_rebooting"
                ],
                "beds": {
                    str(b["bed"]): b["dark_ticks"]
                    for b in last_power_outage["beds"]
                    if b["dark_ticks"] > 0
                },
                "dark_ticks": last_power_outage.get(
                    "dark_ticks", 0
                ),
                "event_count": last_power_outage.get(
                    "event_count", 0
                ),
                "power_outage_hooks": last_power_outage.get(
                    "power_outage_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
            }
            if last_power_outage is not None
            else None
        ),
        # Staffing totals from the last tick: the load factor, how
        # many extra nurses a NEWS2 staffing escalation has called
        # in, the staff an active staff-shortage event has taken
        # off the roster, and the escalation state — whether one
        # is configured and active, how many high-risk episodes
        # are open, and every escalation raised so far.
        "staffing": (
            {
                "load_factor": last_staffing["load_factor"],
                "escalation_nurses": last_staffing.get(
                    "escalation_nurses", 0
                ),
                "shortage": last_staffing.get("shortage", {}),
                "escalation": last_staffing.get(
                    "escalation",
                    {
                        "configured": False,
                        "active": False,
                        "threshold_high": None,
                        "extra_nurses": 0,
                        "open_high_episodes": 0,
                        "escalations_raised": 0,
                        "escalations": [],
                    },
                ),
            }
            if last_staffing is not None
            else None
        ),
        # Network-outage totals from the last tick: how many
        # beds have their link down, how many readings are
        # buffered locally waiting to sync, how many were
        # buffered, synced, and dropped over the run, and how
        # many alarms raised while down are still undelivered
        # and how many have since synced — so a run's cut-off
        # link is visible without replaying every tick — plus
        # how many network-outage hook events fired this run,
        # and whether the network-outage hooks are configured
        # and how many hook deliveries were invoked, failed,
        # retried, and delivered.
        "network_outage": (
            {
                "beds_down": last_network_outage["beds_down"],
                "backlog": last_network_outage["backlog"],
                "buffered": last_network_outage["buffered"],
                "synced": last_network_outage["synced"],
                "dropped": last_network_outage["dropped"],
                "alarms_undelivered": last_network_outage[
                    "alarms_undelivered"
                ],
                "alarms_synced": last_network_outage["alarms_synced"],
                "beds": {
                    str(b["bed"]): b["backlog"]
                    for b in last_network_outage["beds"]
                    if b["backlog"] > 0
                },
                "event_count": last_network_outage.get(
                    "event_count", 0
                ),
                "network_outage_hooks": last_network_outage.get(
                    "network_outage_hooks",
                    {
                        "configured": False,
                        "invoked": 0,
                        "failed": 0,
                        "retries": 0,
                        "delivered": 0,
                    },
                ),
            }
            if last_network_outage is not None
            else None
        ),
        # Medication-stockout totals from the last tick: how many
        # beds have a medication stocked out, how many are
        # drifting off their drug, which medications are out, the
        # largest drift per vital, and each drifting bed's drift
        # — so a run's missed doses and their clinical cost are
        # visible without replaying every tick — plus how many
        # medication-stockout hook events fired this run, and
        # whether the medication-stockout hooks are configured
        # and how many hook deliveries were invoked, failed,
        # retried, and delivered.
        "medication_stockout": (
            {
                "beds_affected": last_medication_stockout[
                    "beds_affected"
                ],
                "beds_drifting": last_medication_stockout[
                    "beds_drifting"
                ],
                "meds": last_medication_stockout["meds"],
                "max_drift": last_medication_stockout["max_drift"],
                "beds": {
                    str(b["bed"]): b["drift"]
                    for b in last_medication_stockout["beds"]
                    if b["drifting"]
                },
                "event_count": last_medication_stockout.get(
                    "event_count", 0
                ),
                "medication_stockout_hooks": (
                    last_medication_stockout.get(
                        "medication_stockout_hooks",
                        {
                            "configured": False,
                            "invoked": 0,
                            "failed": 0,
                            "retries": 0,
                            "delivered": 0,
                        },
                    )
                ),
            }
            if last_medication_stockout is not None
            else None
        ),
        # Pump-failure totals from the last tick: how many beds
        # have their pump down, how many are drifting off their
        # infusion, which infusions are stopped, the largest
        # drift per vital, and each drifting bed's drift — so a
        # run's failed pumps and their clinical cost are visible
        # without replaying every tick — plus how many
        # pump-failure hook events fired this run, and whether
        # the pump-failure hooks are configured and how many
        # hook deliveries were invoked, failed, retried, and
        # delivered.
        "pump_failure": (
            {
                "beds_affected": last_pump_failure["beds_affected"],
                "beds_drifting": last_pump_failure["beds_drifting"],
                "infusions": last_pump_failure["infusions"],
                "max_drift": last_pump_failure["max_drift"],
                "beds": {
                    str(b["bed"]): b["drift"]
                    for b in last_pump_failure["beds"]
                    if b["drifting"]
                },
                "event_count": last_pump_failure.get(
                    "event_count", 0
                ),
                "pump_failure_hooks": (
                    last_pump_failure.get(
                        "pump_failure_hooks",
                        {
                            "configured": False,
                            "invoked": 0,
                            "failed": 0,
                            "retries": 0,
                            "delivered": 0,
                        },
                    )
                ),
            }
            if last_pump_failure is not None
            else None
        ),
        # Lab-delay totals from the last tick: how many beds are
        # still waiting on a result, how many have one past due,
        # how many results are pending, overdue, and landed, how
        # many a delay event pushed out and by how much, and how
        # many overdue alarms raised — plus each bed still
        # waiting, with the results it is waiting on — so a run's
        # late results and the discharges they held up are visible
        # without replaying every tick — plus how many lab-delay
        # hook events fired this run, and whether the lab-delay
        # hooks are configured and how many hook deliveries were
        # invoked, failed, retried, and delivered — and the same
        # for the lab-overdue hook events, which fire when a
        # result goes overdue and when it lands after being
        # overdue.
        "lab_delay": (
            {
                "beds_pending": last_lab_delay["beds_pending"],
                "beds_overdue": last_lab_delay["beds_overdue"],
                "labs_pending": last_lab_delay["labs_pending"],
                "labs_overdue": last_lab_delay["labs_overdue"],
                "labs_landed": last_lab_delay["labs_landed"],
                "labs_delayed": last_lab_delay["labs_delayed"],
                "total_delayed_s": last_lab_delay["total_delayed_s"],
                "overdue_raised": last_lab_delay["overdue_raised"],
                "beds": {
                    str(b["bed"]): {
                        "pending": b["pending"],
                        "overdue": b["overdue"],
                    }
                    for b in last_lab_delay["beds"]
                    if b["pending"] or b["overdue"]
                },
                "event_count": last_lab_delay.get(
                    "event_count", 0
                ),
                "lab_delay_hooks": (
                    last_lab_delay.get(
                        "lab_delay_hooks",
                        {
                            "configured": False,
                            "invoked": 0,
                            "failed": 0,
                            "retries": 0,
                            "delivered": 0,
                        },
                    )
                ),
                "lab_overdue_event_count": last_lab_delay.get(
                    "lab_overdue_event_count", 0
                ),
                "lab_overdue_hooks": (
                    last_lab_delay.get(
                        "lab_overdue_hooks",
                        {
                            "configured": False,
                            "invoked": 0,
                            "failed": 0,
                            "retries": 0,
                            "delivered": 0,
                        },
                    )
                ),
            }
            if last_lab_delay is not None
            else None
        ),
        # Ventilator-failure totals from the last tick: how many
        # beds have a patient on a ventilator, how many have
        # their ventilator down, how many are drifting off
        # support, the largest drift per vital, and each
        # drifting bed's drift — so a run's failed ventilators
        # and their clinical cost are visible without replaying
        # every tick — plus how many ventilator-failure hook
        # events fired this run, and whether the
        # ventilator-failure hooks are configured and how many
        # hook deliveries were invoked, failed, retried, and
        # delivered.
        "ventilator_failure": (
            {
                "beds_on_ventilator": (
                    last_ventilator_failure["beds_on_ventilator"]
                ),
                "beds_affected": (
                    last_ventilator_failure["beds_affected"]
                ),
                "beds_drifting": (
                    last_ventilator_failure["beds_drifting"]
                ),
                "max_drift": last_ventilator_failure["max_drift"],
                "beds": {
                    str(b["bed"]): b["drift"]
                    for b in last_ventilator_failure["beds"]
                    if b["drifting"]
                },
                "event_count": last_ventilator_failure.get(
                    "event_count", 0
                ),
                "ventilator_failure_hooks": (
                    last_ventilator_failure.get(
                        "ventilator_failure_hooks",
                        {
                            "configured": False,
                            "invoked": 0,
                            "failed": 0,
                            "retries": 0,
                            "delivered": 0,
                        },
                    )
                ),
            }
            if last_ventilator_failure is not None
            else None
        ),
        # Dialysis-failure totals from the last tick: how many
        # beds have a patient on dialysis, how many have
        # their machine down, how many are drifting off
        # treatment, the largest drift per vital, and each
        # drifting bed's drift — so a run's failed machines
        # and their clinical cost are visible without replaying
        # every tick.
        "dialysis_failure": (
            {
                "beds_on_dialysis": (
                    last_dialysis_failure["beds_on_dialysis"]
                ),
                "beds_affected": (
                    last_dialysis_failure["beds_affected"]
                ),
                "beds_drifting": (
                    last_dialysis_failure["beds_drifting"]
                ),
                "max_drift": last_dialysis_failure["max_drift"],
                "beds": {
                    str(b["bed"]): b["drift"]
                    for b in last_dialysis_failure["beds"]
                    if b["drifting"]
                },
            }
            if last_dialysis_failure is not None
            else None
        ),
    }
