"""ward-in-a-box: a simulated hospital ward as a load/chaos generator."""

__version__ = "0.59.0"

from .alert_hooks import (
    AlertHooks,
    TrendAlertHooks,
    validate_boarding_full_hooks,
    validate_discharge_readiness_hooks,
    validate_escalation_hooks,
    validate_missed_dose_hooks,
    validate_trend_alert_hooks,
    validate_wait_time_hooks,
)
from .arrivals import ArrivalModel
from .chaos import ChaosEngine, ChaosEvent
from .discharge import DischargePlanner
from .readiness import DischargeReadiness
from .meds import MedSchedule, interaction_warnings, validate_med
from .monitor import Monitor
from .recorder import Recorder, iter_ticks, summarize
from .staffing import Staffing
from .timeline import Timeline, TimelinePhase
from .triage import DEFAULT_CAPACITY, TriageQueue
from .vitals import PROFILES, alarms_for
from .ward import Ward

__all__ = [
    "AlertHooks",
    "ArrivalModel",
    "ChaosEngine",
    "ChaosEvent",
    "DischargePlanner",
    "DischargeReadiness",
    "MedSchedule",
    "Monitor",
    "PROFILES",
    "Recorder",
    "Staffing",
    "Timeline",
    "TimelinePhase",
    "TrendAlertHooks",
    "TriageQueue",
    "Ward",
    "alarms_for",
    "interaction_warnings",
    "iter_ticks",
    "summarize",
    "validate_boarding_full_hooks",
    "validate_discharge_readiness_hooks",
    "validate_escalation_hooks",
    "validate_med",
    "validate_missed_dose_hooks",
    "validate_trend_alert_hooks",
    "validate_wait_time_hooks",
    "__version__",
]
