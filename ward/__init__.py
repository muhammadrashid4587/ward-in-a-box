"""ward-in-a-box: a simulated hospital ward as a load/chaos generator."""

__version__ = "0.29.0"

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
    "TriageQueue",
    "Ward",
    "alarms_for",
    "interaction_warnings",
    "iter_ticks",
    "summarize",
    "validate_med",
    "__version__",
]
