"""ward-in-a-box: a simulated hospital ward as a load/chaos generator."""

__version__ = "0.6.0"

from .chaos import ChaosEngine, ChaosEvent
from .monitor import Monitor
from .recorder import Recorder, iter_ticks, summarize
from .timeline import Timeline, TimelinePhase
from .vitals import PROFILES, alarms_for
from .ward import Ward

__all__ = [
    "ChaosEngine",
    "ChaosEvent",
    "Monitor",
    "PROFILES",
    "Recorder",
    "Timeline",
    "TimelinePhase",
    "Ward",
    "alarms_for",
    "iter_ticks",
    "summarize",
    "__version__",
]
