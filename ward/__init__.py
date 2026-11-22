"""ward-in-a-box: a simulated hospital ward as a load/chaos generator."""

__version__ = "0.1.0"

from .chaos import ChaosEngine, ChaosEvent
from .monitor import Monitor
from .vitals import PROFILES, alarms_for
from .ward import Ward

__all__ = [
    "ChaosEngine",
    "ChaosEvent",
    "Monitor",
    "PROFILES",
    "Ward",
    "alarms_for",
    "__version__",
]
