"""V1.4 Comprehensive Stress Test Suite."""
__version__ = "1.4.0"

__all__ = ['__version__']

# Keep package import lightweight. Heavy modules (torch/sklearn) may be absent
# in scenario-generation environments.
try:
    from .scenarios import Scenario, ScenarioResult, ScenarioRegistry
    __all__.extend(['Scenario', 'ScenarioResult', 'ScenarioRegistry'])
except Exception:
    pass

try:
    from .logger import JSONLogger
    from .dashboard import DashboardGenerator
    __all__.extend(['JSONLogger', 'DashboardGenerator'])
except Exception:
    pass

try:
    from .runner import StressTestRunner, AdaptiveScheduler
    __all__.extend(['StressTestRunner', 'AdaptiveScheduler'])
except Exception:
    pass

try:
    from .models import ModelWrapper
    __all__.append('ModelWrapper')
except Exception:
    pass
