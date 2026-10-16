from .core import System, ModelError, SolverFailure
from .controller import PID
from .loop import ClosedLoop
from .plotting import plot_results, _system_plot_results

# attach the convenience method to System (t is implicit)
System.plot_results = _system_plot_results

__version__ = "2.0.0"
__all__ = [
    "System", "ClosedLoop", "PID",
    "ModelError", "SolverFailure", "plot_results",
]
