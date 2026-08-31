from .core import System, ModelError, SolverFailure
from .plotting import plot_results, _system_plot_results

# attach the convenience method to System (t is implicit)
System.plot_results = _system_plot_results

__version__ = "0.1.0"
__all__ = ["System", "ModelError", "SolverFailure", "plot_results"]
