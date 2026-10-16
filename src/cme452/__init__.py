from .core import System, ModelError, SolverFailure
from .controller import PID
from .loop import ClosedLoop
from .plotting import plot_results, _system_plot_results
from . import stability
from . import frequency
from .stability import (
    jacobian, eigenvalues, eigenvalue_plot,
    augmented_state, one_step, closed_loop_jacobian,
)
from .frequency import response, bode, margins

# attach the convenience method to System (t is implicit)
System.plot_results = _system_plot_results

__version__ = "2.0.0"
__all__ = [
    "System", "ClosedLoop", "PID",
    "ModelError", "SolverFailure", "plot_results",
    "stability", "frequency",
    "jacobian", "eigenvalues", "eigenvalue_plot",
    "augmented_state", "one_step", "closed_loop_jacobian",
    "response", "bode", "margins",
]
