import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Public names for method=, mapped to what SciPy calls them.
_SCIPY_METHODS = {
    "lsoda": "LSODA",   # auto-switching Adams/BDF -- the default alternative
    "bdf":   "BDF",     # implicit multistep, stiff
    "radau": "Radau",   # implicit Runge-Kutta, stiff
    "rk45":  "RK45",    # explicit adaptive -- for showing what adaptivity does
}
_METHODS = ("rk4",) + tuple(_SCIPY_METHODS)

_DT_DEFAULT = 0.01      # sentinel: lets us tell "user passed dt" from "didn't"

class ModelError(Exception):
    """Raised when the student's model function is wrong in a diagnosable way.

    A distinct type so that _probe() can re-raise our own clear messages
    untouched, instead of burying them inside a generic wrapper. (This was a
    real bug in the first draft: the good 'sqrt of a negative' message was
    being swallowed by the catch-all.)
    """

class SolverFailure(RuntimeError):
    """Raised when the SciPy backend gives up before reaching t_end."""

def _is_listlike(obj):
    return isinstance(obj, (list, tuple, np.ndarray))

def _probe_failed(x0, e):
    return (
        "The harness evaluates your model once at t = 0 before integrating, "
        "so that it can check sizes and catch errors early. That call failed.\n"
        f"  initial condition : {x0.tolist()}\n"
        f"  error             : {type(e).__name__}: {e}"
    )

def _decimate_index(n_time, n_out):
    """Indices of about n_out points spread across n_time computed points.

    Returns real computed indices, so no interpolation and no added error.
    Spacing is only approximately uniform when n_out - 1 does not divide
    n_time - 1. That is fine for plotting and honest about what was
    actually computed.
    """
    if n_out is None or n_out >= n_time:
        return np.arange(n_time)
    if n_out < 2:
        raise ModelError(f"n_out must be at least 2; you gave {n_out}.")
    idx = np.round(np.linspace(0, n_time - 1, n_out)).astype(int)
    return np.unique(idx)

class System:
    """A dynamic model plus its parameters and input schedule.

        sys = System(model, params, inputs=[T0, Q])
        sys.run_simulation(t_end=80, IC=[60.0])                  # rk4
        sys.run_simulation(t_end=80, IC=[60.0], method='lsoda')  # scipy

    model  : model(t, x, params, inputs) -> [rates] or [[rates], [values]]
    params : anything -- a dict is recommended, a list works
    inputs : list of functions of time. A single bare function is accepted.
    """
    def __init__(self, model, params, inputs=None):
        # --- model check -------------------------
        if not callable(model):
            raise ModelError(
                "model must be a function, e.g.\n"
                "    def model(t, x, params, inputs):\n"
                "        ...\n"
                "        return [dTdt]\n"
                f"You passed a {type(model).__name__}."
            )
        self.model = model
        self.params = params

        # --- inputs------------------------------
        if inputs is None:
            inputs = []
        if callable(inputs):
            inputs = [inputs]            # students will pass a bare function
        for i, f in enumerate(inputs):
            if not callable(f):
                raise ModelError(
                    f"inputs[{i}] is a {type(f).__name__}, not a function.\n"
                    "Every input must be a function of time:\n"
                    "    def Q(t): return 14400.0"
                )
        self.inputs = list(inputs)

        # --- shapes are unknown until an initial condition is supplied ---
        self.n_states = None
        self.n_vals = None

        # Results, populated by run_simulation.
        self.t = self.x = self.v = None
        self.t_full = self.x_full = None
        self.dt = None
        self.IC = None
        self.method = None
        self._dense = None

    # ---- spelled-out aliases -----------------------------------------
    @property
    def states(self):
        return self.x

    @property
    def values(self):
        return self.v

    # ==================================================================
    #  MODEL EVALUATION
    # ==================================================================
    def _call_model(self, t, x):
        """Call the provided model; return (rates, vals) as 1-D float arrays."""
        out = self.model(t, x, self.params, self.inputs)

        if not _is_listlike(out):
            raise ModelError(
                f"Your model returned a {type(out).__name__}; it must return a list.\n"
                "  physics only           ->  return [dT1dt, dT2dt]\n"
                "  physics + extra values ->  return [[dT1dt, dT2dt], [Q_loss, T_degF]]"
            )

        # Two-list form only if BOTH elements are list-like. A numpy float is
        # not, so a 2-state model returning two numpy scalars reads correctly.
        if len(out) == 2 and _is_listlike(out[0]) and _is_listlike(out[1]):
            rates, vals = out[0], out[1]
        else:
            rates, vals = out, []

        rates = np.atleast_1d(np.asarray(rates, dtype=float)).ravel()
        vals = (np.atleast_1d(np.asarray(vals, dtype=float)).ravel()
                if len(vals) else np.zeros(0))

        if not np.all(np.isfinite(rates)):
            bad = np.where(~np.isfinite(rates))[0].tolist()
            raise ModelError(
                f"Your model returned a non-finite derivative at t = {t:g}.\n"
                f"  state     = {np.asarray(x).tolist()}\n"
                f"  dx/dt     = {rates.tolist()}\n"
                f"  bad entry = {bad}\n"
                "Usual causes: divide by zero, sqrt of a negative number, "
                "log of zero, or an overflow."
            )
        return rates, vals    
    
    def _rates(self, t, x):
        return self._call_model(t, x)[0]

    # ==================================================================
    #  PROBE  (piece 1)
    # ==================================================================
    def _probe(self, x0):
        """Evaluate the model once at t = 0 to learn its shape and check sizes.

        Called at the top of run_simulation(). Sets n_states and n_vals.
        """
        x0 = np.atleast_1d(np.asarray(x0, dtype=float)).ravel()

        try:
            rates, vals = self._call_model(0.0, x0)
        except ModelError:
            raise                        # our own message is already clear
        except ValueError as e:
            if "unpack" in str(e):       # the single most common student error
                raise ModelError(
                    "Your model could not unpack the state vector.\n"
                    f"  You supplied {x0.size} initial condition"
                    f"{'' if x0.size == 1 else 's'}: {x0.tolist()}\n"
                    f"  Python said: {e}\n"
                    "The line 'T1, T2, T3 = x' needs exactly one initial "
                    "condition per state. Count the states in your model, "
                    "then count your ICs."
                ) from e
            raise ModelError(_probe_failed(x0, e)) from e
        except Exception as e:
            raise ModelError(_probe_failed(x0, e)) from e

        if rates.size != x0.size:
            raise ModelError(
                "Size mismatch between your model and your initial condition.\n"
                f"  initial conditions supplied : {x0.size}   {x0.tolist()}\n"
                f"  derivatives returned        : {rates.size}   {rates.tolist()}\n"
                "Every state needs exactly one initial condition and one dx/dt."
            )

        self.n_states = int(rates.size)
        self.n_vals = int(vals.size)
        return rates, vals
        
    # ==================================================================
    #  INTEGRATE
    # ==================================================================    
    def run_simulation(self, t_end, IC, dt=0.01, n_out=None, t_start=0.0):
        """Integrate with fixed-step RK4 from t_start to t_end.

        t_end : float
        IC    : list of initial conditions, one per state
        dt    : integration step. Snapped to a whole number of steps.
        n_out : None (default) keeps every step the integrator took.
                An integer DECIMATES to about that many points -- it keeps
                computed points, it never interpolates. Use it for plots.

        Sets sys.t, sys.x, sys.v (output grid) and sys.t_full, sys.x_full
        (every computed step). Arrays are (n_states, n_time), so sys.x[0]
        is the trajectory of the first state.
        """

        IC = np.atleast_1d(np.asarray(IC, dtype=float)).ravel()
        self._probe(IC)                  # sets n_states / n_vals, checks sizes
        self.IC = IC.copy()

        span = float(t_end) - float(t_start)
        if span <= 0:
            raise ModelError(
                f"t_end ({t_end}) must be greater than t_start ({t_start})."
            )
        if dt <= 0:
            raise ModelError(f"dt must be positive; you gave dt = {dt}.")
        if dt > span:
            raise ModelError(
                f"dt = {dt} is larger than the whole run ({span}).\n"
                "That gives a single step. Pick dt small compared with the "
                "fastest time constant in your model."
            )

        # --- snap dt to a whole number of steps --------------------------
        n_steps = max(int(round(span / dt)), 1)
        dt_used = span / n_steps
        if abs(dt_used - dt) > 1e-9 * dt:
            warnings.warn(
                f"dt adjusted from {dt:g} to {dt_used:g} so that the run "
                f"lands exactly on t_end = {t_end:g} in {n_steps} steps.",
                stacklevel=2,
            )
        self.dt = dt_used

        # --- RK4 loop ------------------------------------------------
        t = t_start + dt_used * np.arange(n_steps + 1)
        x = np.empty((self.n_states, n_steps + 1))
        x[:, 0] = IC

        h = dt_used
        f = self._rates
        for i in range(n_steps):
            ti = t[i]
            xi = x[:, i]
            k1 = f(ti, xi)
            k2 = f(ti + 0.5 * h, xi + 0.5 * h * k1)
            k3 = f(ti + 0.5 * h, xi + 0.5 * h * k2)
            k4 = f(ti + h, xi + h * k3)
            x[:, i + 1] = xi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.t_full = t
        self.x_full = x

        # --- output grid: decimate, never interpolate --------------------
        idx = _decimate_index(t.size, n_out)
        self.t = t[idx]
        self.x = x[:, idx]
        self.v = self._values_at(self.t, self.x)
        return self.t, self.x, self.v
    
    # ------------------------------------------------------------------
    def _values_at(self, t_arr, x_arr):
        """Recompute the reported values at the given (t, x) points.

        Values are algebraic functions of (t, x, params, inputs), so
        evaluating them after the fact is EXACT -- and it means the values
        mechanism does not care how the states were produced. That is the
        hook that will let a SciPy backend report values too.
        """
        if self.n_vals == 0:
            return np.zeros((0, t_arr.size))
        v = np.empty((self.n_vals, t_arr.size))
        for j in range(t_arr.size):
            vals = self._call_model(t_arr[j], x_arr[:, j])[1]
            if vals.size != self.n_vals:
                raise ModelError(
                    "Your model returned a different number of values at "
                    f"t = {t_arr[j]:g} than it did at t = 0.\n"
                    f"  at t = 0    : {self.n_vals} value(s)\n"
                    f"  at t = {t_arr[j]:g} : {vals.size} value(s)\n"
                    "The second returned list must have the same length "
                    "every time the model is called. A value that only "
                    "exists on one branch of an if-statement will do this."
                )
            v[:, j] = vals
        return v

    def plot_results(self,n_plots=2,tdata=None,ydata=None,ylabels=None,xlabel=None):
        fig, axs = plt.subplots(n_plots, 1, sharex=True, figsize=(6,2*n_plots))
        for ax,yi,label in zip(axs,ydata,ylabels):
            for yii in yi:
                ax.plot(tdata,yii)
            ax.set_ylabel(label)
            ax.grid(ls='--')
        axs[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.show()