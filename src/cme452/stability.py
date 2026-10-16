"""
stability  --  eigenvalue analysis of open- and closed-loop models
===============================================================

Two questions, two objects:

* OPEN LOOP (continuous time).  The student's model, linearized at an
  operating point: the Jacobian of dx/dt with respect to x.  Its
  eigenvalues say whether the point is stable (all real parts negative),
  and a complex pair's imaginary part is the damped frequency -- the
  ringing you see in the simulation, with period 2*pi/omega.

* CLOSED LOOP (discrete time).  A ClosedLoop is a digital loop: the
  controller is called once per step and its output is held.  The honest
  object to linearize is therefore the ONE-STEP MAP: take the full state
  (valve position if present, process states, and the controller's
  memory), advance the loop exactly one integration step, and look at
  how the state moves.  The Jacobian of that map is finite-differenced
  here, and the eigenvalue test is |lambda| < 1 instead of Re < 0:
  inside the unit circle the loop is stable, and a complex pair's
  argument gives the ringing period in STEPS (period = 2*pi/angle * dt).

Operating point
---------------
Linearization is LOCAL.  The results say nothing about trajectories far
from the point you linearized at -- a loop can be stable at its
setpoint and unstable if a disturbance pushes it elsewhere.  That
limitation is the point of the exercise, not a bug.

Known limitation: a sensor delay makes the true closed loop
infinite-dimensional (the state would be the whole measurement history),
so closed_loop_jacobian refuses loops built with sensor=(Km, delay).

Examples
--------
    from cme452 import System, ClosedLoop, PID, stability

    # open loop: linearize the plant at an operating point
    J  = stability.jacobian(tank, x_op=[340.0])
    ev = stability.eigenvalues(J)                 # np.linalg.eigvals
    stability.eigenvalue_plot(ev)                 # complex plane, Re = 0 line

    # closed loop: the discrete one-step map at the end of a run
    loop.run_simulation(t_end=60, IC=[0.0])
    z_op = stability.augmented_state(loop)        # [q?, x..., I?, ...]
    J    = stability.closed_loop_jacobian(loop, t_op=loop.t[-1], z_op=z_op)
    ev   = stability.eigenvalues(J)
    stability.eigenvalue_plot(ev, discrete=True)  # unit circle

    # the nonlinear map itself (what J linearizes):
    z_next = stability.one_step(loop, t_op=loop.t[-1], z=z_op)

    # sweep Kc and watch the eigenvalues cross the unit circle
    for Kc in [1, 2, 4, 8]:
        ...run...
        print(stability.eigenvalues(stability.closed_loop_jacobian(...)))
"""

import numpy as np

from .core import ModelError
from .loop import ClosedLoop


# ------------------------------------------------------------------
#  OPEN LOOP  (continuous time)
# ------------------------------------------------------------------
def jacobian(sys, x_op, t=0.0):
    """Finite-difference the Jacobian of dx/dt at an operating point.

    sys   : a System (or ClosedLoop -- then the actuator is included and
            the controller is NOT: you get the open-loop plant as the
            integrator sees it)
    x_op  : operating point, one value per state the SYSTEM sees
            (process states; plus the valve position first, if the
            system is a ClosedLoop with valve=)
    t     : time at which to evaluate (inputs/disturbances are frozen
            at this value)

    Returns the (n, n) Jacobian d(dx/dt)/dx.  Central differences, so
    the error is O(eps^2) in the step and O(machine eps^0.5) overall.
    """
    x_op = np.atleast_1d(np.asarray(x_op, dtype=float)).ravel()
    n = x_op.size
    rates = sys._rates(t, x_op)          # also surfaces model errors early
    if np.asarray(rates).size != n:
        raise ModelError(
            f"x_op has {n} entries but the model returned "
            f"{np.asarray(rates).size} rate(s) at that point. Every state "
            "needs exactly one initial condition and one dx/dt."
        )
    J = np.empty((n, n))
    for j in range(n):
        eps = 1e-6 * max(1.0, abs(x_op[j]))
        xp = x_op.copy(); xp[j] += eps
        xm = x_op.copy(); xm[j] -= eps
        J[:, j] = (sys._rates(t, xp) - sys._rates(t, xm)) / (2.0 * eps)
    return J


def eigenvalues(J):
    """Eigenvalues of a Jacobian.  A thin, documented wrapper around
    np.linalg.eigvals -- the test of stability is where they land:
    Re < 0 for continuous systems, |lambda| < 1 for step maps."""
    return np.linalg.eigvals(np.asarray(J, dtype=float))


def eigenvalue_plot(eigs, discrete=False, ax=None, title=None):
    """Eigenvalues in the complex plane.

    discrete=False: continuous time -- the dividing line is Re = 0
                    (right half plane shaded as unstable).
    discrete=True : step map -- the dividing line is the unit circle
                    (outside shaded as unstable).

    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    eigs = np.asarray(eigs)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 5.0))
    re, im = eigs.real, eigs.imag

    if discrete:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.fill(np.cos(th), np.sin(th), color="red", alpha=0.06)
        ax.plot(np.cos(th), np.sin(th), "r--", lw=1)
        m = np.max(np.abs(eigs)) if eigs.size else 1.0
        lim = max(1.25, 1.25 * m)
    else:
        ax.axvline(0.0, color="r", ls="--", lw=1)
        m = max(1.0, np.max(np.abs(eigs))) if eigs.size else 1.0
        lim = 1.25 * m
    ax.set_xlabel("Re(lambda)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("Im(lambda)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.plot(re, im, "ko", ms=6, mfc="orange",
            label="eigenvalue" if eigs.size else None)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    return ax


# ------------------------------------------------------------------
#  CLOSED LOOP  (discrete one-step map)
# ------------------------------------------------------------------
def _mem_layout(controller):
    """Which controller-memory coordinates actually affect the move.

    Returns a list of (attribute, name) pairs, in the order the memory
    sits inside the augmented state:
        P   -> []
        PI  -> [I, e_prev]          (trapezoid uses the previous error)
        PD  -> [y_prev]             (derivative on the measurement)
        PID -> [I, e_prev, y_prev]

    Memory that is not used is left OUT on purpose: including it would
    add spurious zero eigenvalues that a student would have to
    explain away.
    """
    if not (hasattr(controller, "tauI") and hasattr(controller, "tauD")):
        raise ModelError(
            "the stability functions support PID controllers (they need "
            f"to know the controller's memory layout). You passed a "
            f"{type(controller).__name__}."
        )
    layout = []
    if controller.tauI is not None:
        layout += [("_I", "I"), ("_e", "e_prev")]
    if controller.tauD is not None:
        layout += [("_y", "y_prev")]
    return layout


def _split(loop, z):
    """Augmented state -> (plant state as the integrator sees it, mem)."""
    n_s = loop.n_states
    return np.asarray(z, dtype=float).ravel()[:n_s], \
        np.asarray(z, dtype=float).ravel()[n_s:]


def augmented_state(loop):
    """The full closed-loop state at the END of the last run:

        [valve position (if valve= was given), process states...,
         controller memory (I and e_prev for PI/PID, y_prev for PD/PID)]

    Feed it to closed_loop_jacobian as z_op to linearize the loop where
    it just finished -- usually its steady state.
    """
    if loop.x_full is None:
        raise ModelError(
            "augmented_state needs a finished run_simulation first "
            "(it reads the final state and the controller's current "
            "memory)."
        )
    parts = []
    if loop._valve is not None:
        parts.append(np.atleast_1d(loop.mv_full[-1]))
    parts.append(loop.x_full[:, -1])
    for attr, _name in _mem_layout(loop.controller):
        parts.append(np.atleast_1d(getattr(loop.controller, attr)))
    return np.concatenate(parts)


def _check(loop, z):
    """Shared validation for one_step / closed_loop_jacobian."""
    if not isinstance(loop, ClosedLoop):
        raise ModelError("expected a ClosedLoop (you passed a "
                         f"{type(loop).__name__}).")
    if loop.dt is None:
        raise ModelError(
            "this needs the integration step: run "
            "loop.run_simulation(...) first so the loop knows its dt."
        )
    if loop._delay > 0:
        raise ModelError(
            "a sensor delay is not supported: the delayed loop's state "
            "would be the whole measurement history (an "
            "infinite-dimensional system), so a single Jacobian cannot "
            "describe it. Linearize the no-delay loop and treat the "
            f"delay as extra phase ({loop._delay:g} time units of it)."
        )
    layout = _mem_layout(loop.controller)
    n_s = loop.n_states
    z = np.asarray(z, dtype=float).ravel()
    if z.size != n_s + len(layout):
        names = ", ".join(
            (["valve"] if loop._valve is not None else [])
            + [f"x{i}" for i in range(n_s - (1 if loop._valve is not None else 0))]
            + [name for _a, name in layout]
        )
        raise ModelError(
            f"the state has {z.size} entries but this loop's augmented "
            f"state has {n_s + len(layout)}: [{names}]. "
            "augmented_state(loop) builds it for you after a run."
        )
    return loop.dt, layout, z


def one_step(loop, t_op, z):
    """Advance a ClosedLoop exactly one integration step from z.

    z is the augmented state [valve (if given), process states..., controller
    memory...].  This is the map that closed_loop_jacobian differentiates:
    read the state, call controller.update once, hold the output, RK4 the
    plant.  Handy for seeing linearization in action -- iterate
    x_{k+1} = x + J (x - z_op) by hand and compare with one_step.

    The loop's controller memory and zero-order hold are restored exactly
    as found, so the call cannot disturb a later run_simulation.
    """
    dt, layout, z = _check(loop, z)
    ctrl = loop.controller
    s, m = _split(loop, z)
    y_meas = float(np.asarray(loop.measure(s), dtype=float).ravel()[0])
    y_s, sp = loop._transduce(y_meas, t_op)
    e_meas = sp - y_s                            # the controller's error

    # seed the controller's memory from z (unused coords get values
    # consistent with the operating point, not garbage); _t is set so
    # update() integrates over exactly THIS run's dt
    mem = {"_I": 0.0, "_y": y_meas, "_e": e_meas}
    for (attr, _name), val in zip(layout, m):
        mem[attr] = float(val)
    mem["_t"] = t_op - dt

    saved = (ctrl._I, ctrl._t, ctrl._y, ctrl._e)
    saved_zoh = (loop._zoh._t.copy(), loop._zoh._u.copy())
    for attr, val in mem.items():
        setattr(ctrl, attr, val)

    u = ctrl.update(t_op, y_s, sp)
    # apply feedforward the same way the loop does (loop.py ff_move):
    # the ZOH holds u_PID + Kff*d_j, not the bare controller output
    if loop._ff is not None:
        Kff, j = loop._ff
        d = float(np.asarray(loop.inputs[j + 1](t_op), dtype=float).ravel()[0])
        u = u + Kff * d
    loop._zoh.set(t_op, u)

    # one RK4 step of the plant (valve + process), as the loop does it
    f = loop._rates
    h = dt
    k1 = f(t_op, s)
    k2 = f(t_op + 0.5 * h, s + 0.5 * h * k1)
    k3 = f(t_op + 0.5 * h, s + 0.5 * h * k2)
    k4 = f(t_op + h, s + h * k3)
    s_next = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    m_next = np.array([getattr(ctrl, attr) for attr, _n in layout],
                      dtype=float)
    ctrl._I, ctrl._t, ctrl._y, ctrl._e = saved
    loop._zoh._t, loop._zoh._u = saved_zoh
    return np.concatenate([np.asarray(s_next).ravel(), m_next])


def closed_loop_jacobian(loop, t_op, z_op):
    """Jacobian of the discrete ONE-STEP MAP of a ClosedLoop, at z_op.

    The map is exactly what the simulation does in one step (one_step):
    read the state, call controller.update once, hold the output
    (zero-order hold), advance the plant (valve + process) with RK4.
    This Jacobian is finite-differenced around z_op, so its eigenvalues
    are the discrete-time stability test: |lambda| < 1 is stable, and a
    complex pair with argument theta rings with period 2*pi/theta steps
    (= 2*pi/theta * dt time units).

    loop  : a ClosedLoop (run at least once, so the map knows its dt)
    t_op  : operating time; inputs, disturbances, and the setpoint are
            evaluated at this value
    z_op  : operating point of the augmented state, laid out as
            [valve (if given), process states..., controller memory...]
            -- augmented_state(loop) produces exactly this after a run.

    The loop's controller memory and zero-order hold are restored
    exactly as found, so the call cannot disturb a later run.
    """
    _dt, _layout, z_op = _check(loop, z_op)

    J = np.empty((z_op.size, z_op.size))
    for j in range(z_op.size):
        eps = 1e-6 * max(1.0, abs(z_op[j]))
        zp = z_op.copy(); zp[j] += eps
        zm = z_op.copy(); zm[j] -= eps
        J[:, j] = (one_step(loop, t_op, zp) - one_step(loop, t_op, zm)) \
                  / (2.0 * eps)
    return J
