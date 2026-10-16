# cme452

Teaching tools for **CME 452 — Chemical Process Dynamics and Control**.

The package has one job: let you write a process model the way you'd write it
on paper, simulate it, close a feedback loop around it, and plot the results —
without getting tripped up by Python. Every object maps directly onto the
block diagrams in class.

```
                +---------+    +---------+    +-----------+
   y_sp  ------>(|controller|---|> actuator|---|>  process |---- y
                 +---------+    +---------+    +-----+-----+
                       ^                                   |
                       |          +----------+             |
                       +----------|  sensor   |<------------+
                                   +----------+
```

## What's inside

| Name          | What it is |
|---------------|------------|
| `System`      | Open-loop ODE harness: fixed-step RK4, model checking, extra outputs, plotting. |
| `PID`         | Discrete PI(D) controller: fixed error sign, derivative on the measurement, optional saturation and anti-windup. |
| `ClosedLoop`  | Closes a `PID` around a `System`. One argument per block in the diagram. |
| `stability`   | Eigenvalue analysis: plant Jacobian at an operating point, and the Jacobian of the loop's discrete one-step map. |
| `plot_results`| Stacked time-series panels: one argument == one panel. |

## Installation

From a checkout:

```bash
pip install -e .
```

In Colab:

```python
!pip install git+https://github.com/hinklek1/CME452
```

Dependencies: numpy, matplotlib, tqdm (all installed automatically).

## The model convention

Everything starts from one function, used identically open- and closed-loop:

```python
def model(t, x, params, inputs):
    """Return [rates], one per state."""
    ...
    return [dTdt]
```

* `x` — the state vector (a sequence; `T, = x` unpacks a single state).
* `params` — anything you like; a dict is recommended.
* `inputs` — a **list of functions of time**. Inside the model you *call*
  them: `Q = inputs[0](t)`, `T0 = inputs[1](t)`. Forgetting the `(t)` is the
  classic first bug.
* Optional: return `[[rates], [values]]` to also record extra quantities
  (setpoints, inputs, intermediate variables). They come back in `sys.v`
  and are re-evaluated consistently on the output grid.

### Open-loop example

```python
import numpy as np
from cme452 import System

V, q, rho, Cp = 4, 0.1, 1000, 1.0
params = [V, q, rho, Cp]

def model(t, z, params, inputs):
    T, = z
    V, q, rho, Cp = params
    Q, T0 = inputs                      # functions of time
    dTdt = (rho*q*Cp*T0(t) + Q(t) - rho*q*Cp*T) / (rho*V*Cp)
    return [[dTdt], [Q(t), T0(t)]]      # rates + extra values

def Q(t):  return 500.0 if t > 20 else 0.0
def T0(t): return 0.0

tank = System(model, params, inputs=[Q, T0])
tank.run_simulation(t_end=200, IC=[0.0])
tank.plot_results(tank.x[0], tank.v[0], ylabels=["T", "Q"])
```

`run_simulation(t_end, IC, dt=0.01, n_out=None, t_start=0.0, progress=True)`
returns `(t, x, v)` and also stores them: `sys.t`, `sys.x`, `sys.v` (output
grid) and `sys.t_full`, `sys.x_full` (every computed step). Arrays are
`(n_states, n_time)`, so `sys.x[0]` is the first state's trajectory.

* `n_out` — keep about that many *computed* points (decimation, no
  interpolation). Handy for plots.
* `progress=False` — silence the tqdm bar (use it inside parameter-sweep
  loops).
* The model is evaluated once at `t = 0` before integrating, so size and
  type errors surface immediately with a readable message.

## Closed loops

`ClosedLoop` wraps the same model and adds one slot per block:

```python
from cme452 import ClosedLoop, PID

Km, Kv, Kc = 5/100, 1000/5, 50.0

def T0(t):  return 0.0                            # disturbance
def Tsp(t): return 0.0 if t < 20 else 2.0         # setpoint (process units)

loop = ClosedLoop(
    model, params,
    PID(Kc, 5.0, umin=0.0, umax=1.5),   # controller (PI here)
    measure=0,                          # which process state is sensed
    y_sp=Tsp,                           # number, or function of t
    inputs=T0,                          # disturbances (bare fn, list, or tuple)
    valve=(Kv, 1.0),                    # actuator: (Kv, tauv); None = direct
    valve0=0.5,                         # initial valve position
    sensor=Km,                          # gain, (Km, delay), or a function
)
loop.run_simulation(t_end=200, IC=[0.0])
loop.plot_results([loop.v[0], loop.x[0]], [loop.mv],
                  ylabels=["Response", "Manip. Var."],
                  labels=[("Set Point", "Actual"), None])
```

Argument-to-block correspondence:

| Argument      | Block         | Notes |
|---------------|---------------|-------|
| `model`, `params`, `IC` | process | Exactly the open-loop model. The manipulated variable is `inputs[0](t)`. |
| `controller`  | controller    | A `PID` (or anything with `update(t, y, y_sp)` and `reset()`). |
| `measure`     | sensor (which) | Index of the sensed state, or a function `x -> y`. |
| `y_sp`        | setpoint      | **Process units.** Number or function of `t`. |
| `inputs`      | disturbances  | Enter the process at `inputs[1], inputs[2], ...` — same indices as open-loop. |
| `valve`       | actuator      | `(Kv, tauv)`: first-order valve `dq/dt = (Kv*u - q)/tauv`. `None` = the controller drives `inputs[0]` directly. |
| `valve0`      | —             | Initial valve position (an initial condition, not a controller bias). |
| `sensor`      | transducer (gain, delay) | `Km`, or `(Km, delay)` to add a measurement delay: the controller sees the reading from `delay` time units ago (before the first delayed sample, it sees the initial reading). The delay is quantized to the integration step, like the output hold. A function of `y` is allowed for noise/nonlinearity — then give `y_sp` in sensor units (a function cannot carry a delay). |

Results: `loop.x` (process states — **same indexing as the open-loop model**,
the valve is not in it), `loop.u` (controller output), `loop.mv`
(manipulated variable: valve position when `valve=` is given, else `u`),
plus `loop.v` and the `_full` variants. The controller is called once per
step on the real state and its output is held (zero-order hold), so the I
term accumulates correctly and mid-step evaluations see the move that was
actually in effect.

### The PID

```
u = Kc*e + (Kc/tauI)*I - Kc*tauD*(dy/dt),     dI/dt = e,     e = y_sp - y
```

`PID(Kc, tauI=None, tauD=None, umin=None, umax=None, antiwindup=True)`

* The sign is fixed (`e = y_sp - y`): you cannot build a positive loop by
  accident.
* Derivative acts on the **measurement**, so a setpoint step produces no
  derivative kick.
* `umin`/`umax` clamp the **controller output `u`** — the signal *into*
  the actuator. With a valve of gain `Kv`, a physical limit `Q_max` on the
  manipulated variable is `umax = Q_max / Kv`.
* With limits set, the I term stops accumulating while saturated
  (conditional integration, the standard simple anti-windup).
  `antiwindup=False` switches it off — the fastest way to show windup:
  same loop, one flag, and the overshoot gets dramatically worse
  (in the class setpoint-step example: 0.07 becomes 0.79).

### Parameter sweeps

A plain loop; append each run's trajectory to a list:

```python
temps = []
for Kc in [10.0, 25.0, 50.0, 100.0]:
    loop = ClosedLoop(model, params, PID(Kc, 5.0, umin=0.0, umax=1.5),
                      measure=0, y_sp=Tsp, inputs=T0,
                      valve=(Kv, 1.0), valve0=0.5, sensor=Km)
    loop.run_simulation(t_end=200, IC=[0.0], progress=False)
    temps.append(loop.x[0])

for Kc, T in zip([10.0, 25.0, 50.0, 100.0], temps):
    plt.plot(loop.t, T, label=f"Kc = {Kc}")
plt.legend(); plt.xlabel("t"); plt.ylabel("T")
```

Diverged runs usually don't raise — they return huge values, and only
raise `ModelError` once a derivative goes non-finite. Guard with
`np.all(np.isfinite(loop.x))` (or a magnitude threshold) before appending.

## Stability analysis (eigenvalues)

Two objects, matching the two time domains:

* **Open loop** — the Jacobian of dx/dt at an operating point.  All
  eigenvalues in the left half plane (Re < 0) is stable; a complex pair
  with imaginary part `w` rings with period `2*pi/w`.
* **Closed loop** — the loop is *discrete* (controller called once per
  step, output held), so the honest object is the **one-step map**:
  full state in, one integration step, full state out.  The state is the
  augmented vector `[valve?, process states..., controller memory...]`
  (`I` and `e_prev` for PI/PID, `y_prev` for PD/PID).  The test is
  `|lambda| < 1`, and a complex pair with argument `theta` rings with
  period `2*pi/theta` steps.

```python
from cme452 import System, ClosedLoop, PID, stability

# open loop: linearize the plant at an operating point
J  = stability.jacobian(sys, x_op=[340.0])
ev = stability.eigenvalues(J)                    # np.linalg.eigvals
stability.eigenvalue_plot(ev)                     # Re = 0 line

# closed loop: the one-step map at the end of a run
loop.run_simulation(t_end=60, IC=[0.0])
z_op = stability.augmented_state(loop)            # [q?, x..., I?, ...]
J    = stability.closed_loop_jacobian(loop, t_op=loop.t[-1], z_op=z_op)
ev   = stability.eigenvalues(J)
stability.eigenvalue_plot(ev, discrete=True)      # unit circle
```

The natural sweep: rebuild the loop for each gain, run, linearize — and
watch the eigenvalues cross the unit circle as Kc grows.
`stability.one_step(loop, t_op, z)` is the nonlinear map itself (what the
Jacobian linearizes); iterate `z + J @ (z - z_op)` by hand and compare.

Two caveats, both worth showing students:

* Linearization is **local** — it says nothing about trajectories far
  from the operating point.
* A sensor **delay** makes the true loop infinite-dimensional (the state
  would be the whole measurement history), so `closed_loop_jacobian`
  refuses `sensor=(Km, delay)` loops and tells you to treat the delay as
  extra phase instead.

## plot_results

One rule: **one argument == one panel.**

```python
sys.plot_results(sys.x[0], sys.v[0])      # 2 panels, 1 curve each
sys.plot_results([a, b], [c, d])          # 2 panels, 2 curves each
sys.plot_results(sys.x)                   # 1 panel, all states overlaid
sys.plot_results(*sys.x)                  # 1 panel per state
```

Each argument is a 1-D series, a list of 1-D series, or a 2-D array
(one curve per row). Keywords: `ylabels`, `labels` (one entry per panel;
each entry a list matching that panel's curve count, or `None`), `xlabel`,
`title`, `figsize`, `ylims`, `sharex`, `grid`, `show`. `sys.plot_results(...)`
uses the last run's `t` automatically; the module-level `plot_results(t, ...)`
takes the time vector explicitly.

## Tests

```bash
PYTHONPATH=src python tests/test_closedloop.py    # 35 checks
PYTHONPATH=src python tests/test_stability.py     # 9 checks
```

The first: P/PI behavior, anti-windup on/off, derivative kick,
zero-order-hold timing, valve-ODE accuracy against the analytic solution,
sensor-gain invariance, measurement delay, and error messages.  The
second: Jacobians against hand-computed eigenvalues, predicted vs
simulated ringing period, the one-step map against its exact ZOH
discretization, and that linearizing disturbs nothing.

## Project layout

```
src/cme452/
    core.py        System: RK4 harness, probing, values, decimation, progress
    controller.py  PID
    loop.py        ClosedLoop: controller scheduling, ZOH, valve, sensor + delay
    stability.py   jacobian, one-step map, eigenvalue plots
    plotting.py    plot_results
tests/
    test_closedloop.py
    test_stability.py
```
