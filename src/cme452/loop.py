"""
ClosedLoop  --  a plant System with a discrete controller closed around it
==========================================================================

The arguments map one-to-one onto the blocks of the control loop:

    y_sp --+--> [ controller ] -- u --> [ valve ] -- mv --> [ process ] --+
           |            ^                                                  |
           |            |         [ sensor ]                               |
           |            +--------------------------------------------------+
           |
    (inputs: disturbances enter the process, after the manipulated variable)

    loop = ClosedLoop(model, params,
                      PID(0.5, 8.0, umin=0.0, umax=1.0),
                      measure=0, y_sp=300.0,
                      inputs=(T0_fn,),
                      valve=(1.0, 2.0), valve0=0.3,
                      sensor=2.0)
    loop.run_simulation(t_end=40, IC=[250.0])

The blocks:

  model, params, IC   the PROCESS, written exactly as for an open-loop
                      System: a list of ODEs whose manipulated variable
                      is inputs[0](t). IC holds PROCESS initial
                      conditions only.
  controller          an object with update(t, y, y_sp) -> u and reset(),
                      e.g. PID. Its gains live in the object. Exactly one
                      of controller= and source= is required.
  source              instead of a controller, PRESCRIBE the manipulated
                      variable directly: an open-loop test signal. A tuple
                      (w, A) means u(t) = A*sin(w*t) (w the angular
                      frequency, A the amplitude; A defaults to 1.0), or a
                      function t -> u for an arbitrary drive. With source=
                      there is no feedback: y_sp is ignored and the loop is
                      the open chain  source -> valve -> process -> y ->
                      sensor. This is how a frequency scan (cme452.frequency)
                      drives the loop. The source is a settable property, so
                      a sweep can reassign loop.source per frequency.
  measure             which process state the sensor reads: a state index
                      (measure=0 reads the first process state) or a
                      function of the process state, x -> y, e.g. for a
                      differential measurement.
  y_sp                the setpoint, in PROCESS units: a number, or a
                      function of time (def SP(t)) for a moving setpoint.
  inputs              additional input functions (disturbances, feeds),
                      placed AFTER the manipulated variable, so they keep
                      the same indices as in the open-loop model: a plant
                      written open-loop with inputs = [Q_fn, T0_fn]
                      becomes ClosedLoop(..., inputs=(T0_fn,)) -- the
                      hold takes index 0 and T0 stays at inputs[1](t).
  valve               the ACTUATOR, as (Kv, tauv): a first-order valve,
                      d(mv)/dt = (Kv*u - mv)/tauv. The controller output
                      u is then a signal (normalize it to 0..1 and set
                      umin=0, umax=1 on the controller). valve=None means
                      the controller output acts directly as the
                      manipulated variable. Anything more exotic than a
                      first-order actuator (deadband, on/off, 2nd order)
                      is written as state(s) inside the model, with
                      valve=None.
  valve0              the initial valve position (required when valve= is
                      given). This is an initial condition -- where the
                      hardware starts -- not a controller bias.
  sensor              the TRANSDUCER: the gain Km (default 1.0), or a
                      two-item sequence (Km, delay) to add a measurement
                      delay of `delay` time units. The controller sees
                      e = Km*(y_sp - y): the sensor reads Km*y, the
                      setpoint is entered as Km*y_sp, and the gain
                      cancels in the subtraction. With Kc fixed, changing
                      Km changes the effective loop gain Kc*Km -- which
                      is why Kc must be quoted per sensor unit. With a
                      delay, the controller sees the reading from `delay`
                      time units ago; before the first delayed sample
                      exists it sees the initial reading (the sensor was
                      already measuring at t_start, its output just takes
                      time to arrive). The delay is quantized to the
                      integration step, like the zero-order hold on the
                      output side. (An advanced escape hatch: pass a
                      function of the measured value, y -> y_sensor, for
                      sensor noise or nonlinearity; then y_sp must be in
                      sensor units. A function cannot carry a delay.)

How the loop runs: at every integration step the controller is called
ONCE on the real state -- never on RK4's fictitious mid-step states --
and its output is held by a zero-order hold until the next step. That is
the physical picture of a digital control loop, and it is what keeps the
I-term accumulated correctly. With source= there is no controller: the
prescribed signal is sampled at each step and held by the same ZOH, so the
valve and the disturbance timing are identical to the closed loop.

Results after run_simulation: everything System has (t, x, v, t_full,
x_full) plus

    loop.u      controller output (actuator signal; with source=, the
                prescribed signal), on the output grid
    loop.u_full the same, at every integration step
    loop.mv     the manipulated variable: the valve position when
                valve= is given, identical to u otherwise

When valve= is given, x and x_full contain the PROCESS states only --
the valve is reported in mv -- so state indexing matches the student's
open-loop model exactly.
"""

import numpy as np

from .core import System, ModelError, _is_listlike

def _as_input_list(inputs):
    """Accept a bare function or a list/tuple of functions, as System does."""
    if callable(inputs):
        return [inputs]
    return list(inputs)


class _Delay:
    """A measurement delay buffer: answers __call__(t) with the reading
    taken `delay` time units before t.

    Recordings come in on the integration grid (one per step, in time
    order), so the look-back is a search for the last recording at or
    before t - delay. Before any such recording exists, the buffer
    answers y0: the sensor was already measuring at t_start, its output
    just has not arrived yet. The delay is quantized to a whole number
    of steps, like the zero-order hold on the output side.
    """
    def __init__(self, delay, dt, y0):
        self._delay = delay
        self._offset = dt * int(round(delay / dt)) if delay > 0 else 0.0
        self._y0 = y0
        self._t = np.zeros(0)
        self._y = np.zeros(0)

    def add(self, t, y):
        self._t = np.append(self._t, t)
        self._y = np.append(self._y, y)

    def __call__(self, t):
        tq = t - self._offset
        if self._t.size == 0 or tq <= self._t[0]:
            return self._y0
        j = np.searchsorted(self._t, tq, side="right") - 1
        return self._y[max(j, 0)]


class _Hold:
    """A zero-order hold that remembers WHICH value was in effect WHEN.

    Records (t, u) pairs -- one per integration step -- and answers
    __call__(t) with the most recent value set at or before t. Being
    time-aware matters twice: (1) inside one RK4 step the three
    mid-stage evaluations at t + h/2 and t + h must all see the same
    held value, and (2) the values mechanism re-evaluates the model at
    PAST times, so the input there must be the move that was actually in
    effect at t, not the last one.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._t = np.zeros(0)
        self._u = np.zeros(0)

    def set(self, t, u):
        if self._t.size and t == self._t[-1]:
            self._u[-1] = u          # same time again: overwrite, don't append
            return
        self._t = np.append(self._t, t)
        self._u = np.append(self._u, u)

    def __call__(self, t):
        if self._t.size == 0:
            return 0.0
        j = np.searchsorted(self._t, t, side="right") - 1
        return self._u[max(j, 0)]


class ClosedLoop(System):
    """A plant model with a discrete controller in feedback. See the
    module docstring for the full argument-by-argument description; each
    argument is one block of the loop diagram."""

    def __init__(self, model, params, controller=None, measure=0,
                 y_sp=0.0, inputs=(), valve=None, valve0=None, sensor=1.0,
                 feedforward=None, source=None):
        # ---- valve (actuator) ---------------------------------------
        self._valve = None
        self._valve0 = None
        if valve is not None:
            try:
                Kv, tauv = float(valve[0]), float(valve[1])
            except (TypeError, IndexError, ValueError):
                raise ModelError(
                    "valve must be (Kv, tauv): the actuator gain and its "
                    f"time constant. You passed {valve!r}."
                )
            if not np.isfinite(Kv) or Kv == 0.0:
                raise ModelError(f"valve gain Kv must be a nonzero finite number (got {Kv}).")
            if not np.isfinite(tauv) or tauv <= 0.0:
                raise ModelError(f"valve time constant tauv must be positive (got {tauv}).")
            self._valve = (Kv, tauv)
            if valve0 is None:
                raise ModelError(
                    "valve0 (the initial valve position) is required when "
                    "valve= is given."
                )
            self._valve0 = float(valve0)
        elif valve0 is not None:
            raise ModelError(
                "valve0 was given but no valve=. The initial valve "
                "position only exists when a valve is declared."
            )

        # ---- sensor (transducer) -------------------------------------
        self._delay = 0.0
        if callable(sensor):
            self._sensor_fn = sensor
            self._sensor_gain = None
        elif _is_listlike(sensor):
            if len(sensor) != 2:
                raise ModelError(
                    "a sensor sequence must be (Km, delay): the transducer "
                    f"gain and a measurement delay. You passed {sensor!r} "
                    "with " + str(len(sensor)) + " item(s); for a gain "
                    "alone, pass the number directly (sensor=Km)."
                )
            try:
                Km, d = float(sensor[0]), float(sensor[1])
            except (TypeError, ValueError):
                raise ModelError(
                    "a sensor sequence must be (Km, delay): the transducer "
                    f"gain and a measurement delay. You passed {sensor!r}."
                )
            if not np.isfinite(Km) or Km <= 0.0:
                raise ModelError(f"sensor gain Km must be a positive number (got {Km}).")
            if not np.isfinite(d) or d < 0.0:
                raise ModelError(f"sensor delay must be a nonnegative number (got {d}).")
            self._delay = d
            self._sensor_fn = None
            self._sensor_gain = Km
        else:
            sensor = float(sensor)
            if not np.isfinite(sensor) or sensor <= 0.0:
                raise ModelError(f"sensor gain Km must be a positive number (got {sensor}).")
            self._sensor_fn = None
            self._sensor_gain = sensor

        # ---- controller OR source (the u block) ----------------------
        # Exactly one: a controller closes the loop; a source prescribes
        # the manipulated variable (an open-loop test signal).
        if (controller is None) == (source is None):
            raise ModelError(
                "give exactly one of controller= or source=: controller= "
                "closes the feedback loop, source= prescribes the "
                "manipulated variable directly (open-loop test signal). "
                "You passed "
                + ("both." if controller is not None else "neither.")
            )
        self.controller = None
        self._source = None
        if controller is not None:
            if not (hasattr(controller, "update")
                    and hasattr(controller, "reset")):
                raise ModelError(
                    "controller must have update(t, y, y_sp) and reset() "
                    "methods (e.g. a PID). You passed a "
                    f"{type(controller).__name__}."
                )
            self.controller = controller
        else:
            self._set_source(source)

        # ---- measure (which process state is sensed) ------------------
        self._measure_idx = None
        if isinstance(measure, (int, np.integer)):
            if measure < 0:
                raise ModelError(f"measure={measure} is negative.")
            self._measure_idx = int(measure)
            if self._valve is not None:
                # valve state is prepended by the wrapper; index into the
                # PROCESS state
                measure = (lambda x, i=self._measure_idx: x[1 + i])
            else:
                measure = (lambda x, i=self._measure_idx: x[i])
        elif not callable(measure):
            raise ModelError(
                "measure must be a state index (measure=0 measures the "
                "first state) or a function of the state: measure(x) -> y."
            )
        elif self._valve is not None:
            # the wrapper prepends the valve state; a function measure is
            # documented to read the PROCESS state, so hand it x[1:]
            # (exactly what the index path does with its +1 offset)
            measure = (lambda x, f=measure: f(x[1:]))
        self.measure = measure

        self.y_sp = y_sp
        self._sp = y_sp if callable(y_sp) else (lambda t: float(y_sp))

        self._zoh = _Hold()
        self._delay_buf = None     # built at the start of each run
        self._plant = model        # the student's process model, as written
        super().__init__(self._wrapped_model, params,
                         inputs=[self._zoh, *_as_input_list(inputs)])

        # ---- feedforward ------------------------------------------
        # A static gain on a chosen disturbance, summed with the
        # controller output at the actuator:  actuator input =
        # u_PID + Kff * d_j(t).  The controller's umin/umax clamp limits
        # the FEEDBACK term only; the sum is what the actuator sees.
        # (In simulation the disturbance trajectory is known -- in the
        # real world you would measure it.)
        self._ff = None
        if feedforward is not None:
            n_in = len(self.inputs) - 1
            if n_in == 0:
                raise ModelError(
                    "feedforward needs a disturbance to act on: give "
                    "inputs= with at least one disturbance function."
                )
            if _is_listlike(feedforward) and len(feedforward) != 2:
                raise ModelError(
                    "feedforward must be (Kff, j): a gain and the index "
                    f"of the disturbance in inputs (got {len(feedforward)} "
                    "values)."
                )
            try:
                Kff, j = float(feedforward[0]), feedforward[1]
            except (TypeError, IndexError, ValueError):
                raise ModelError(
                    "feedforward must be (Kff, j): a gain and the index "
                    "of the disturbance in inputs."
                ) from None
            if not np.isfinite(Kff):
                raise ModelError(
                    f"feedforward gain Kff must be finite (got {Kff})."
                )
            if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
                raise ModelError(
                    "feedforward channel j must be an integer index into "
                    f"inputs (got {j!r})."
                )
            j = int(j)
            if not (0 <= j < n_in):
                raise ModelError(
                    f"feedforward channel j={j} but inputs has {n_in} "
                    f"disturbance(s); valid indices are 0..{n_in - 1}."
                )
            self._ff = (Kff, j)

        self.u_full = None
        self.u = None
        self.mv_full = None
        self.mv = None

    # ------------------------------------------------------------------
    def _wrapped_model(self, t, xx, params, inputs):
        """The plant as seen by the integrator.

        Without a valve this IS the student's model. With a valve it
        prepends the valve state q:  dq/dt = (Kv*u - q)/tauv,  and feeds
        q into the process as inputs[0](t). The process model itself is
        called with the process state only (xx[1:]) and the same
        disturbance inputs, so it is written exactly as open-loop.
        """
        if self._valve is None:
            return self._plant(t, xx, params, inputs)
        Kv, tauv = self._valve
        q = xx[0]
        q_now = (lambda t, _q=q: _q)
        out = self._plant(t, xx[1:], params, [q_now] + list(inputs[1:]))
        dqv = (Kv * inputs[0](t) - q) / tauv
        # preserve the [[rates],[values]] form if the process uses it
        if (_is_listlike(out) and len(out) == 2
                and _is_listlike(out[0]) and _is_listlike(out[1])):
            rates, vals = out
            return [[dqv] + list(rates), vals]
        if not _is_listlike(out):
            out = [out]
        return [dqv] + list(out)

    def _set_source(self, source):
        """Validate and store an open-loop source (t -> u) for the
        manipulated variable. A (w, A) tuple becomes A*sin(w*t); A
        defaults to 1.0. A callable is used as-is."""
        if callable(source):
            self._source = (lambda t, f=source: float(f(t)))
            return
        try:
            if _is_listlike(source):
                if len(source) == 1:
                    w, A = float(source[0]), 1.0
                elif len(source) == 2:
                    w, A = float(source[0]), float(source[1])
                else:
                    raise ValueError
            else:
                # a bare number is a frequency with unit amplitude
                w, A = float(source), 1.0
        except (TypeError, ValueError, IndexError):
            raise ModelError(
                "source must be a (w, A) tuple for A*sin(w*t), a bare "
                "frequency w (unit amplitude), or a function t -> u. "
                f"You passed {source!r}."
            ) from None
        if not np.isfinite(w) or w <= 0.0:
            raise ModelError(f"source frequency w must be positive (got {w}).")
        if not np.isfinite(A):
            raise ModelError(f"source amplitude A must be finite (got {A}).")
        self._source = (lambda t, w=w, A=A: A * np.sin(w * t))

    @property
    def source(self):
        """The open-loop source (t -> u), or None when controller= is set.
        Settable: a frequency sweep reassigns loop.source per frequency."""
        return self._source

    @source.setter
    def source(self, value):
        self._set_source(value)

    def _transduce(self, y, t):
        """Apply the sensor: return (sensor reading, setpoint) in the
        units the controller works in. For a gain Km both are scaled:
        the controller then sees e = Km*(y_sp - y)."""
        if self._sensor_fn is not None:
            y_s = float(np.asarray(self._sensor_fn(y), dtype=float).ravel()[0])
            return y_s, self._sp(t)
        return self._sensor_gain * y, self._sensor_gain * self._sp(t)

    # ------------------------------------------------------------------
    @property
    def manipulated(self):
        """Alias for mv -- the manipulated variable on the output grid."""
        return self.mv

    # ------------------------------------------------------------------
    def run_simulation(self, t_end, IC, dt=0.01, n_out=None, t_start=0.0,
                       progress=True):
        """As System.run_simulation, but IC holds PROCESS initial
        conditions only (the valve starts at valve0). The controller is
        reset and called once per step; loop.u records its output, loop.mv
        the manipulated variable (valve position, or the held actuator
        input -- controller output plus feedforward -- when valve=None)."""
        IC = np.atleast_1d(np.asarray(IC, dtype=float)).ravel()

        if self._valve is not None:
            # Probe the PROCESS model with a bare System so a bad IC
            # produces the clean, process-level error message.
            System(self._plant, self.params,
                   inputs=[(lambda t: self._valve0)] + list(self.inputs[1:]))._probe(IC)
            IC_full = np.concatenate(([self._valve0], IC))
        else:
            IC_full = IC

        if self._measure_idx is not None and self._measure_idx >= IC.size:
            raise ModelError(
                f"measure={self._measure_idx} but the process has only "
                f"{IC.size} state(s); valid indices are 0..{IC.size - 1}."
            )

        # measure must return a single number
        y0 = np.asarray(self.measure(IC_full), dtype=float).ravel()
        if y0.size != 1:
            raise ModelError(
                "measure must return a single number; it returned "
                f"{y0.size} values: {y0.tolist()}"
            )

        self._zoh.reset()
        self._u_pid = []              # actuator signal (u), per step
        using_source = self._source is not None
        if not using_source:
            self.controller.reset()

        def ff_move(t, u_pid):
            # the actuator sees the controller output plus the feedforward
            if self._ff is None:
                return u_pid
            Kff, j = self._ff
            d = float(np.asarray(self.inputs[j + 1](t), dtype=float).ravel()[0])
            return u_pid + Kff * d

        y0 = float(y0[0])

        # Measurement delay: the controller sees the reading taken
        # `delay` time units ago; before the first delayed sample exists
        # it sees y0 (the sensor was already measuring at t_start).
        # Quantized to the (snapped) integration step, like the ZOH.
        # With a source there is no controller, so the delay buffer is
        # unused -- but we still build it so the code path is uniform.
        # Snap exactly as System.run_simulation does (its self.dt is only
        # assigned once that call is below, so replicate the arithmetic).
        span = float(t_end) - float(t_start)
        dt_used = span / max(int(round(span / dt)), 1)
        self._delay_buf = _Delay(self._delay, dt_used, y0)

        # Prime the hold with the first move. System probes the model once
        # at t_start before integrating, and the plant should see a
        # physical input there -- not 0.
        if using_source:
            u0 = self._source(t_start)
        else:
            # The repeated call in on_step at the same t is a no-op for the
            # controller: dt = 0. (No _u_pid append here: on_step runs
            # again at t_start for the first step and records that move.)
            u0 = self.controller.update(
                t_start, *self._transduce(self._delay_buf(t_start), t_start))
        self._zoh.set(t_start, ff_move(t_start, u0))

        def on_step(t, x):
            if using_source:
                u_pid = self._source(t)
            else:
                y = float(np.asarray(self.measure(x), dtype=float).ravel()[0])
                self._delay_buf.add(t, y)
                u_pid = self.controller.update(
                    t, *self._transduce(self._delay_buf(t), t))
            self._u_pid.append(u_pid)
            self._zoh.set(t, ff_move(t, u_pid))

        (self.t, self.x, self.v) = super().run_simulation(
            t_end, IC_full, dt, n_out, t_start, progress=progress,
            _on_step=on_step
        )

        # u is piecewise constant. Pad the last point so u_full sits on
        # the same grid as x_full, then decimate with the same indices
        # used for the states -- exact, no interpolation.
        u_arr = np.asarray(self._u_pid, dtype=float)
        self.u_full = np.append(u_arr, u_arr[-1])
        self.u = self.u_full[self._out_idx]

        if self._valve is not None:
            # The wrapper prepended the valve state for the integrator.
            # Report it as mv and drop it from the state arrays, so the
            # states the student sees are the PROCESS states with the
            # same indexing as their open-loop model.
            self.mv_full = self.x_full[0]
            self.mv = self.x[0]
            self.x_full = self.x_full[1:]
            self.x = self.x[1:]
        else:
            # no valve: the manipulated variable IS the held actuator
            # input -- the controller output plus feedforward, if any
            # (identical to u when there is no feedforward).
            mv_arr = np.asarray(self._zoh._u, dtype=float)
            mv_full = np.append(mv_arr, mv_arr[-1])
            self.mv_full, self.mv = mv_full, mv_full[self._out_idx]
        return self.t, self.x, self.v
