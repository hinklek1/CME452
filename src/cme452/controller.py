"""
PID  --  discrete-time controller for use with ClosedLoop
=========================================================

    pid = PID(Kc)               P
    pid = PID(Kc, tauI)         PI
    pid = PID(Kc, tauI, tauD)   PID

A PID is meant to be handed to a ClosedLoop, which calls it once per
integration step with the current time, the measured value, and the
setpoint:

    u = pid.update(t, y, y_sp)

The law implemented is

    e  = y_sp - y
    u  = Kc*e + (Kc/tauI)*I  -  Kc*tauD*(dy/dt),      dI/dt = e

Design notes
------------
* The error sign is fixed here (e = y_sp - y), so a student cannot
  silently build a negative-feedback loop.
* The I term is integrated with the trapezoidal rule.
* The derivative acts on the MEASUREMENT, not the error. A step in the
  setpoint therefore produces no derivative spike ("derivative kick").
* With umin/umax given, the output is clamped and the I term stops
  accumulating while the output is saturated (conditional integration --
  the simplest form of anti-windup).  Set antiwindup=False to disable
  this and watch classical windup: the I term keeps growing while the
  output is pinned, so the controller keeps demanding a saturated move
  long after the setpoint has been reached.
* reset() clears all internal memory. ClosedLoop calls it automatically
  at the start of every run_simulation, so re-running a simulation can
  never leak an old I term into the new one.
"""

import numpy as np


class PID:
    """Discrete PI(D) controller. See the module docstring for the law."""

    def __init__(self, Kc, tauI=None, tauD=None, umin=None, umax=None,
                 antiwindup=True):
        Kc = float(Kc)
        if not np.isfinite(Kc) or Kc == 0.0:
            raise ValueError(f"Kc must be a nonzero finite number; you gave {Kc}.")
        if tauI is not None and float(tauI) <= 0:
            raise ValueError(f"tauI must be positive; you gave {tauI}.")
        if tauD is not None and float(tauD) < 0:
            raise ValueError(f"tauD must be nonnegative; you gave {tauD}.")
        if umin is not None and umax is not None and umin >= umax:
            raise ValueError(f"umin ({umin}) must be less than umax ({umax}).")

        self.Kc = Kc
        self.tauI = float(tauI) if tauI is not None else None
        self.tauD = float(tauD) if tauD is not None else None
        self.umin = None if umin is None else float(umin)
        self.umax = None if umax is None else float(umax)
        self.antiwindup = bool(antiwindup)
        self.reset()

    # ------------------------------------------------------------------
    @property
    def mode(self):
        """'P', 'PI', 'PD', or 'PID' -- whatever was actually given."""
        m = "P"
        if self.tauI is not None:
            m += "I"
        if self.tauD is not None:
            m += "D"
        return m

    def reset(self):
        """Clear all internal memory (I term, previous samples)."""
        self._I = 0.0
        self._t = None
        self._y = None
        self._e = None

    # ------------------------------------------------------------------
    def update(self, t, y, y_sp):
        """Advance the controller one sample and return the control move.

        t    : current time
        y    : measured process value
        y_sp : setpoint

        Must be called with nondecreasing t, once per control interval.
        """
        y = float(y)
        e = float(y_sp) - y

        if self._t is None:
            dt = 0.0
        else:
            dt = t - self._t
            if dt < 0:
                raise ValueError(
                    f"PID.update was called with decreasing time "
                    f"({self._t:g} followed by {t:g}). Time must move forward."
                )

        # trapezoidal accumulation of the error
        dI = 0.5 * (e + self._e) * dt if (dt > 0 and self._e is not None) else 0.0
        # derivative on the measurement (no setpoint kick)
        dydt = (y - self._y) / dt if (dt > 0 and self._y is not None) else 0.0

        I = self._I + dI
        u = self.Kc * e
        if self.tauI is not None:
            u += (self.Kc / self.tauI) * I
        if self.tauD is not None:
            u -= self.Kc * self.tauD * dydt

        # saturation; with anti-windup (conditional integration), the I
        # increment is undone while the output is clamped, so the I term
        # cannot grow while the move is pinned
        if self.umax is not None and u > self.umax:
            u = self.umax
            if self.antiwindup:
                I = self._I
        elif self.umin is not None and u < self.umin:
            u = self.umin
            if self.antiwindup:
                I = self._I

        self._I, self._t, self._y, self._e = I, t, y, e
        return float(u)

    def __repr__(self):
        s = f"PID(Kc={self.Kc:g}"
        if self.tauI is not None:
            s += f", tauI={self.tauI:g}"
        if self.tauD is not None:
            s += f", tauD={self.tauD:g}"
        if self.umin is not None:
            s += f", umin={self.umin:g}"
        if self.umax is not None:
            s += f", umax={self.umax:g}"
        if not self.antiwindup:
            s += ", antiwindup=False"
        return s + ")"
