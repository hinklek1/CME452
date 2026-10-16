"""
frequency  --  Bode / frequency analysis by the lab experiment
==============================================================

Instead of building transfer-function machinery (state-space -> tf,
freqresp), this module SIMULATES the experiment you would run in lab:

    for each frequency w:
        start from the steady-state operating point IC_ss,
        drive the manipulated variable with A*sin(w t) via ClosedLoop(source=),
        run long enough for the transient to decay,
        fit the amplitude ratio and phase by correlating the measured
        output with sin(w t) and cos(w t) over the last few periods.

That is literally what the sine test is, now in a loop. It reuses
System/ClosedLoop unchanged -- no Laplace anywhere -- and the operating
point is the same one stability.jacobian linearizes at, so the two
features describe the same local behavior.

The object you pass is a ClosedLoop built with source= (no controller):

    plant = ClosedLoop(model, params, measure=0, source=(1.0, 1.0),
                       valve=(Kv, tauv), valve0=q0, sensor=Km)
    w, gain, phase = frequency.response(plant, IC_ss, w_grid)

The chain measured is  source -> valve -> process -> y -> sensor, i.e.
the open loop the controller actually closes on.

GAIN and PHASE
--------------
The measured output y_s(t) is decomposed against the drive at the same
frequency:

    y_s(t) ~= G * A * sin(w t + phi)
           = G*A*[cos(phi) sin(wt) + sin(phi) cos(wt)]

Correlating with sin(wt) and cos(wt) over the fit window gives the
in-phase (a) and quadrature (b) components:

    a = <y_s * sin> / <sin^2>,   b = <y_s * cos> / <cos^2>

    gain  = sqrt(a^2 + b^2) / A
    phase = atan2(b, a)          (radians, +90 deg max lead)

Because the fit divides out the drive amplitude, the result is the
small-signal (linearized) frequency response: it does not depend on A.

SETTLE / FIT
------------
Each w runs for (settle_periods + fit_periods) periods. The transient
from IC_ss decays during the settle window and is thrown away; the fit
uses only the last fit_periods periods. Both are explicit arguments so a
student trades accuracy for speed and sees the transient they discard.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .core import ModelError


# ---------------------------------------------------------------------
#  internals
# ---------------------------------------------------------------------

def _validate_source_loop(plant):
    """Return True if `plant` is a ClosedLoop in source mode; raise a
    clean error otherwise (the common mistake: forgetting source=)."""
    from .loop import ClosedLoop
    if not isinstance(plant, ClosedLoop):
        raise ModelError(
            "frequency.response expects a ClosedLoop built with source= "
            f"(an open-loop drive), not a {type(plant).__name__}."
        )
    if plant._source is None:
        raise ModelError(
            "that ClosedLoop was built with a controller, not source=. "
            "Rebuild it as ClosedLoop(model, params, measure=0, source=(w, A)) "
            "so the manipulated variable is prescribed instead of controlled."
        )
    return True


def _fit_one(plant, IC_ss, w, settle_periods, fit_periods, dt,
             settle_time=None, dt_max=0.05):
    """Run one frequency and return (gain, phase) for that w.

    The fit window is `fit_periods` whole periods (whole periods => no
    half-period leakage into the sin/cos correlation). The settle window
    is `settle_time` (a fixed TIME duration, default ~6 x the slowest time
    constant) if given, else `settle_periods` periods. Settling by a fixed
    process time instead of by sine periods avoids running 10x longer than
    needed for a fast process at low w.

    The integration step is min(dt, dt_max, 0.2/w) so the sine is never
    under-resolved (w*dt <= 0.2) and low frequencies use a coarser, faster
    step. dt_max caps the step so it never outruns the process dynamics.
    """
    fit = max(1, int(round(fit_periods)))
    # per-frequency step: the LARGEST step that (a) still resolves the sine
    # (w*h <= 0.2) and (b) still resolves the process dynamics (h <= dt_max).
    # This makes low frequencies fast (coarse step) and high frequencies
    # accurate (fine step) automatically. If the caller passed an explicit
    # dt, it is a further ceiling (a small dt forces a finer, slower step);
    # dt=None (the default) means "auto" -- use only dt_max and 0.2/w.
    caps = [float(dt_max), 0.2 / w]
    if dt is not None:
        caps.append(float(dt))
    h = min(caps)
    h = max(h, 1e-6)                      # guard against w -> inf
    # settle duration in TIME units
    if settle_time is None:
        settle_time = max(1, int(round(settle_periods))) * (2.0 * np.pi / w)
    settle_time = float(settle_time)
    fit_time = fit * (2.0 * np.pi / w)
    t_end = settle_time + fit_time
    # n_fit in POINTS for the fit window: fit periods worth of samples
    n_fit = max(2, int(round(fit_time / h)))
    # re-point the source at this frequency (unit amplitude internally;
    # the fit divides it out, so A does not change the answer)
    plant.source = (w, 1.0)
    # n_out=None keeps EVERY integration step: the fit correlates the
    # output with sin/cos on a dense grid, so the output must not be
    # decimated (decimating to ~n points would alias the w-sine).
    t, x, v = plant.run_simulation(
        t_end=t_end, IC=IC_ss, dt=h, n_out=None, progress=False
    )
    # measured output on the output grid, in SENSOR units (the chain the
    # controller closes on: process -> valve -> sensor). x_full is every
    # integration step of the PROCESS state; for a valve loop, measure
    # expects the WRAPPED state [valve, process...], so prepend mv_full.
    # Then apply the sensor via _transduce (gain Km, or a custom fn).
    xf = plant.x_full
    if plant._valve is not None:
        xf = np.vstack([plant.mv_full, xf])
    y = np.empty(xf.shape[1])
    for j in range(xf.shape[1]):
        y_raw = float(np.asarray(plant.measure(xf[:, j]),
                                  dtype=float).ravel()[0])
        # _transduce returns (sensor_reading, setpoint); we want the reading
        y[j], _ = plant._transduce(y_raw, t[j] if t is not None else 0.0)
    y = y.ravel()

    # fit over the LAST `fit` periods only (the settle window is dropped)
    t_f = t[-n_fit:]
    y_f = y[-n_fit:]

    # Remove the DC level (the operating point) before correlating: the fit
    # measures the AC response to the AC drive, not the steady value they
    # are superposed on. <sin> and <cos> over whole periods are 0, so
    # detrending does not bias the in-phase / quadrature components.
    y_f = y_f - float(np.mean(y_f))

    # Correlate the response with sin(w t) and cos(w t) over the fit window.
    # y_f ~= A_g*sin(w t + phi) = A_g*[cos(phi) sin + sin(phi) cos], so the
    # two correlations give the in-phase (a) and quadrature (b) components.
    s = np.sin(w * t_f)
    c = np.cos(w * t_f)
    s2 = float(np.mean(s * s))
    c2 = float(np.mean(c * c))
    a = float(np.mean(y_f * s)) / s2      # in-phase component
    b = float(np.mean(y_f * c)) / c2      # quadrature component
    gain = float(np.hypot(a, b))          # amplitude ratio (drive A = 1)
    phase = float(np.arctan2(b, a))       # radians (+ = lead)
    return gain, phase


# ---------------------------------------------------------------------
#  public API
# ---------------------------------------------------------------------

def response(plant, IC_ss, omegas, settle_periods=2.0, fit_periods=3.0,
             dt=None, settle_time=None, dt_max=0.05):
    """Simulate a sine test at each frequency and return the frequency
    response of the open chain (source -> valve -> process -> sensor).

    Parameters
    ----------
    plant : ClosedLoop
        Built with source= (no controller). The chain measured is
        whatever valve= / sensor= / measure= you gave it.
    IC_ss : array_like
        The steady-state initial condition (the operating point). Run the
        DC case once to find it, or reuse the one from
        stability.jacobian -- the same local behavior is being measured.
    omegas : array_like
        Angular frequencies (rad/time), any order; a log-spaced grid of
        ~15-20 points makes a good Bode picture.
    settle_periods : float, optional
        How many periods to run before the fit so the transient from
        IC_ss decays. Default 2. Ignored when settle_time is given.
    fit_periods : float, optional
        How many periods to correlate over. Default 3.
    dt : float or None, optional
        Integration-step ceiling. Default None ("auto"): each frequency uses
        the largest safe step, min(dt_max, 0.2/w), so low frequencies are
        fast and high frequencies stay resolved. Pass a small dt (e.g.
        0.01) to force a finer, slower, more accurate step at every w.
    settle_time : float or None, optional
        If given, the settle window is this many TIME units (a fixed
        duration), not `settle_periods` periods. Set it to ~5-10 x the
        slowest process time constant to settle the transient without
        over-running. Default None -> use `settle_periods`.
    dt_max : float, optional
        Upper bound on the per-frequency integration step (default 0.05).
        Keeps the step fine enough to resolve fast transients even at low
        w, while still letting low frequencies use a coarser, faster step.

    Returns
    -------
    (omegas, gain, phase)
        Arrays, in the order of `omegas`. gain is the amplitude ratio
        (|G|), phase is the phase in RADIANS (positive = lead).
    """
    _validate_source_loop(plant)
    IC_ss = np.atleast_1d(np.asarray(IC_ss, dtype=float)).ravel()
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float)).ravel()
    if np.any(omegas <= 0) or not np.all(np.isfinite(omegas)):
        raise ModelError(
            f"omegas must be positive and finite (got {omegas.tolist()})."
        )
    if settle_periods <= 0 or fit_periods <= 0:
        raise ModelError(
            "settle_periods and fit_periods must be positive."
        )
    if settle_time is not None and settle_time <= 0:
        raise ModelError("settle_time must be positive when given.")
    gain = np.empty_like(omegas)
    phase = np.empty_like(omegas)
    for k, w in enumerate(omegas):
        gain[k], phase[k] = _fit_one(
            plant, IC_ss, float(w), settle_periods, fit_periods, dt,
            settle_time=settle_time, dt_max=dt_max,
        )
    return omegas, gain, phase


def bode(plant, IC_ss, omegas, settle_periods=2.0, fit_periods=3.0,
         dt=None, settle_time=None, dt_max=0.05, ax=None, **kwargs):
    """Simulate the sine test and draw the two-panel Bode plot
    (gain in dB, phase in degrees) on a log-frequency axis.

    Returns (omegas, gain, phase, figure) so the numbers are available
    after plotting. All keyword arguments of response are accepted, plus
    matplotlib kwargs forwarded to the two curves.
    """
    omegas, gain, phase = response(
        plant, IC_ss, omegas,
        settle_periods=settle_periods, fit_periods=fit_periods, dt=dt,
        settle_time=settle_time, dt_max=dt_max,
    )
    if ax is None:
        fig, ax = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6),
            gridspec_kw={"height_ratios": [1, 1]},
        )
    else:
        fig = ax.figure
    gain_db = 20.0 * np.log10(np.maximum(gain, 1e-12))
    phase_deg = np.degrees(phase)

    ax[0].semilogx(omegas, gain_db, "b-", **kwargs)
    ax[0].axhline(0.0, color="0.7", lw=0.8, ls="--")
    ax[0].set_ylabel("Gain (dB)")
    ax[0].set_title("Bode plot")
    ax[0].grid(True, which="both", alpha=0.3)

    ax[1].semilogx(omegas, phase_deg, "r-", **kwargs)
    ax[1].axhline(-180.0, color="0.7", lw=0.8, ls="--")
    ax[1].set_ylabel("Phase (deg)")
    ax[1].set_xlabel("Frequency (rad/time)")
    ax[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return omegas, gain, phase, fig


def margins(gain, phase, omegas):
    """Read the gain and phase margins off a frequency-response curve.

    Parameters
    ----------
    gain, phase, omegas
        The arrays from response() (any order, but paired by index).
        phase is in RADIANS.

    Returns
    -------
    (gm, pm, w_gc, w_pc)
        gm : gain margin (dB) -- how much gain can grow before the loop
             hits |G| = 1 at the phase = -180 deg crossing. inf if there
             is no -180 deg crossing.
        pm : phase margin (deg) -- how far the phase is from -180 deg at
             the |G| = 1 crossover. inf if there is no |G| = 1 crossover.
        w_gc : the |G| = 1 crossover frequency (gain crossover), or None.
        w_pc : the phase = -180 deg crossing frequency, or None.

    The crossing frequencies are interpolated linearly in log(w) between
    the two grid points that bracket each crossing.
    """
    gain = np.asarray(gain, dtype=float)
    phase = np.asarray(phase, dtype=float)
    omegas = np.asarray(omegas, dtype=float)
    if not (gain.shape == phase.shape == omegas.shape):
        raise ModelError(
            "gain, phase, and omegas must have the same shape "
            f"({gain.shape}, {phase.shape}, {omegas.shape})."
        )
    if np.any(omegas <= 0):
        raise ModelError("omegas must be positive.")

    # |G| = 1 crossover (gain crossing): where log10(gain) crosses 0
    w_gc = _crossing(omegas, np.log10(gain), 0.0)
    # phase = -180 deg crossing
    w_pc = _crossing(omegas, np.degrees(phase), -180.0)

    if w_gc is None:
        pm = np.inf
    else:
        # phase at the gain crossover, in degrees
        phi_gc = _interp(omegas, np.degrees(phase), w_gc)
        pm = float(phi_gc + 180.0)

    if w_pc is None:
        gm = np.inf
    else:
        # gain at the phase crossover, in dB
        g_pc = _interp(omegas, 20.0 * np.log10(np.maximum(gain, 1e-12)),
                       w_pc)
        gm = float(-g_pc)

    return gm, pm, w_gc, w_pc


def _crossing(omegas, values, target):
    """Linearly interpolate (in log w) the frequency where `values`
    crosses `target`. Returns None if there is no sign change."""
    d = np.asarray(values, dtype=float) - target
    for i in range(len(omegas) - 1):
        if d[i] == 0.0:
            return float(omegas[i])
        if d[i] * d[i + 1] < 0.0:
            w0, w1 = omegas[i], omegas[i + 1]
            f = d[i] / (d[i] - d[i + 1])
            return float(np.exp(np.log(w0) + f * (np.log(w1) - np.log(w0))))
    return None


def _interp(omegas, values, w):
    """Linear interpolation (in log w) of `values` at frequency w."""
    if w <= omegas[0]:
        return float(values[0])
    if w >= omegas[-1]:
        return float(values[-1])
    i = int(np.searchsorted(omegas, w)) - 1
    i = max(0, min(i, len(omegas) - 2))
    w0, w1 = omegas[i], omegas[i + 1]
    f = (np.log(w) - np.log(w0)) / (np.log(w1) - np.log(w0))
    return float(values[i] + f * (values[i + 1] - values[i]))
