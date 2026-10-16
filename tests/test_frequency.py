"""
tests/test_frequency.py  --  Bode / frequency analysis by the sine test

Numbered checks, asserts + prints, in the style of test_closedloop.py.
Run:  PYTHONPATH=src python tests/test_frequency.py

The whole module SIMULATES the lab experiment: drive a ClosedLoop built
with source= (no controller) with A*sin(w t) at each w, let the transient
settle, and fit the amplitude ratio and phase by correlating the measured
(sensored) output with sin/cos over the last few periods. No Laplace, no
transfer-function machinery -- the same System/RK4 the rest of the
package uses.
"""
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless: bode() must render without a display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import cme452
from cme452 import ClosedLoop, frequency

# A first-order process, written open-loop (mv is inputs[0]).
def plant(t, x, p, inputs):
    return [inputs[0](t) - x[0]]

# The open chain to measure: source -> (valve) -> process -> sensor.
# Built with source= (no controller). A placeholder (1.0, 1.0) is fine;
# frequency.response reassigns loop.source per frequency.
def make_plant(valve=None, valve0=None, sensor=1.0):
    return ClosedLoop(plant, {}, measure=0, source=(1.0, 1.0),
                      valve=valve, valve0=valve0, sensor=sensor)


def check(label, ok, extra=""):
    print(("OK  " if ok else "FAIL") + f" {label}" + (f"  {extra}" if extra else ""))
    if not ok:
        raise AssertionError(label)


# ======================================================================
# 1. first-order process: gain and phase match the analytic 1/(1 + s)
# ======================================================================
print("1. first-order process matches 1/(1+s)")
p = make_plant()
w = np.logspace(-1, 1, 25)
w_, gain, phase = frequency.response(p, IC_ss=[1.0], omegas=w,
                                     settle_periods=3, fit_periods=5, dt=0.01)
g_ana = 1.0 / np.sqrt(1.0 + w**2)
ph_ana = -np.arctan(w)
rel_g = float(np.max(np.abs(gain - g_ana) / g_ana))
abs_p = float(np.max(np.abs(phase - ph_ana)))
check("first-order gain/phase",
      rel_g < 0.02 and abs_p < 0.10,
      f"max rel gain err {rel_g:.1e}, max phase err {np.degrees(abs_p):.2f} deg")
# the returned omegas are in the SAME order as the input
check("omegas order preserved", np.array_equal(w_, w))

# ======================================================================
# 2. the measured chain includes the valve and the sensor
#    G(s) = Km * Kv/(1 + tauv s) * 1/(1 + s)
# ======================================================================
print("2. valve + sensor chain is the product of the parts")
Km, Kv, tauv = 3.0, 2.0, 0.5
p2 = make_plant(valve=(Kv, tauv), valve0=0.0, sensor=Km)
# keep w*dt <= ~0.5 so the ZOH-sampled sine is well resolved (dt = 0.01)
w2 = np.logspace(-1, 0.7, 25)
w2_, gain2, phase2 = frequency.response(p2, IC_ss=[1.0], omegas=w2,
                                        settle_periods=3, fit_periods=5, dt=0.01)
g_ana2 = Km * Kv / np.sqrt((1 + (tauv * w2)**2) * (1 + w2**2))
ph_ana2 = -(np.arctan(tauv * w2) + np.arctan(w2))
rel_g2 = float(np.max(np.abs(gain2 - g_ana2) / g_ana2))
abs_p2 = float(np.max(np.abs(phase2 - ph_ana2)))
check("valve+sensor gain/phase",
      rel_g2 < 0.03 and abs_p2 < 0.15,
      f"max rel gain err {rel_g2:.1e}, max phase err {np.degrees(abs_p2):.2f} deg")

# ======================================================================
# 3. 2nd-order process: resonance peak near wn for low damping
#    x'' + 2 zeta wn x' + wn^2 x = u  ->  G = wn^2/(s^2 + 2 zeta wn s + wn^2)
# ======================================================================
print("3. 2nd-order resonance peak")
wn, zeta = 1.0, 0.1
def plant2(t, x, p, inputs):
    # states [x, v=x']; x'' = u - 2 zeta wn v - wn^2 x
    return [x[1], inputs[0](t) - 2.0 * zeta * wn * x[1] - wn**2 * x[0]]

p3 = ClosedLoop(plant2, {}, measure=0, source=(1.0, 1.0))
w3 = np.logspace(-1, 1, 60)
w3_, gain3, phase3 = frequency.response(p3, IC_ss=[0.0, 0.0], omegas=w3,
                                        settle_periods=6, fit_periods=6, dt=0.01)
# resonant frequency wr = wn*sqrt(1-2 zeta^2); peak gain ~ 1/(2 zeta)
wr = wn * np.sqrt(max(1.0 - 2.0 * zeta**2, 0.0))
i_pk = int(np.argmax(gain3))
check("resonance near wn", abs(w3[i_pk] - wr) / wr < 0.15,
      f"peak at w={w3[i_pk]:.3f}, expected ~{wr:.3f}")
check("resonant gain ~ 1/(2 zeta)",
      abs(gain3[i_pk] - 1.0 / (2.0 * zeta)) / (1.0 / (2.0 * zeta)) < 0.25,
      f"peak gain {gain3[i_pk]:.3f}, expected ~{1.0/(2.0*zeta):.3f}")

# ======================================================================
# 4. margins on synthetic curves with known crossings
# ======================================================================
print("4. gain/phase margins")
# (a) no phase = -180 crossing: gm = inf, w_pc = None
w = np.logspace(-1, 2, 200)
gain = 10.0 / (1.0 + 0.5 * w)**2
phase = -2.0 * np.arctan(0.5 * w)
gm, pm, w_gc, w_pc = frequency.margins(gain, phase, w)
check("no phase crossing -> gm=inf, w_pc=None",
      np.isinf(gm) and w_pc is None)
check("gain crossover interpolated",
      abs(w_gc - 2.0 * (np.sqrt(10.0) - 1.0)) < 0.1, f"w_gc={w_gc:.4f}")

# (b) phase crosses -180 (cubic lag): w_pc analytic
w3 = np.logspace(-1, 2.3, 400)
g3 = 1.0 / (1.0 + 0.1 * w3)**3
p3c = -3.0 * np.arctan(0.1 * w3)
gm3, pm3, wgc3, wpc3 = frequency.margins(g3, p3c, w3)
check("phase crossing at analytic w_pc",
      wpc3 is not None and abs(wpc3 - (1.0 / 0.1) * np.tan(np.pi / 3.0)) < 0.5,
      f"w_pc={wpc3:.3f}, expected {(1/0.1)*np.tan(np.pi/3):.3f}")

# (c) error: mismatched shapes
try:
    frequency.margins(np.array([1.0, 2.0]), np.array([0.0]), np.array([1.0, 2.0]))
    check("margins shape error", False)
except cme452.ModelError:
    check("margins shape error", True)

# ======================================================================
# 5. bode() renders (headless) and returns the numbers + a figure
# ======================================================================
print("5. bode() renders headless")
p = make_plant(valve=(1.0, 0.5), valve0=0.0, sensor=1.0)
w5 = np.logspace(-1, 1, 20)
out = frequency.bode(p, IC_ss=[1.0], omegas=w5,
                     settle_periods=3, fit_periods=5, dt=0.01)
w5_, gain5, phase5, fig = out
check("bode returns (omegas, gain, phase, fig)",
      isinstance(fig, plt.Figure) and len(out) == 4)
check("bode has 2 axes (gain dB / phase deg)", len(fig.axes) == 2)
_w6, g6, p6 = frequency.response(p, [1.0], w5,
                                  settle_periods=3, fit_periods=5, dt=0.01)
check("bode gain/phase match response()",
      np.allclose(gain5, g6) and np.allclose(phase5, p6))
plt.close(fig)

# ======================================================================
# 6. source-mode validation: a controller loop is rejected cleanly
# ======================================================================
print("6. a controller loop is rejected")
from cme452 import PID
bad = ClosedLoop(plant, {}, PID(1.0, 4.0), measure=0, y_sp=1.0)
try:
    frequency.response(bad, IC_ss=[1.0], omegas=[1.0])
    check("rejects controller loop", False)
except cme452.ModelError as e:
    check("rejects controller loop", "source=" in str(e))
# non-ClosedLoop rejected
try:
    frequency.response(plant, IC_ss=[1.0], omegas=[1.0])
    check("rejects non-ClosedLoop", False)
except cme452.ModelError:
    check("rejects non-ClosedLoop", True)
# bad omegas
p = make_plant()
try:
    frequency.response(p, [1.0], [0.0, -1.0])
    check("rejects nonpositive omegas", False)
except cme452.ModelError:
    check("rejects nonpositive omegas", True)

print()
print("ALL FREQUENCY TESTS PASSED (6 checks)")
sys.exit(0)
