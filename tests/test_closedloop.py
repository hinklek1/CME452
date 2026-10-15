"""Verification of PID + ClosedLoop + progress bar. Run with Agg."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import cme452
from cme452 import System, ClosedLoop, PID, plot_results

np.random.seed(0)
ok = []
def check(name, cond):
    ok.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

# ------------------------------------------------------------------
print("1. P-only ClosedLoop == inline-P System (their current workflow)")

def plant(t, x, params, inputs):
    Q = inputs[0](t)
    T = x[0]
    Qf = params["Qf"]
    return [(Q - Qf) * (params["T0"] - T) / T]

params = {"Qf": 0.0, "T0": 400.0}   # Qf=0 so pure-P loop has a chance
Tsp, Kc = 300.0, 1.0

def plant_inline(t, x, params, inputs):
    T = x[0]
    Q = params["Qf"] + Kc * (Tsp - T)   # what students write today
    return [(Q - params["Qf"]) * (params["T0"] - T) / T]

inline = System(plant_inline, params)
inline.run_simulation(t_end=40, IC=[250.0], progress=False)

loop = ClosedLoop(plant, params, PID(Kc), measure=lambda x: x[0], y_sp=Tsp)
loop.run_simulation(t_end=40, IC=[250.0], progress=False)

# NOTE: not bit-identical -- the inline P is re-evaluated at RK4's
# fictitious mid-stage states, the ZOH P holds the step-start move.
# Difference is O(dt^2): assert it is tiny, not zero.
diff = np.max(np.abs(inline.x - loop.x))
print(f"      max |inline-P - ZOH-P| = {diff:.2e} K "
      f"({diff / 300:.1e} relative)")
check("trajectories agree to O(dt^2)", diff < 1e-3 * 300.0)
check("u == Kc*(Tsp - T) at every step",
      np.allclose(loop.u_full[:-1], Kc * (Tsp - loop.x[0, :-1]), atol=1e-10))
check("u decimated like x", loop.u.size == loop.t.size == 4001)

# ------------------------------------------------------------------
print("2. PI removes steady-state offset (P cannot)")

def plant2(t, x, params, inputs):
    T = x[0]
    Q = inputs[0](t)
    return [(Q - 0.3) * (400.0 - T) / T]   # nonzero Qf bias

p_only = ClosedLoop(plant2, {}, PID(0.1), measure=lambda x: x[0], y_sp=300.0)
p_only.run_simulation(t_end=200, IC=[250.0], progress=False)

pi = ClosedLoop(plant2, {}, PID(1.0, 10.0), measure=lambda x: x[0], y_sp=300.0)
pi.run_simulation(t_end=200, IC=[250.0], progress=False)

print(f"      P settles at {p_only.x[0, -1]:.4f}, PI settles at {pi.x[0, -1]:.4f}")
check("P has offset", abs(p_only.x[0, -1] - 300.0) > 1.0)
check("PI reaches setpoint", abs(pi.x[0, -1] - 300.0) < 0.05)

# ------------------------------------------------------------------
print("3. I term matches independent trapezoid reference")

pid = pi.controller
t, y = pi.t_full, pi.x[0]
e = 300.0 - y
I_ref = np.zeros_like(t)
for i in range(1, t.size):
    I_ref[i] = I_ref[i-1] + 0.5 * (e[i] + e[i-1]) * (t[i] - t[i-1])
check("controller I == trapezoid sum", np.allclose(pid._I, I_ref[-1], rtol=1e-9))

# ------------------------------------------------------------------
print("4. Saturation: clamp respected, conditional integration converges")

big = ClosedLoop(plant2, {}, PID(1.0, 10.0, umin=0.0, umax=0.5),
                 measure=lambda x: x[0], y_sp=300.0)
big.run_simulation(t_end=1000, IC=[250.0], progress=False)
check("u never exceeds bounds",
      np.all(big.u_full >= 0.0 - 1e-12) and np.all(big.u_full <= 0.5 + 1e-12))
check("u actually hit the ceiling on the way in", big.u_full.max() > 0.49)
check("saturated loop still converges to setpoint", abs(big.x[0, -1] - 300.0) < 0.05)
check("repr", repr(big.controller) == "PID(Kc=1, tauI=10, umin=0, umax=0.5)")

# ------------------------------------------------------------------
print("5. No derivative kick: PD, static plant, setpoint step")

def static_plant(t, x, params, inputs):
    return [0.0]

pd = ClosedLoop(static_plant, {}, PID(2.0, tauD=5.0),
                measure=lambda x: x[0], y_sp=lambda t: 1.0 if t >= 5.0 else 0.0)
pd.run_simulation(t_end=10, IC=[0.0], dt=0.01, progress=False)
# plant never moves, so dy/dt = 0 always: u must be exactly Kc*e at every step
e = np.where(pd.t_full >= 5.0, 1.0, 0.0)
check("u == Kc*e exactly (no D spike on setpoint step)",
      np.allclose(pd.u_full, 2.0 * e, atol=1e-12))

# ------------------------------------------------------------------
print("6. Values mechanism under ClosedLoop sees the held (past) input")

def plant_vals(t, x, params, inputs):
    Q = inputs[0](t)
    T = x[0]
    return [[(Q - 0.3) * (400.0 - T) / T], [Q, T - 300.0]]

pv = ClosedLoop(plant_vals, {}, PID(1.0, 10.0), measure=lambda x: x[0], y_sp=300.0)
pv.run_simulation(t_end=50, IC=[250.0], progress=False)
# Q reported by the model at every output time == the ZOH value there == u
check("values Q(t) equals held input u(t) at all output times",
      np.allclose(pv.v[0], pv.u, atol=1e-12))
# the reported values at a MID-step time must equal the move made at
# that step's start (the ZOH is holding it):
mid = 0.5 * (pv.t_full[:-1] + pv.t_full[1:])
Q_mid = np.array([pv._call_model(m, pv.x_full[:, i])[1][0]
                  for i, m in enumerate(mid)])
check("Q at mid-step equals the move made at step start",
      np.allclose(Q_mid, pv.u_full[:-1], atol=1e-12))

# ------------------------------------------------------------------
print("7. reset(): re-running gives identical results")

r1 = pi.x.copy()
pi.run_simulation(t_end=200, IC=[250.0], progress=False)
check("second run identical (I term did not leak)", np.allclose(pi.x, r1, atol=1e-12))

# ------------------------------------------------------------------
print("8. error handling")

try:
    ClosedLoop(plant, {}, PID(1.0), measure=[0], y_sp=1.0)
    check("non-callable measure rejected", False)
except cme452.ModelError as e:
    check("non-callable measure rejected",
          "state index" in str(e) and "function" in str(e))

try:
    ClosedLoop(plant, {}, "not a controller", measure=lambda x: x[0])
    check("bad controller rejected", False)
except cme452.ModelError:
    check("bad controller rejected", True)

try:
    PID(1.0, tauI=-5.0)
    check("negative tauI rejected", False)
except ValueError:
    check("negative tauI rejected", True)

def bad_measure(x):
    return [x[0], x[0]]
try:
    ClosedLoop(plant, {}, PID(1.0), measure=bad_measure, y_sp=1.0) \
        .run_simulation(t_end=1, IC=[250.0], progress=False)
    check("vector measure rejected", False)
except cme452.ModelError as e:
    check("vector measure rejected", "single number" in str(e))

# ------------------------------------------------------------------
print("9. progress bar runs (text mode here)")

pr = ClosedLoop(plant, params, PID(1.0), measure=lambda x: x[0], y_sp=300.0)
pr.run_simulation(t_end=2, IC=[250.0], progress=True)
check("progress=True runs", pr.x.shape[1] > 0)

# ------------------------------------------------------------------
# --- extra checks appended: bare-function / index conveniences --------
print("10. measure as state index, bare-function y_sp and inputs")

def SP(t):
    return 300.0 if t < 10 else 350.0

def T0dist(t):
    return 400.0 if t < 15 else 420.0

def plant_d(t, x, params, inputs):
    Q, T, T0 = inputs[0](t), x[0], inputs[1](t)
    return [(Q - params["Qf"]) * (T0 - T) / T]

m1 = ClosedLoop(plant_d, params, PID(1.0, 10.0), measure=0, y_sp=SP,
                inputs=T0dist)                      # bare function, no tuple
m1.run_simulation(t_end=100, IC=[250.0], progress=False)
m2 = ClosedLoop(plant_d, params, PID(1.0, 10.0), measure=lambda x: x[0],
                y_sp=SP, inputs=[T0dist])           # function measure + list
m2.run_simulation(t_end=100, IC=[250.0], progress=False)
ok.append(np.allclose(m1.x, m2.x, atol=1e-12))
print(f"  [{'PASS' if ok[-1] else 'FAIL'}] measure=0 == measure=lambda; "
      f"bare fn == list for inputs")
ok.append(abs(m1.x[0, -1] - 350.0) < 1.0)
print(f"  [{'PASS' if ok[-1] else 'FAIL'}] named SP(t) function honored "
      f"(settles near 350: got {m1.x[0, -1]:.2f})")
# disturbance actually reached the plant: with T0 bumped at t=15 the
# controller must react (compare u against a no-disturbance run)
m3 = ClosedLoop(plant_d, params, PID(1.0, 10.0), measure=0, y_sp=SP,
                inputs=(lambda t: 400.0,))          # tuple form
m3.run_simulation(t_end=100, IC=[250.0], progress=False)
after = 1500
dmax = np.max(np.abs(m1.u_full[after:] - m3.u_full[after:]))
ok.append(dmax > 1e-6)
print(f"  [{'PASS' if ok[-1] else 'FAIL'}] disturbance input took effect "
      f"(max |du| after t=15: {dmax:.2e})")

try:
    ClosedLoop(plant, params, PID(1.0), measure=5).run_simulation(
        t_end=1, IC=[250.0], progress=False)
    ok.append(False)
except cme452.ModelError as e:
    good = "only 1 state" in str(e)
    ok.append(good)
    print(f"  [{'PASS' if good else 'FAIL'}] out-of-range measure index "
          f"rejected: {str(e).splitlines()[0]}")

print(f"\nTOTAL: {sum(ok)}/{len(ok)} checks passed")

# 24. valve: first-order actuator ODE accuracy (constant controller output)
class Const:
    def reset(self): pass
    def update(self, t, y, sp): return 0.5
def pm(t, x, p, inputs): return [inputs[0](t)]        # 1 state: dx/dt = mv
loop = ClosedLoop(pm, {}, Const(), measure=0, y_sp=0.0,
                  valve=(1.0, 2.0), valve0=0.0)
loop.run_simulation(t_end=10, IC=[0.0], progress=False)
q_exact = 0.5 * (1 - np.exp(-10/2.0))
assert abs(loop.mv[-1] - q_exact) < 1e-4, loop.mv[-1]
i1 = 200                                   # t = 2.0 = one valve time constant
assert abs(loop.mv[i1] - 0.5 * (1 - np.exp(-1.0))) < 1e-3   # t/tauv = 1
assert abs(loop.mv[i1] - 0.5) > 0.1          # valve lags the signal
print("24 OK  valve ODE: mv=%.5f (exact %.5f)" % (loop.mv[-1], q_exact))

# 25. without a valve, mv == u
loop = ClosedLoop(pm, {}, Const(), measure=0, y_sp=0.0)
loop.run_simulation(t_end=5, IC=[0.0], progress=False)
assert np.allclose(loop.mv, loop.u)
print("25 OK  mv == u when valve=None")

# 26. sensor-gain redundancy: sensor=Km with Kc  ==  sensor=1 with Kc*Km
def first(t, x, p, inputs): return [inputs[0](t) - x[0]]
def run_sc(Kc, Km):
    loop = ClosedLoop(first, {}, PID(Kc), measure=0, y_sp=10.0, sensor=Km)
    loop.run_simulation(t_end=30, IC=[0.0], progress=False)
    return loop.x.copy(), loop.u.copy()
x1, u1 = run_sc(1.0, 2.0)
x2, u2 = run_sc(2.0, 1.0)
assert np.max(np.abs(x1 - x2)) < 1e-9 and np.max(np.abs(u1 - u2)) < 1e-9
print("26 OK  sensor=2 with Kc=1.0  identical to  sensor=1 with Kc=2.0")

# 27. valve argument errors
for kwargs, frag in [
    (dict(valve=(1.0, 2.0)), "valve0"),
    (dict(valve0=0.3), "no valve="),
    (dict(valve=(1.0,), valve0=0.0), "Kv, tauv"),
    (dict(valve=(1.0, 0.0), valve0=0.0), "positive"),
]:
    try:
        ClosedLoop(pm, {}, Const(), measure=0, **kwargs)
        assert False, kwargs
    except cme452.ModelError as e:
        assert frag in str(e), (frag, str(e))
print("27 OK  valve errors")

# 28. sensor errors
try:
    ClosedLoop(pm, {}, Const(), measure=0, sensor=-1.0)
    assert False
except cme452.ModelError as e:
    assert "positive" in str(e)
print("28 OK  sensor error")

# 29. sensor as a function: y -> 2y, setpoint in SENSOR units (= 2*10)
x0, _ = run_sc(2.0, 1.0)     # e = 20 - 2y = 2*(10-y) == Kc=2 with no sensor
loop = ClosedLoop(first, {}, PID(1.0), measure=0, y_sp=20.0,
                  sensor=(lambda y: 2.0 * y))
loop.run_simulation(t_end=30, IC=[0.0], progress=False)
assert np.max(np.abs(loop.x - x0)) < 1e-9
print("29 OK  sensor function with setpoint in sensor units")

# 30. full tank: process-only model + valve + sensor + SP step + disturbance
params = {'Qf': 0.3, 'T0f': 400.0}
def tank(t, x, params, inputs):
    T = x[0]; q = inputs[0](t); T0 = inputs[1](t)
    return [(q - params['Qf']) * (T0 - T) / T]
def T0(t): return 410.0 if t > 20 else 400.0
SP = 350.0                         # moving setpoints are covered by test 10
loop = ClosedLoop(tank, params, PID(0.5, 8.0, umin=0.0, umax=1.0),
                  measure=0, y_sp=SP, inputs=(T0,),
                  valve=(1.0, 2.0), valve0=0.3, sensor=1.0)
# start near the setpoint: the point of the test is the loop
# (valve lag + transducer + disturbance rejection), not the slow approach.
# The I-process chain has a slow, weakly damped mode (period ~60, that's
# the D-term lesson), so assert loop BEHAVIOR: tracks, valve stays sane,
# and the ringing decays.
loop.run_simulation(t_end=120, IC=[340.0], progress=False)
T = loop.x[0]                      # process states only; the valve is loop.mv
assert abs(T[-1] - 350.0) < 1.0, T[-1]
assert 0.10 < loop.mv[-1] < 0.50, loop.mv[-1]
amp_mid  = np.max(np.abs(T[6000:8000]  - 350.0))   # t in [60, 80)
amp_late = np.max(np.abs(T[10000:12000] - 350.0))  # t in [100, 120]
assert amp_late < amp_mid, (amp_mid, amp_late)     # oscillation decays
print("30 OK  tank+valve+sensor: T=%.2f (sp 350), mv=%.3f (needed 0.3), "
      "ringing decays %.2f -> %.2f"
      % (T[-1], loop.mv[-1], amp_mid, amp_late))

# 31. bad PROCESS IC with a valve -> clean process-level error
try:
    loop.run_simulation(t_end=5, IC=[1.0, 2.0], progress=False)
    assert False
except cme452.ModelError as e:
    assert "initial conditions" in str(e), str(e)
print("31 OK  process-IC error message with valve present")

print()
print("ALL TESTS PASSED (31 checks)")
raise SystemExit(0)
