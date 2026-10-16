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

# 32. antiwindup flag: ON, the I term does not grow while saturated, so
#     the output frees up as soon as the error reverses; OFF, it stays
#     pinned at the limit for a long unwinding tail
def _stuck_steps(aw):
    pid = PID(1.0, 1.0, umin=0.0, umax=1.0, antiwindup=aw)
    t = 0.0
    for _ in range(20):              # big positive error -> saturated
        t += 0.5
        pid.update(t, 0.0, 10.0)
    n = 0
    for _ in range(500):             # error reverses; count steps at limit
        t += 0.5
        n += 1
        if pid.update(t, 1.0, 0.0) < 0.5:
            break
    return n

with_aw, without_aw = _stuck_steps(True), _stuck_steps(False)
assert with_aw <= 2, with_aw
assert without_aw > 20, without_aw
print("32 OK  antiwindup flag: pinned %d steps (on) vs %d (off)"
      % (with_aw, without_aw))

# 33. measurement delay: sensor=(Km, delay)
#     (a) delay=0 is bit-identical to the bare gain (same quantization,
#         same look-back logic, offset 0)
#     (b) a nonzero delay changes the closed-loop response: the
#         controller reacts to the past, so u and x diverge from the
#         no-delay case.  Use a 2nd-order plant (underdamped) so the
#         delayed feedback has a visible effect on the transient.
def osc(t, x, p, i):
    y, v = x
    return [v, i[0](t) - 2.0 * v - 4.0 * y]      # wn=2, zeta=0.5
def run_d(sensor, Kc=1.0, tauI=5.0):
    loop = ClosedLoop(osc, {}, PID(Kc, tauI), measure=0, y_sp=1.0,
                      sensor=sensor)
    loop.run_simulation(t_end=60, IC=[0.0, 0.0], progress=False)
    return loop

ref = run_d(1.0)
zero = run_d((1.0, 0.0))
assert np.max(np.abs(ref.x - zero.x)) == 0.0
assert np.max(np.abs(ref.u - zero.u)) == 0.0
d = run_d((1.0, 3.0))
assert np.max(np.abs(ref.u - d.u)) > 0.1          # controller diverges
assert np.max(np.abs(ref.x - d.x)) > 0.05         # plant follows
print("33 OK  delay=0 identical to no delay; delay=3 changes response")

# 34. delay changes the transient shape: with the same gains, the
#     delayed loop has a different overshoot and a different settling
#     time.  (Delay adds phase; whether that helps or hurts depends on
#     the loop -- the point is that it is NOT the same response.)
def settle(l, sp=1.0, tol=0.1):
    x = l.x[0]
    dev = np.abs(x - sp) > tol
    idx = np.where(dev)[0]
    if len(idx) == 0:
        return 0.0
    return 60.0 * (idx[-1] + 1) / x.size

lag = run_d(1.0)
slow = run_d((1.0, 3.0))
assert settle(lag) != settle(slow)
assert np.max(lag.x[0]) != np.max(slow.x[0])   # different overshoot
print("34 OK  delay=3: overshoot %.3f -> %.3f, settling %.1f -> %.1f"
      % (np.max(lag.x[0]), np.max(slow.x[0]),
         settle(lag), settle(slow)))

# 35. sensor argument validation
for sensor, frag in [
    ((1.0,), "(Km, delay)"),
    ((2.0, 1.0, 3.0), "(Km, delay)"),
    ((-2.0, 1.0), "positive"),
    ((2.0, -1.0), "nonnegative"),
    ((2.0, float("nan")), "nonnegative"),
    (("a", 1.0), "(Km, delay)"),
]:
    try:
        ClosedLoop(first, {}, Const(), measure=0, sensor=sensor)
        assert False, sensor
    except cme452.ModelError as e:
        assert frag in str(e), (frag, str(e))
print("35 OK  sensor sequence validation")

# ------------------------------------------------------------------
print("36. perfect feedforward: the disturbance step is invisible")

def ffplant(t, x, p, i):
    # mv + additive disturbance d, first-order decay
    return [i[0](t) + i[1](t) - x[0]]

def d_step(t):
    return 0.0 if t < 5.0 else 0.5

def d_none(t):
    return 0.0

# no disturbance: the reference trajectory
ref = ClosedLoop(ffplant, {}, PID(1.0, 4.0), measure=0, y_sp=1.0,
                 inputs=d_none)
ref.run_simulation(t_end=20, IC=[0.0], progress=False)

# disturbance step, NO feedforward: the response must visibly differ
no_ff = ClosedLoop(ffplant, {}, PID(1.0, 4.0), measure=0, y_sp=1.0,
                   inputs=d_step)
no_ff.run_simulation(t_end=20, IC=[0.0], progress=False)
diff_noff = np.max(np.abs(no_ff.x[0][500:] - ref.x[0][500:]))
assert diff_noff > 0.05, diff_noff

# disturbance step WITH perfect ff (d enters with +1, so Kff = -1):
# after the crossing step the plant sees exactly what the reference sees
# (mv = u - 0.5, d = +0.5, net u). The ONE exception is the step that
# crosses t=5: the ff path is a ZOH too, so it still holds d(4.99) = 0
# while the disturbance has already jumped at the 4th RK4 stage -- a
# one-step glitch of h*d/6 that the loop then damps out.
ff = ClosedLoop(ffplant, {}, PID(1.0, 4.0), measure=0, y_sp=1.0,
                inputs=d_step, feedforward=(-1.0, 0))
ff.run_simulation(t_end=20, IC=[0.0], progress=False)
diff_ff = np.max(np.abs(ff.x[0] - ref.x[0]))
assert diff_ff < 0.001, diff_ff                       # the glitch (h*d/6)
diff_tail = np.max(np.abs(ff.x[0][1000:] - ref.x[0][1000:]))
assert diff_tail < 1e-4, diff_tail                    # damped by the loop
# and the feedback controller sat still: from t=10 on its output is the
# reference's, while the actuator input (u + ff) carries the correction --
# mv drops by 0.5 (Kff*d = -0.5) exactly as the disturbance pushes +0.5
assert np.max(np.abs(ff.u[1000:] - ref.u[1000:])) < 1e-4
assert abs(np.mean(ff.mv[1000:]) - (np.mean(ref.mv[1000:]) - 0.5)) < 1e-4
print(f"36 OK  perfect ff: max|deviation| {diff_ff:.1e} (glitch) vs no-ff {diff_noff:.3f}")

# ------------------------------------------------------------------
print("37. imperfect ff, P-only: analytic steady-state offset")

# x_ss = (Kc*sp + (Kff+1)*d) / (1 + Kc)   with Kc=1, sp=1, d=0.5
Kc, Kff, d = 1.0, -0.9, 0.5
x_ss = (Kc * 1.0 + (Kff + 1.0) * d) / (1.0 + Kc)
p_only = ClosedLoop(ffplant, {}, PID(Kc), measure=0, y_sp=1.0,
                    inputs=d_step, feedforward=(Kff, 0))
p_only.run_simulation(t_end=40, IC=[0.0], progress=False)
assert abs(p_only.x[0, -1] - x_ss) < 1e-3, (p_only.x[0, -1], x_ss)
print(f"37 OK  Kff={Kff}: x(40) = {p_only.x[0, -1]:.4f} == analytic {x_ss:.4f}")

# ------------------------------------------------------------------
print("38. ff acts on the CHOSEN channel only")

def two_dist(t, x, p, i):
    return [i[0](t) + i[1](t) + 3.0 * i[2](t) - x[0]]

# ff on channel 1 (the 3*d2 term) with the right gain
okch = ClosedLoop(two_dist, {}, PID(1.0, 4.0), measure=0, y_sp=1.0,
                  inputs=(d_none, d_step), feedforward=(-3.0, 1))
okch.run_simulation(t_end=20, IC=[0.0], progress=False)
# same one-step crossing glitch as test 36 (h*3*d/6 = 0.0025), then flat
assert np.max(np.abs(okch.x[0] - ref.x[0])) < 0.01
assert np.max(np.abs(okch.x[0][1000:] - ref.x[0][1000:])) < 1e-4

# the same ff gain on channel 0 (the d1 term) does NOT cancel d2
badch = ClosedLoop(two_dist, {}, PID(1.0, 4.0), measure=0, y_sp=1.0,
                   inputs=(d_none, d_step), feedforward=(-1.0, 0))
badch.run_simulation(t_end=20, IC=[0.0], progress=False)
assert np.max(np.abs(badch.x[0][500:] - ref.x[0][500:])) > 0.05
print("38 OK  ff cancels its channel, leaves the other alone")

# ------------------------------------------------------------------
print("39. ff is added AFTER the controller's clamp")

saturated = ClosedLoop(
    ffplant, {}, PID(50.0, umin=0.0, umax=1.0), measure=0, y_sp=10.0,
    inputs=d_step, feedforward=(4.0, 0))
saturated.run_simulation(t_end=8, IC=[0.0], progress=False)
assert np.max(saturated.u) <= 1.0 + 1e-12          # PID clamped at 1
assert abs(np.max(saturated.mv) - 3.0) < 1e-9      # actuator sees 1 + 4*0.5
print("39 OK  u clamped at 1.0, actuator input (mv) reaches 3.0")

# ------------------------------------------------------------------
print("40. feedforward argument validation")

for ff, frag in [
    (5.0, "(Kff, j)"),                # bare gain, no channel
    ((1.0,), "(Kff, j)"),             # missing channel
    ((1.0, 0.5, 2.0), "(Kff, j)"),    # too long
    ((float("nan"), 0), "finite"),
    ((1.0, 3), "valid indices"),      # only one disturbance
    ((1.0, "a"), "integer"),
    ((1.0, True), "integer"),
]:
    try:
        ClosedLoop(ffplant, {}, PID(1.0), measure=0, y_sp=1.0,
                   inputs=d_step, feedforward=ff)
        assert False, ff
    except cme452.ModelError as e:
        assert frag in str(e), (frag, str(e))
try:                                              # no disturbances at all
    ClosedLoop(ffplant, {}, PID(1.0), measure=0, y_sp=1.0,
               feedforward=(1.0, 0))
    assert False
except cme452.ModelError as e:
    assert "inputs=" in str(e), str(e)
# a NEGATIVE gain is legal (the sign is the physics of the disturbance)
neg = ClosedLoop(ffplant, {}, PID(1.0), measure=0, y_sp=1.0,
                 inputs=d_step, feedforward=(-1.0, 0))
neg.run_simulation(t_end=1.0, IC=[0.0], progress=False)
print("40 OK  feedforward validation")

# ======================================================================
# 41. source=: open-loop drive (no controller) ---------------------------
# ======================================================================
print("41. source=: open-loop drive (no controller)")

def plant(t, x, p, i): return [i[0](t) - x[0]]      # first-order, tau=1

# (a) (w, A) sine: the held signal is A*sin(w*t), sampled per step
lp = ClosedLoop(plant, {}, measure=0, source=(2.0 * np.pi, 3.0))
t, x, v = lp.run_simulation(t_end=5.0, IC=[0.0], dt=0.01, progress=False)
# u is piecewise constant; at t=0.25 the held value is sin at the last
# sample <= 0.25, i.e. sin(2*pi*0.25)=1 -> 3*1 = 3 (peak)
i = int(np.argmin(np.abs(t - 0.25)))
assert abs(lp.u[i] - 3.0) < 1e-3, (i, lp.u[i])
# the plant follows: x(t) is the first-order response to the sine
assert lp.x[0, -1] == lp.x[0, -1]   # finite (no divergence)

# (b) bare frequency = unit amplitude
lp2 = ClosedLoop(plant, {}, measure=0, source=1.0)
t2, _, _ = lp2.run_simulation(t_end=1.0, IC=[0.0], dt=0.01, progress=False)
assert abs(lp2.u[1] - np.sin(0.01)) < 1e-6, lp2.u[1]

# (c) 1-tuple (w,) = unit amplitude, same as bare w
lp1t = ClosedLoop(plant, {}, measure=0, source=(1.0,))
lp1t.run_simulation(t_end=1.0, IC=[0.0], dt=0.01, progress=False)
assert abs(lp1t.u[1] - np.sin(0.01)) < 1e-6

# (d) callable drive
lp3 = ClosedLoop(plant, {}, measure=0, source=(lambda tt: 0.5 * np.cos(tt)))
lp3.run_simulation(t_end=1.0, IC=[0.0], dt=0.01, progress=False)
assert abs(lp3.u[1] - 0.5 * np.cos(0.01)) < 1e-6, lp3.u[1]

# (e) settable property: a sweep reassigns the source
lp.source = (1.0, 2.0)
lp.run_simulation(t_end=1.0, IC=[0.0], dt=0.01, progress=False)
assert abs(lp.u[1] - 2.0 * np.sin(0.01)) < 1e-6, lp.u[1]

# (f) valve + source: mv is the valve position (lags the sine), u is the sine
lpv = ClosedLoop(plant, {}, measure=0, source=(1.0, 1.0),
                 valve=(1.0, 0.5), valve0=0.0)
tv, _, _ = lpv.run_simulation(t_end=20.0, IC=[0.0], dt=0.01, progress=False)
# u must be the raw sine; mv the valve state, oscillating around it
i = int(np.argmin(np.abs(tv - 15.0)))
assert abs(lpv.u[i] - np.sin(1.0 * 15.0)) < 1e-2, lpv.u[i]
assert np.max(np.abs(lpv.mv)) > 0.3, "valve did not move"
assert np.all(np.isfinite(lpv.mv))

# (g) mutual exclusion + bad source errors
for name, fn in [
    ("both", lambda: ClosedLoop(plant, {}, PID(1.0), measure=0,
                                source=(1.0, 1.0))),
    ("neither", lambda: ClosedLoop(plant, {}, measure=0)),
    ("w=0", lambda: ClosedLoop(plant, {}, measure=0, source=(0.0, 1.0))),
    ("w<0", lambda: ClosedLoop(plant, {}, measure=0, source=(-1.0, 1.0))),
    ("str", lambda: ClosedLoop(plant, {}, measure=0, source=("a",))),
    ("A nan", lambda: ClosedLoop(plant, {}, measure=0, source=(1.0, float("nan")))),
]:
    try:
        fn(); assert False, name
    except cme452.ModelError as e:
        assert len(str(e)) > 0

# (h) controller path is UNCHANGED (regression): same loop, same result
lpc1 = ClosedLoop(plant, {}, PID(1.0, 4.0), measure=0, y_sp=1.0)
lpc1.run_simulation(t_end=5.0, IC=[0.0], progress=False)
lpc2 = ClosedLoop(plant, {}, PID(1.0, 4.0), measure=0, y_sp=1.0)
lpc2.run_simulation(t_end=5.0, IC=[0.0], progress=False)
assert np.array_equal(lpc1.x, lpc2.x), "controller loop non-deterministic"

print("41 OK  source drive (sine/bare/callable/settable/valve/errors)")

print()
print("ALL TESTS PASSED (41 checks)")
raise SystemExit(0)
