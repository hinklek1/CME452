"""Verification of stability (eigenvalue analysis). Run with Agg."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import cme452
from cme452 import System, ClosedLoop, PID, stability

# ------------------------------------------------------------------
print("1. open-loop jacobian: diagonal and rotation+decay systems")

def diag(t, x, p, i):
    return [p["a"] * x[0], p["b"] * x[1]]

J = stability.jacobian(System(diag, {"a": -2.0, "b": -0.5}), x_op=[3.0, -1.0])
ev = stability.eigenvalues(J)
assert np.allclose(np.sort(ev.real), [-2.0, -0.5], atol=1e-5), ev
assert np.max(np.abs(ev.imag)) < 1e-5

def rotdec(t, x, p, i):
    # x1' = -0.1 x1 - x2 ; x2' = x1 - 0.1 x2  ->  lambda = -0.1 +/- 1i
    return [-0.1 * x[0] - x[1], x[0] - 0.1 * x[1]]

sys2 = System(rotdec, {})
J = stability.jacobian(sys2, x_op=[1.0, 0.5], t=1.0)
ev = sorted(stability.eigenvalues(J), key=lambda z: (z.imag, z.real))
assert abs(ev[0] - (-0.1 - 1.0j)) < 1e-5, ev
assert abs(ev[1] - (-0.1 + 1.0j)) < 1e-5, ev
print("1 OK  Jacobian recovers known eigenvalues (diag, rotation+decay)")

# ------------------------------------------------------------------
print("2. predicted ringing period matches the simulated ringing")

def osc(t, x, p, i):
    # x1' = -0.05 x1 - x2 ; x2' = x1 - 0.05 x2  ->  lambda = -0.05 +/- i
    return [-0.05 * x[0] - x[1], x[0] - 0.05 * x[1]]

so = System(osc, {})
J = stability.jacobian(so, x_op=[1.0, 0.0])
lam = max(stability.eigenvalues(J), key=lambda z: abs(z.imag))
T_pred = 2 * np.pi / abs(lam.imag)        # damped period
so.run_simulation(t_end=60, IC=[1.0, 0.0], progress=False)
seg = so.x[0][1000:]                       # skip the first quarter cycle
t_seg = so.t[1000:]
cross = np.where(np.diff(np.sign(seg)) != 0)[0]
half_periods = np.diff(cross) * (t_seg[1] - t_seg[0])
T_meas = 2.0 * np.median(half_periods)
assert abs(T_meas - T_pred) / T_pred < 0.03, (T_pred, T_meas)
print(f"2 OK  predicted period {T_pred:.3f} vs simulated {T_meas:.3f}")

# ------------------------------------------------------------------
print("3. closed-loop P on a first-order plant: map eigenvalue is exact")

def first(t, x, p, i):
    return [i[0](t) - x[0]]

Kc, dt = 0.4, 0.01
lp = ClosedLoop(first, {}, PID(Kc), measure=0, y_sp=1.0)
lp.run_simulation(t_end=2.0, IC=[0.0], dt=dt, progress=False)
z_op = stability.augmented_state(lp)
assert z_op.size == 1, z_op
J = stability.closed_loop_jacobian(lp, t_op=lp.t[-1], z_op=z_op)
lam = stability.eigenvalues(J)[0]
# x' = u - x with u = Kc(sp - x) held for the step (ZOH):
# x_next = u + (x - u) e^{-dt}  ->  slope = e^{-dt}(1 + Kc) - Kc
# (note: the discrete eigenvalue of a first-order process is e^{-dt},
#  not 1 - dt -- forward Euler would give 0.98600 here, off by 7e-5)
expected = np.exp(-dt) * (1.0 + Kc) - Kc
assert abs(lam - expected) < 1e-6, (lam, expected)
print(f"3 OK  P-loop map eigenvalue {lam:.8f} == e^-dt(1+Kc) - Kc = {expected:.8f}")

# ------------------------------------------------------------------
print("4. PI loop: augmented state [x, I, e_prev], stable vs unstable")

def pi_eigs(Kc, tauI, umax=2.0):
    lp = ClosedLoop(first, {}, PID(Kc, tauI, umin=0.0, umax=umax),
                    measure=0, y_sp=1.0)
    lp.run_simulation(t_end=20.0, IC=[0.0], progress=False)
    z_op = stability.augmented_state(lp)
    assert z_op.size == 3, z_op
    J = stability.closed_loop_jacobian(lp, t_op=lp.t[-1], z_op=z_op)
    return stability.eigenvalues(J)

ev_ok = pi_eigs(1.0, 10.0)
assert ev_ok.size == 3 and np.max(np.abs(ev_ok)) < 1.0, ev_ok
ev_bad = pi_eigs(250.0, 0.05, umax=500.0)   # Kc*dt ~ 2.5: discrete-time unstable
assert np.max(np.abs(ev_bad)) > 1.0, ev_bad
print(f"4 OK  PI map: max|lambda| = {np.max(np.abs(ev_ok)):.3f} (stable) "
      f"vs {np.max(np.abs(ev_bad)):.3f} (unstable)")

# ------------------------------------------------------------------
print("5. valve included: augmented state [q, x, I, e_prev]")

lv = ClosedLoop(first, {}, PID(1.0, 5.0, umin=0.0, umax=1.0),
                measure=0, y_sp=1.0, valve=(1.0, 1.0), valve0=0.0)
lv.run_simulation(t_end=20.0, IC=[0.0], progress=False)
z_op = stability.augmented_state(lv)
assert z_op.size == 4, z_op
J = stability.closed_loop_jacobian(lv, t_op=lv.t[-1], z_op=z_op)
ev = stability.eigenvalues(J)
assert ev.size == 4 and np.max(np.abs(ev)) < 1.0, ev
print(f"5 OK  valve+PI map: {ev.size} eigenvalues, max|lambda| = "
      f"{np.max(np.abs(ev)):.3f}")

# ------------------------------------------------------------------
print("6. the map matches the simulation (not just the hand formula)")

# perturb the (stable PI) operating point slightly and check that the
# finite-diff Jacobian predicts the one-step move to first order
lp = ClosedLoop(first, {}, PID(1.0, 10.0, umin=0.0, umax=2.0),
                measure=0, y_sp=1.0)
lp.run_simulation(t_end=20.0, IC=[0.0], progress=False)
z_op = stability.augmented_state(lp)
t_op = lp.t[-1]
delta = np.array([0.01, 0.001, -0.002])
J = stability.closed_loop_jacobian(lp, t_op=t_op, z_op=z_op)
z0 = stability.one_step(lp, t_op, z_op)
pred = z0 + J @ delta            # linearized map
exact = stability.one_step(lp, t_op, z_op + delta)
err = np.max(np.abs(exact - pred))
scale = np.max(np.abs(J @ delta))
assert err / max(scale, 1e-12) < 1e-2, (err, scale)   # first-order accuracy
print(f"6 OK  J @ delta predicts the one-step move (rel. error {err/scale:.1e})")

# ------------------------------------------------------------------
print("7. calling the jacobian disturbs nothing (memory + ZOH restored)")

lp = ClosedLoop(first, {}, PID(1.0, 10.0, umin=0.0, umax=2.0),
                measure=0, y_sp=1.0)
lp.run_simulation(t_end=20.0, IC=[0.0], progress=False)
z_op = stability.augmented_state(lp)
mem_before = (lp.controller._I, lp.controller._y, lp.controller._e)
zoh_before = (lp._zoh._t.copy(), lp._zoh._u.copy())
x_before = lp.x.copy()
J1 = stability.closed_loop_jacobian(lp, t_op=lp.t[-1], z_op=z_op)
J2 = stability.closed_loop_jacobian(lp, t_op=lp.t[-1], z_op=z_op)
assert np.allclose(J1, J2)                                   # deterministic
assert (lp.controller._I, lp.controller._y, lp.controller._e) == mem_before
assert np.array_equal(lp._zoh._t, zoh_before[0]) and \
       np.array_equal(lp._zoh._u, zoh_before[1])
lp.run_simulation(t_end=20.0, IC=[0.0], progress=False)      # re-run is clean
assert np.max(np.abs(lp.x - x_before)) == 0.0
print("7 OK  deterministic, memory/ZOH restored, re-run bit-identical")

# ------------------------------------------------------------------
print("8. error messages")

# sensor delay is refused
try:
    ld = ClosedLoop(first, {}, PID(1.0), measure=0, y_sp=1.0, sensor=(1.0, 2.0))
    ld.run_simulation(t_end=2.0, IC=[0.0], progress=False)
    stability.closed_loop_jacobian(ld, t_op=1.0,
                                   z_op=stability.augmented_state(ld))
    assert False
except cme452.ModelError as e:
    assert "delay" in str(e), str(e)

# non-PID controller is refused with a clear message
class Const:
    def update(self, t, y, y_sp): return 1.0
    def reset(self): pass
try:
    lc = ClosedLoop(first, {}, Const(), measure=0, y_sp=1.0)
    lc.run_simulation(t_end=2.0, IC=[0.0], progress=False)
    stability.closed_loop_jacobian(lc, t_op=1.0, z_op=np.array([0.0]))
    assert False
except cme452.ModelError as e:
    assert "memory layout" in str(e), str(e)

# wrong-size z_op names the expected layout
try:
    lp2 = ClosedLoop(first, {}, PID(1.0, 10.0), measure=0, y_sp=1.0)
    lp2.run_simulation(t_end=2.0, IC=[0.0], progress=False)
    stability.closed_loop_jacobian(lp2, t_op=1.0, z_op=np.array([0.0, 0.0]))
    assert False
except cme452.ModelError as e:
    assert "x0" in str(e) and "I" in str(e) and "e_prev" in str(e), str(e)

# no run yet -> dt unknown
try:
    lp3 = ClosedLoop(first, {}, PID(1.0), measure=0, y_sp=1.0)
    stability.closed_loop_jacobian(lp3, t_op=1.0, z_op=np.array([0.0]))
    assert False
except cme452.ModelError as e:
    assert "run_simulation" in str(e), str(e)
print("8 OK  delay / non-PID / bad z_op / no-run all raise clean errors")

# ------------------------------------------------------------------
print("9. eigenvalue_plot smoke test (continuous and discrete)")

ax = stability.eigenvalue_plot([-2.0, -0.5 + 0.3j], title="open loop")
assert ax is not None
ax = stability.eigenvalue_plot([0.9 + 0.2j, 1.1], discrete=True)
assert ax is not None
matplotlib.pyplot.close("all")
print("9 OK  both plot styles render")

print()
print("ALL STABILITY TESTS PASSED (9 checks)")
raise SystemExit(0)
