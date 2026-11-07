"""
Kicked one-dimensional system:  dx/dt = x^2 - 1  on M = [-1, 1],
plus a "fire" x -> x/K applied at times T, 2T, 3T, ...

Three independent ways of computing E_t[O] starting from f0 = 1/2:
  1. exact     : closed form, using the fact that all maps are Moebius maps
  2. direct    : E_t[O] = E_0[O o S_t]           (average over initial points)
  3. ttcf      : E_0[O] + flow TTCF terms + jump terms  (formula of the notes)
  4. kawasaki  : E_0[O * exp(Omega_t) * 1_{S_t(M)}]     (Eulerian / weighted f0)

Only numpy is needed.  Run:  python3 kicked_ttcf.py
"""
import numpy as np

K = 2.0          # fire factor
T = 0.5          # time between fires
O = lambda x: x  # observable

# ---------------------------------------------------------------- dynamics
def flow(x, t):
    """Phi^t x = tanh(artanh(x) - t), written as a Moebius map (safe at x=+-1)."""
    th = np.tanh(t)
    return (x - th) / (1.0 - th * x)

def kick(x):
    return x / K

def evolve(x, s, t):
    """S_{s,t}: evolve from time s to time t >= s. Kicks at kT (k>=1), applied
    when the trajectory reaches kT; a start at s = kT means 'just after' kick k."""
    x = np.array(x, dtype=float)
    k_next = np.floor(s / T + 1e-12) + 1          # first kick strictly after s
    cur = s
    while k_next * T <= t + 1e-12:
        x = kick(flow(x, k_next * T - cur))
        cur = k_next * T
        k_next += 1
    return flow(x, t - cur)

# Gauss-Legendre nodes on [-1,1] for the average over f0 = 1/2
xg, wg = np.polynomial.legendre.leggauss(400)
E0 = lambda vals: 0.5 * np.sum(wg * vals)

# ---------------------------------------------------------------- 1. exact
def moebius_matrix(t):
    """2x2 matrix of S_{0,t} acting as x -> (a x + b)/(c x + d)."""
    def Mflow(tau):
        th = np.tanh(tau)
        return np.array([[1.0, -th], [-th, 1.0]])
    Mk = np.array([[1.0, 0.0], [0.0, K]])
    n = int(np.floor(t / T + 1e-12))
    M = np.eye(2)
    for _ in range(n):
        M = Mk @ Mflow(T) @ M
    return Mflow(t - n * T) @ M

def exact_mean_x(t):
    (a, b), (c, d) = moebius_matrix(t)
    if abs(c) < 1e-14:                       # t = 0: identity map
        return b / d
    # (1/2) int_{-1}^{1} (a x + b)/(c x + d) dx
    return 0.5 * (2 * a / c + (b * c - a * d) / c**2 * np.log(abs((c + d) / (d - c))))

# ---------------------------------------------------------------- 2. direct
def direct(t):
    return E0(O(evolve(xg, 0.0, t)))

# ---------------------------------------------------------------- 3. TTCF with jumps
Omega0 = lambda x: -2.0 * x          # Omega^{(0)} = -Lambda - G f0'/f0, f0 uniform
sg, swg = np.polynomial.legendre.leggauss(200)   # quadrature in s on each segment

def ttcf(t):
    n = int(np.floor(t / T + 1e-12))
    res = E0(O(xg))
    # flow part: sum over segments (kT, min((k+1)T, t))
    for k in range(n + 1):
        a, b = k * T, min((k + 1) * T, t)
        if b - a < 1e-14:
            continue
        s_nodes = 0.5 * (b - a) * sg + 0.5 * (a + b)
        inner = np.array([E0(Omega0(xg) * O(evolve(xg, s, t))) for s in s_nodes])
        res += 0.5 * (b - a) * np.sum(swg * inner)
    # jump part: sum over kicks k = 1..n
    for k in range(1, n + 1):
        res += E0(O(evolve(kick(xg), k * T, t)) - O(evolve(xg, k * T, t)))
    return res

# ---------------------------------------------------------------- 4. Kawasaki
def inverse_and_jacobian(t):
    """Matrix of S_t^{-1} and its determinant, for f_t = f0(S^-1 x)|dS^-1/dx| 1_{S_t(M)}."""
    M = moebius_matrix(t)
    Minv = np.linalg.inv(M)
    return Minv, np.linalg.det(Minv)

def kawasaki(t):
    """E_t[O] = int f0(x) exp(Omega_t(x)) 1_{S_t(M)}(x) O(x) dx with f0 = 1/2,
    where exp(Omega_t(x)) = |d S_t^{-1} x/dx| (Jacobian of the inverse map).
    The indicator is handled by integrating directly over the support S_t(M)."""
    Minv, det = inverse_and_jacobian(t)
    (a, b), (c, d) = Minv
    lo, hi = sorted(evolve(np.array([-1.0, 1.0]), 0.0, t))   # S_t(M) = [lo, hi]
    xs = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
    jac = abs(det) / (c * xs + d) ** 2                       # exp(Omega_t(x))
    return 0.5 * (hi - lo) * np.sum(wg * O(xs) * jac * 0.5)

# ---------------------------------------------------------------- fixed point
tau = np.tanh(T)
xstar = ((K - 1) - np.sqrt((K - 1) ** 2 + 4 * K * tau**2)) / (2 * K * tau)

# ---------------------------------------------------------------- non-uniform f0
# f0(x) = (3/4)(1 - x^2):  Omega^{(0)} = -Lambda - G f0'/f0 = -2x - 2x = -4x
f0b = lambda x: 0.75 * (1 - x**2)
E0b = lambda vals: np.sum(wg * f0b(xg) * vals)
def direct_b(t):
    return E0b(O(evolve(xg, 0.0, t)))
def ttcf_b(t):
    n = int(np.floor(t / T + 1e-12))
    res = E0b(O(xg))
    for k in range(n + 1):
        a, b = k * T, min((k + 1) * T, t)
        if b - a < 1e-14:
            continue
        s_nodes = 0.5 * (b - a) * sg + 0.5 * (a + b)
        inner = np.array([E0b(-4 * xg * O(evolve(xg, s, t))) for s in s_nodes])
        res += 0.5 * (b - a) * np.sum(swg * inner)
    for k in range(1, n + 1):
        res += E0b(O(evolve(kick(xg), k * T, t)) - O(evolve(xg, k * T, t)))
    return res

if __name__ == "__main__":
    print(f"K = {K}, T = {T},  x* = {xstar:.10f}")
    print(f"{'t':>6} {'exact':>14} {'direct':>14} {'ttcf+jumps':>14} {'kawasaki':>14}  support")
    for t in [0.25, 0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 5.0]:
        lo, hi = sorted(evolve(np.array([-1.0, 1.0]), 0.0, t))
        print(f"{t:6.2f} {exact_mean_x(t):14.10f} {direct(t):14.10f} "
              f"{ttcf(t):14.10f} {kawasaki(t):14.10f}  [{lo:.4f}, {hi:.4f}]")

    print("\nNon-uniform f0 = (3/4)(1-x^2):")
    print(f"{'t':>6} {'direct':>14} {'ttcf+jumps':>14}")
    for t in [0.25, 0.5, 1.0, 1.3, 2.0, 3.0]:
        print(f"{t:6.2f} {direct_b(t):14.10f} {ttcf_b(t):14.10f}")
