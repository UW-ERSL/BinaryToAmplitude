"""The Fourier blocks of the Poisson isometry (Q4 elements, two dimensions).

1. Cost at N = 1024, F = 45, G = 4, by stage (Table 1 of the paper).
2. Error of the prepared vector c~(k) against c(k)/||c(k)||, simulated bit by bit
   over every k != 0, for N = 4 ... 256 and F = 12, 16, ..., 32 (42 cases),
   with the proven bound of Corollary 6 checked at every k.
3. The block itself: e^(i(phi1+phi2)) T c~(k) against Vhat(k) assembled element
   by element from the Q4 gradient, at N = 16, F = 24."""
import numpy as np
from binary_to_amplitude import poisson_isometry, exact_unit

# 1. cost
F, n = 45, 10
sv = poisson_isometry(n, F)
t, r, tr = sv.toffoli(), sv.tree.rotation_counts(), sv.tree
assert t == sv.formulas()
vec = 4 * tr.I * (tr.W - 1)
const = 2 * (tr.W - 1) * (len(tr.digits[(1, 1)]) - 1)
assert t["tree"] == 2 * vec + const
total = sum(t.values()) + r["toffoli"]
print(f"N = {2**n}, F = {F}, G = 4 (registers of {sv.w} bits)")
for name, val in (("angles phi1 +- phi2 (adder, subtractor)", t["angles"]),
                  ("cosines and sines (two rotation-mode runs)", t["trig"]),
                  ("a, b, zeta (sum to product)", t["rows"]),
                  ("omega: vectoring of (a, b)", vec),
                  (f"sqrt(2/3) K zeta: constant product, s = {len(tr.digits[(1, 1)])}", const),
                  ("tau: vectoring of (K r, sqrt(2/3) K zeta)", vec),
                  ("rotation stage: selection of the omega bits", r["toffoli"])):
    print(f"  {name:48s} {val:7,d}")
print(f"  {'total':48s} {total:7,d}")
print(f"  rotations: {r['controlled_ry']} controlled, {r['fixed_ry']} fixed, {r['flag_z']} flag Z; "
      f"work qubits {sv.c.n - 2 * n:,}")

# 2. error, 42 cases
print("\n  N   F   max err 2^F/N   proven bound 2^F/N (max over k)")
worst = 0
for n in range(2, 9):
    for F in range(12, 33, 4):
        sv = poisson_isometry(n, F)
        amp, vreg, vex, treg, tex, ck = sv.simulate()
        assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"] and ck["no_overflow"]
        e = np.linalg.norm(amp - exact_unit(vex), axis=1)
        b = sv.end_to_end_bound(vex)
        assert np.all(e <= b)
        worst = max(worst, e.max() * 2 ** F / 2 ** n)
        if F in (12, 32):
            print(f"{2**n:4d}  {F}      {e.max() * 2**F / 2**n:6.3f}        {b.max() * 2**F / 2**n:11.1f}")
print(f"max err 2^F/N over the 42 cases: {worst:.2f}")


# 3. the block Vhat(k), assembled element by element
def vhat_exact(N, k1, k2):
    q = 1 / np.sqrt(3)
    gp = [(-q, -q), (q, -q), (q, q), (-q, q)]                     # (xi_g, eta_g)
    corners, xi = [(0, 0), (1, 0), (1, 1), (0, 1)], [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    t1, t2 = 2 * np.pi * k1 / N, 2 * np.pi * k2 / N
    ph = np.array([np.exp(1j * (t1 * cx + t2 * cy)) for cx, cy in corners])
    rows = []
    for r_, s_ in gp:
        Bx = [xa * (1 + ya * s_) for xa, ya in xi]
        By = [ya * (1 + xa * r_) for xa, ya in xi]
        rows += [np.dot(Bx, ph), np.dot(By, ph)]
    v = np.array(rows)
    return v / np.linalg.norm(v)


q = 1 / np.sqrt(3)
T = np.zeros((8, 4), complex)                                   # T|l> = t_l, l = 0..3
for g, (xg, eg) in enumerate([(-q, -q), (q, -q), (q, q), (-q, q)]):
    T[2 * g, 0], T[2 * g + 1, 1] = 0.5j, 0.5j
    T[2 * g, 2], T[2 * g + 1, 3] = -np.sqrt(3) / 2 * eg, -np.sqrt(3) / 2 * xg
n, F = 4, 24
N = 2 ** n
sv = poisson_isometry(n, F)
amp, vreg, vex, treg, tex, ck = sv.simulate()
k1, k2 = [a.ravel()[1:] for a in np.meshgrid(np.arange(N), np.arange(N), indexing="ij")]
err = max(np.linalg.norm(np.exp(1j * np.pi * (a + b) / N) * T @ amp[i] - vhat_exact(N, a, b))
          for i, (a, b) in enumerate(zip(k1, k2)))
print(f"\nN = {N}, F = {F}: max |e^(i(phi1+phi2)) T c~(k) - Vhat(k)| = {err:.1e}")
