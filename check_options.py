"""Options: known-angle nodes, and the integer bits G.

1. The check of the options: G = 4, F = 45, m = 4, vectoring nodes for
   the leaf pair (a, b) and the root, and one known-angle node (pi/4) for the
   equal leaves (zeta/sqrt3, zeta/sqrt3).
2. Accuracy with known nodes on the per-frequency vectors
   (a, b, zeta/sqrt3, zeta/sqrt3), a = sin phi1 cos phi2, b = cos phi1 sin phi2,
   zeta = sin phi1 sin phi2, phi_j = pi k_j / N, and on random vectors that
   satisfy the declared relations.
3. Overflow detection when G is too small."""
import numpy as np
from binary_to_amplitude import TreeCircuit, exact_unit


def run(tc, V):
    amp, ck = tc.simulate(V)
    assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"], ck
    e = np.linalg.norm(amp - exact_unit(V), axis=1)
    return e, np.linalg.norm(V, axis=1), ck["no_overflow"]


# 1. counts
F = 45
t = TreeCircuit(2, F, G=4, known={(1, 1): np.pi / 4}).tree
r = t.rotation_counts()
assert t.toffoli() == t.formulas()["toffoli"] == 18624
assert (r["controlled_ry"], r["fixed_ry"], r["flag_z"]) == (2 * F + 1, 2, 1)
print(f"F = {F}, G = 4, known (1,1) = pi/4: Toffoli {t.toffoli()} "
      f"(vectorings 2 x {4 * t.I * (t.W - 1)}, constant product "
      f"{2 * (t.W - 1) * (len(t.digits[(1, 1)]) - 1)}, s = {len(t.digits[(1, 1)])}); "
      f"{r['controlled_ry']} controlled Ry, {r['fixed_ry']} fixed Ry, {r['flag_z']} flag Z")

# 2a. per-frequency vectors
for N in (64, 256):
    k1, k2 = [a.ravel()[1:] for a in np.meshgrid(np.arange(N), np.arange(N), indexing="ij")]
    f1, f2 = np.pi * k1 / N, np.pi * k2 / N
    a, b, z = np.sin(f1) * np.cos(f2), np.cos(f1) * np.sin(f2), np.sin(f1) * np.sin(f2)
    for F, G in ((24, 4), (24, None), (32, 4)):
        tc = TreeCircuit(2, F, G=G, known={(1, 1): np.pi / 4})
        V = np.round(np.stack([a, b, z / np.sqrt(3), z / np.sqrt(3)], 1) * 2 ** F) / 2 ** F
        e, nv, ok = run(tc, V)
        assert ok
        print(f"N = {N:3d}, F = {F}, G = {tc.tree.G}: max err {e.max():.2e}, "
              f"err 2^F/N {e.max() * 2 ** F / N:.2f}, Toffoli {tc.tree.toffoli()}")

# 2b. random vectors satisfying the declared relations, seed 7
rng = np.random.default_rng(7)
cases = [({(0, 0): 0.4}, (), 2), ({(1, 0): -0.3}, ((1, 0),), 2),
         ({(1, 1): 0.6, (0, 0): 0.2}, (), 2), ({(2, 1): -0.5, (1, 1): 0.7}, ((2, 1),), 3)]
for known, signed, L in cases:
    m, F = 2 ** L, 24
    V = rng.uniform(-1, 1, (4096, m)) * 2.0 ** -rng.uniform(0, F / 2, (4096, 1))
    for (q, p), th in sorted(known.items(), reverse=True):          # leaves first
        w, lo = 2 ** (L - q - 1), p * 2 ** (L - q)
        if q == L - 1:
            if (q, p) not in signed:
                V[:, lo] = np.abs(V[:, lo])
            V[:, lo + 1] = np.tan(th) * V[:, lo]
        else:
            nl = np.linalg.norm(V[:, lo:lo + w], axis=1)
            nr = np.linalg.norm(V[:, lo + w:lo + 2 * w], axis=1)
            V[:, lo + w:lo + 2 * w] *= (np.tan(th) * nl / nr)[:, None]
    V = np.round(V * 2 ** F) / 2 ** F
    for (q, p), th in known.items():                               # right leaf is not read
        if q == L - 1:
            V[:, 2 * p + 1] = np.tan(th) * V[:, 2 * p]
    tc = TreeCircuit(L, F, known=known, signed=signed)
    e, nv, ok = run(tc, V)
    assert ok and tc.tree.toffoli() == tc.tree.formulas()["toffoli"]
    print(f"m = {m}, known {known}, signed {list(signed)}: max err ||v|| 2^F "
          f"{(e * nv).max() * 2 ** F:.2f}, Toffoli {tc.tree.toffoli()}")

# 3. overflow
_, _, ok = run(TreeCircuit(2, 16, G=2), rng.uniform(-1, 1, (512, 4)))
assert not ok
print("G = 2 for |v_i| <= 1: overflow detected")
