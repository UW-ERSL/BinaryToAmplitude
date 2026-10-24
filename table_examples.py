"""Table 2: cost of the full circuit (Parts 1 and 2) for the two examples at
F = 32, compute plus uncompute. The counts do not depend on
the index size N."""
import numpy as np
from multiplexed_state_preparation import SymbolVector

F, n = 32, 10


def lap(d):
    C = np.zeros((d, 2 * d + 1))
    for j in range(d):
        C[j, 2 + 2 * j] = 1.0                       # sin theta_j
    return SymbolVector(n, d, F, [(tuple(int(i == j) for i in range(d)), 0) for j in range(d)], C)


stencil = SymbolVector(n, 1, F, [((2,), -1), ((2,), 1)],
                       [[-0.25, 0, -0.125, 0, 0], [0.5, 0, 0.125, 0, 0.125], [-0.25, 0, 0, 0, -0.125]])
print(" case               R  m   angles   trig    rows    tree   rot.  total   ctrl.Ry  qubits")
for name, sv in (("Laplacian, d = 2", lap(2)), ("Laplacian, d = 3", lap(3)), ("stencil rows", stencil)):
    t, r = sv.toffoli(), sv.tree.rotation_counts()
    assert t == sv.formulas()
    tot = sum(t.values()) + r["toffoli"]
    work = sv.c.n - sv.d * sv.n
    print(f" {name:17s}  {len(sv.angles)}  {sv.tree.m}  {t['angles']:6d}  {t['trig']:6d}  {t['rows']:5d}"
          f"  {t['tree']:6d}  {r['toffoli']:4d}  {tot:6d}   {r['controlled_ry']:5d}   {work:6d}")
