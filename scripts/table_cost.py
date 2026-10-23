"""Table 1: cost and measured error at F = 32 (g = 5), m = 2 to 16.
Error column: max of ||v|| 2^F ||v~ - v/||v|| || over 4,096 inputs
u 2^-e, u ~ U(-1, 1), e ~ U(0, F/2), seed 4."""
import numpy as np
from binary_to_amplitude import TreeCircuit, exact_unit

F, B = 32, 4096
rng = np.random.default_rng(4)
print(f"F = {F}, {B} random inputs, seed 4")
print(" m  G   arith.  rot.  ctrl.Ry  qubits   error")
for L in (1, 2, 3, 4):
    tc = TreeCircuit(L, F)
    t, m = tc.tree, 2 ** L
    e = rng.uniform(0, F / 2, (B, m))
    V = np.round(rng.uniform(-1, 1, (B, m)) * 2.0 ** -e * 2 ** F) / 2 ** F
    V = V[np.linalg.norm(V, axis=1) > 0]
    amp, ck = tc.simulate(V)
    assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"]
    nv = np.linalg.norm(V, axis=1)
    err = (np.linalg.norm(amp - exact_unit(V), axis=1) * nv).max() * 2 ** F
    r = t.rotation_counts()
    print(f"{m:2d}  {t.G}  {t.toffoli():7d}  {r['toffoli']:4d}  {r['controlled_ry']:5d}"
          f"  {t.work_qubits:6d}   {err:.2f}")
