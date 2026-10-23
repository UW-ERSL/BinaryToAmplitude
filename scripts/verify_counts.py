"""Sec. 5: assert the Toffoli and work-qubit counts against the gate lists, and
check that the work register returns to zero and the inputs are unchanged."""
import numpy as np
from binary_to_amplitude import TreeCircuit

rng = np.random.default_rng(0)
print(" m   F   g  G   Toffoli (arith.)  work qubits  rot. Toffoli  ctrl. Ry  clean")
for L in (1, 2, 3, 4):
    for F in (16, 32, 45):
        tc = TreeCircuit(L, F)
        t, f = tc.tree, tc.tree.formulas()
        assert t.toffoli() == f["toffoli"] and t.work_qubits == f["work"]
        V = rng.uniform(-1, 1, (256, 2 ** L))
        _, ck = tc.simulate(V)
        assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"]
        r = t.rotation_counts()
        print(f"{2**L:2d}  {F:3d}  {t.g}  {t.G}   {t.toffoli():10d}   {t.work_qubits:10d}"
              f"   {r['toffoli']:8d}   {r['controlled_ry']:6d}    yes")
print("all counts match the formulas of Sec. 5")
