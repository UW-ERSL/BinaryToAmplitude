"""Sec. 5: assert the Toffoli and work-qubit counts against the gate lists, and
check that the work register returns to zero and the inputs are unchanged."""
import numpy as np
from binary_to_amplitude import TreeCircuit

rng = np.random.default_rng(0)
print(" m   F   G   Toffoli (arith.)  work qubits  rot. Toffoli  ctrl. Ry  clean")
for L in (1, 2, 3, 4):
    for F in (16, 32, 45):
        tc = TreeCircuit(L, F)
        t, f = tc.tree, tc.tree.formulas()
        assert t.toffoli() == f["toffoli"] and t.work_qubits == f["work"]
        V = rng.uniform(-1, 1, (256, 2 ** L))
        _, ck = tc.simulate(V)
        assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"]
        r = t.rotation_counts()
        print(f"{2**L:2d}  {F:3d}  {t.G}   {t.toffoli():10d}   {t.work_qubits:10d}"
              f"   {r['toffoli']:8d}   {r['controlled_ry']:6d}    yes")
print("all counts match the formulas of Sec. 5")

# Part 1: the counts of the full circuit against its gate lists, on random
# angles and coefficient matrices, with the work register checked on every index
print("\nPart 1 (SymbolVector): random angles p in {-3..3}^d, q in {-4..4}, random C")
from binary_to_amplitude import SymbolVector, exact_unit
for trial in range(6):
    d, R, m0 = 1 + trial % 2, 1 + trial % 3, 2 + trial % 3
    n, F = 4, 16 + 4 * (trial % 3)
    angles = [(tuple(int(x) for x in rng.integers(-3, 4, d)), int(rng.integers(-4, 5))) for _ in range(R)]
    C = np.round(rng.uniform(-1, 1, (m0, 2 * R + 1)) * 8) / 8 / (2 * R + 1)
    sv = SymbolVector(n, d, F, angles, C)
    assert sv.toffoli() == sv.formulas()
    amp, vreg, vex, treg, tex, ck = sv.simulate()
    assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"] and ck["no_overflow"]
    assert np.all(np.abs(vreg - vex) <= sv.entry_bounds())
    e = np.linalg.norm(amp - exact_unit(vex), axis=1)
    print(f"  d = {d}, R = {R}, m = {m0}, F = {F}: Toffoli {sv.toffoli()}, entries within bound, "
          f"max err {e.max():.1e}, clean")
print("all Part 1 counts match the formulas of Sec. 5")
