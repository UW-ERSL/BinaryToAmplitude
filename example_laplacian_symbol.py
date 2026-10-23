"""Sec. 6, Example 1: gradient symbol of the finite-difference Laplacian,
v(k) = (sin(pi k_1/N), ..., sin(pi k_d/N)), A (CORDIC rotation mode) and R
simulated bit by bit over every k != 0."""
import numpy as np
from binary_to_amplitude import SymbolCircuit, exact_unit

print(" d     N   F   end-to-end err   err 2^F/N   R alone: max err ||v|| 2^F   A: max|v - v_exact| 2^F")
worst = [0, 0, 0]
for d, ns in ((2, (4, 6, 8, 10)), (3, (4, 5, 6))):
    for n in ns:
        for F in (16, 24, 32):
            sc = SymbolCircuit(n, d, F)
            amp, vreg, vex, ck = sc.simulate()
            assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"]
            N = 2 ** n
            e = np.linalg.norm(amp - exact_unit(vex), axis=1).max()
            eR = (np.linalg.norm(amp - exact_unit(vreg), axis=1)
                  * np.linalg.norm(vreg, axis=1)).max() * 2 ** F
            eA = np.abs(vreg - vex).max() * 2 ** F
            worst = [max(worst[0], e * 2 ** F / N), max(worst[1], eR), max(worst[2], eA)]
            print(f" {d}  {N:4d}  {F}     {e:.3e}       {e * 2**F / N:6.3f}"
                  f"            {eR:6.3f}                    {eA:6.2f}")
print(f"max err 2^F/N = {worst[0]:.2f},  R alone {worst[1]:.2f},  A {worst[2]:.1f}")
