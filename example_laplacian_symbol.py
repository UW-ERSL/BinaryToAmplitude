"""Sec. 7, Example 1: gradient symbol of the finite-difference Laplacian,
v(k) = (sin(pi k_1/N), ..., sin(pi k_d/N)), angles theta_j = pi k_j / N and
C selecting the sines. Parts 1 and 2 simulated bit by bit over every k != 0."""
import numpy as np
from binary_to_amplitude import SymbolVector, exact_unit

print(" d     N   F   err 2^F/N   trig err 2^F (bound)   R alone ||v|| 2^F   proven/measured")
worst = dict(e=0, t=0, r=0)
for d, ns in ((2, (4, 6, 8, 10)), (3, (4, 5, 6))):
    angles = [tuple(int(i == j) for i in range(d)) for j in range(d)]
    C = np.zeros((d, 2 * d + 1))
    for j in range(d):
        C[j, 2 + 2 * j] = 1.0
    for n in ns:
        for F in (16, 24, 32):
            sv = SymbolVector(n, d, F, [(p, 0) for p in angles], C)
            amp, vreg, vex, treg, tex, ck = sv.simulate()
            assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"] and ck["no_overflow"]
            assert sv.toffoli() == sv.formulas()
            N = 2 ** n
            e = np.linalg.norm(amp - exact_unit(vex), axis=1)
            et = np.hypot(treg[:, 0::2] - tex[:, 0::2], treg[:, 1::2] - tex[:, 1::2]).max()
            assert et <= sv.eps_trig()
            eR = (np.linalg.norm(amp - exact_unit(vreg), axis=1) * np.linalg.norm(vreg, axis=1)).max()
            bound = sv.end_to_end_bound(vex)
            assert np.all(e <= bound)
            worst = dict(e=max(worst["e"], e.max() * 2 ** F / N), t=max(worst["t"], et * 2 ** F),
                         r=max(worst["r"], eR * 2 ** F))
            print(f" {d}  {N:4d}  {F}   {e.max() * 2**F / N:7.3f}     {et * 2**F:6.2f} ({sv.eps_trig() * 2**F:.2f})"
                  f"          {eR * 2**F:6.3f}           {(bound / e).min():8.0f}")
print(f"max err 2^F/N = {worst['e']:.2f}, trig err 2^F = {worst['t']:.2f}, R alone {worst['r']:.2f}")
