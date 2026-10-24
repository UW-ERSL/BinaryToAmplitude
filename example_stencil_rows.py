"""Sec. 7, Example 2: row states of the one-dimensional diffusion matrix with
coefficient c(x) = 1 + sin(2 pi x)/2 on N = 1024 nodes x_j = j/N:
v = (-c_{j-1/2}, c_{j-1/2} + c_{j+1/2}, -c_{j+1/2}, 0)/4, that is, angles
theta_-+ = pi (2j -+ 1)/N and the rows of C below. Parts 1 and 2 simulated
bit by bit over every row; default guard bits."""
import numpy as np
from binary_to_amplitude import SymbolVector, exact_unit

C = [[-0.25, 0, -0.125, 0, 0],         # -c_{j-1/2}/4
     [0.5, 0, 0.125, 0, 0.125],        # (c_{j-1/2} + c_{j+1/2})/4
     [-0.25, 0, 0, 0, -0.125]]         # -c_{j+1/2}/4
n = 10
for F in (16, 24, 32):
    sv = SymbolVector(n, 1, F, [((2,), -1), ((2,), 1)], C)
    amp, vreg, vex, treg, tex, ck = sv.simulate()
    assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"] and ck["no_overflow"]
    assert sv.toffoli() == sv.formulas()
    e = np.linalg.norm(amp - exact_unit(vex), axis=1)
    dv = np.abs(vreg - vex).max(0)
    assert np.all(dv <= sv.entry_bounds()) and np.all(e <= sv.end_to_end_bound(vex))
    print(f" F = {F}   min ||v|| = {np.linalg.norm(vex, axis=1).min():.3f}   max err {e.max():.2e}"
          f"   entry err 2^F {dv.max() * 2**F:.2f} (bound {sv.entry_bounds().max() * 2**F:.2f})"
          f"   proven/measured {(sv.end_to_end_bound(vex) / e).min():.0f}")
