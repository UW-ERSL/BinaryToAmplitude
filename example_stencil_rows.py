"""Sec. 6, Example 2: row states of the one-dimensional diffusion matrix with
coefficient c(x) = 1 + sin(2 pi x)/2, v = (-c_{j-1/2}, c_{j-1/2} + c_{j+1/2},
-c_{j+1/2}, 0)/4. Inputs rounded to F bits classically; A is not simulated."""
import numpy as np
from binary_to_amplitude import TreeCircuit, exact_unit

N = 1024
x = (np.arange(N) + 0.5) / N
cm = 1 + 0.5 * np.sin(2 * np.pi * (x - 0.5 / N))
cp = 1 + 0.5 * np.sin(2 * np.pi * (x + 0.5 / N))
V = np.stack([-cm, cm + cp, -cp, 0 * cm], 1) / 4
print(f"N = {N} rows, min ||v|| = {np.linalg.norm(V, axis=1).min():.3f}")
for F in (16, 24, 32):
    Vq = np.round(V * 2 ** F) / 2 ** F
    amp, ck = TreeCircuit(2, F).simulate(Vq)
    assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"]
    e = np.linalg.norm(amp - exact_unit(Vq), axis=1)
    print(f" F = {F}   max err {e.max():.2e}")
