"""Lemma 2 (rotation mode) of the companion paper: check the bound
||(x_I, y_I) - (cos theta, sin theta)|| <= 2^(1-I) + I u_z / 2 + K u / 2 + sqrt(2) kappa (I - 1) u, kappa = 1.0415,
on integer rotation mode with floor-truncated shifts, angle constants rounded to
the angle grid, and the start value 1/K rounded, as in revcirc.cordic_rotate.
I = F' fractional bits for x, y and the angle (in units of pi), u = 2^-I,
u_z = pi 2^-I; 200,000 random angles in [-pi/2, pi/2) per I, seed 6. The check
also confirms the step of the proof: the residual angle is at most
alpha_(I-1) + I u_z / 2."""
import numpy as np

rng = np.random.default_rng(6)
print(" I   max err / bound   max residual / bound")
for I in (8, 16, 24, 32, 40):
    u, B = 2.0 ** -I, 200000
    K = np.prod(np.sqrt(1 + 4.0 ** -np.arange(I)))
    z0 = rng.integers(-2 ** (I - 1), 2 ** (I - 1), B)          # theta = pi z0 2^-I
    a = np.round(np.arctan(2.0 ** -np.arange(I)) / np.pi * 2 ** I).astype(np.int64)
    x = np.full(B, int(np.round(2 ** I / K)), np.int64)
    y = np.zeros(B, np.int64)
    z = z0.copy()
    phi = np.zeros(B)
    for i in range(I):
        s = np.where(z < 0, -1, 1)
        x, y = x - s * (y >> i), y + s * (x >> i)
        z = z - s * a[i]
        phi += s * np.arctan(2.0 ** -i)
    th = np.pi * z0 * u
    err = np.hypot(x * u - np.cos(th), y * u - np.sin(th))
    uz = np.pi * u
    bound = 2.0 ** (1 - I) + I * uz / 2 + K * u / 2 + np.sqrt(2) * 1.0415 * (I - 1) * u
    rbound = np.arctan(2.0 ** -(I - 1)) + I * uz / 2
    print(f"{I:2d}      {err.max() / bound:.3f}             {np.abs(th - phi).max() / rbound:.3f}")
    assert err.max() <= bound and np.abs(th - phi).max() <= rbound
print("Lemma 2 holds in every case")
