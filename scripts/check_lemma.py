"""Lemma 2: check every inequality of the proof on integer vectoring with
floor-truncated shifts (I fractional bits, I iterations, u = 2^-I), on
200,000 random pairs per I with magnitudes down to 2^-I, seed 5."""
import numpy as np

rng = np.random.default_rng(5)
print(" I   x nondecr.  |y_I| bound   eps/bound_a   lenerr/bound_b")
for I in (8, 16, 24, 32, 40):
    u, B = 2.0 ** -I, 200000
    x0 = np.floor(rng.uniform(0, 1, B) * 2.0 ** -rng.uniform(0, I, B) * 2 ** I).astype(np.int64)
    y0 = np.floor(rng.uniform(-1, 1, B) * 2.0 ** -rng.uniform(0, I, B) * 2 ** I).astype(np.int64)
    keep = (x0 != 0) | (y0 != 0)
    x0, y0 = x0[keep], y0[keep]

    # vectoring
    x, y, th, mono = x0.copy(), y0.copy(), np.zeros(len(x0)), True
    for i in range(I):
        s = np.where(y >= 0, 1, -1)
        xn, yn = x + s * (y >> i), y - s * (x >> i)
        mono &= bool(np.all(xn >= x))
        x, y = xn, yn
        th += s * np.arctan(2.0 ** -i)

    # the two claims of Lemma 2
    K = np.prod(np.sqrt(1 + 4.0 ** -np.arange(I)))
    X0, Y0, X, Y = x0 * u, y0 * u, x * u, y * u
    rho = np.hypot(X0, Y0)
    resid = bool(np.all(np.abs(Y) <= 2.0 ** (1 - I) * X + (I - 1) * u))
    eps = np.hypot(rho * np.cos(th) - X0, rho * np.sin(th) - Y0)
    ra = (eps / (2 * np.sqrt(2) * 2.0 ** -I * rho + 2.95 * (I - 1) * u)).max()
    rb = (np.abs(X / K - rho) / (2 * 2.0 ** -I * rho + 2.52 * (I - 1) * u)).max()
    print(f"{I:2d}      {mono}        {resid}        {ra:.3f}          {rb:.3f}")
