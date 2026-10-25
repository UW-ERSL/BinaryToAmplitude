"""Sec. 5, Table 3: the CORDIC circuit against the two standard constructions
(multiplexor of standard state preparation, QROM lookup of tabulated angles),
at the accuracy delta = 1.955e-10 of the companion paper (rho = 10,
eps_p = 1e-6), for N = 8 to 4096. Asserts the statements of Sec. 5 and 6,
prints every N, and writes the rows of Table 3 (N = 16, 64, 256, 1024, 2048,
4096) to tables/options.tex.

    python table_options.py      # about 1 minute"""
import os

import numpy as np
from multiplexed_state_preparation.alternatives import options

delta = 1.955e-10
Ns = [2 ** n for n in range(3, 13)]
o = {N: options(N, delta) for N in Ns}
T = lambda N, k: o[N][k]["T"]
assert all(T(N, "multiplexor") < T(N, "arithmetic") for N in Ns if N <= 16)
assert all(T(N, "multiplexor") > T(N, "arithmetic") for N in Ns if N >= 32)
assert all(T(N, "lookup") < T(N, "arithmetic") for N in Ns if N <= 2048)
assert T(4096, "lookup") > T(4096, "arithmetic")
assert o[1024]["arithmetic"]["F"] == 45 and o[1024]["arithmetic"]["toffoli"] == 44424
assert o[1024]["lookup"]["b"] == 35

print(f"delta = {delta:g}")
print("     N   F    T: multiplexor    lookup  arithmetic   Toffoli: lookup  arithmetic"
      "   work: lookup  arithmetic")
for N in Ns:
    m, l, a = o[N]["multiplexor"], o[N]["lookup"], o[N]["arithmetic"]
    print(f"{N:6d}  {a['F']:2d}    {m['T']:14.3g}  {l['T']:8.3g}  {a['T']:10.3g}"
          f"   {l['toffoli']:14,d}  {a['toffoli']:10,d}   {l['work']:12,d}  {a['work']:10,d}")
l, a = o[1024]["lookup"], o[1024]["arithmetic"]
print(f"N = 1024: arithmetic / lookup Toffoli = {a['toffoli'] / l['toffoli']:.2f}; "
      f"rotation synthesis of the arithmetic {a['rotations']} rotations, "
      f"{a['T'] - 4 * a['toffoli']:,.0f} T gates")


# ----- the rows of Table 3, as typeset in the paper
def sci(x):
    e = int(np.floor(np.log10(x)))
    m = round(x / 10 ** e, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"${m:.1f} \\times 10^{{{e}}}$"


def intx(x):
    return f"${x:,d}$".replace(",", "{,}")


rows = []
for N in (16, 64, 256, 1024, 2048, 4096):
    m, l, a = o[N]["multiplexor"], o[N]["lookup"], o[N]["arithmetic"]
    rows.append(" & ".join([f"${N}$", f"${a['F']}$", sci(m["T"]), sci(l["T"]), sci(a["T"]),
                            intx(l["toffoli"]), intx(a["toffoli"]), intx(l["work"]), intx(a["work"])]) + r"\\")
os.makedirs("tables", exist_ok=True)
open("tables/options.tex", "w").write("\n".join(rows) + "\n")
print("\nrows of Table 3 (tables/options.tex):")
print("\n".join(rows))
