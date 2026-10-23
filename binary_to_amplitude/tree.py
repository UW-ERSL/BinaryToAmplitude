"""tree -- binary-to-amplitude conversion of an m-entry vector by a tree of
CORDIC vectorings (the circuit R of the paper).

Given m = 2^L value registers v_0..v_{m-1} (two's complement, F fractional
bits), the arithmetic stage copies them into working registers with
F' = F + g fractional bits (g guard bits, default ceil(log2 F)) and
W = F' + G bits, and runs m - 1 vectorings of I = F' iterations from the
leaves to the root. Node (q, p) at level q covers leaves p 2^(L-q) to
(p+1) 2^(L-q) - 1; its decision bits encode atan2(norm of right half, norm of
left half). A leaf pair with v_2p < 0 is negated under a flag, which adds pi to
its angle. The uncomputation is the exact reverse.

The arithmetic stage is built and simulated at gate level. The rotation stage
(target qubits t_0 .. t_{L-1}, t_0 most significant) is counted by
rotation_counts() but not simulated at gate level: the simulation applies the
angles it encodes, theta = A - 2 sum_i d_i alpha_i (+ pi f at the leaves).
"""
from __future__ import annotations

import numpy as np

from .revcirc import (Circuit, BitSim, copy_view, shl, load_select,
                      cordic_rotate, cordic_vector, cordic_gain, to_int, reverse)


def default_guard(F):
    """Guard bits g = ceil(log2 F)."""
    return int(np.ceil(np.log2(F)))


def range_bits(L, vmax=1.0):
    """Integer bits G with K^L sqrt(m) vmax < 2^(G-1) (root register range)."""
    bound = cordic_gain(64) ** L * np.sqrt(2 ** L) * vmax
    return int(np.ceil(np.log2(bound))) + 2


def _toffoli(gates):
    """Toffoli count, compute plus uncompute, of a one-way gate list."""
    k = {t: sum(1 for q in gates if q[0] == t) for t in ("ccx", "and", "andu")}
    return 2 * k["ccx"] + k["and"] + k["andu"]


# ==========================================================================
# the circuit R (arithmetic stage), on an existing circuit
# ==========================================================================
class Tree:
    def __init__(self, c, regs, F, g=None, G=None):
        m = len(regs)
        L = int(np.log2(m))
        assert 2 ** L == m, "pad with zero registers to a power of 2"
        self.c, self.regs, self.F, self.m, self.L = c, regs, F, m, L
        self.g = default_guard(F) if g is None else g
        self.Fi = F + self.g                     # internal fractional bits F'
        self.I = self.Fi
        self.G = range_bits(L) if G is None else G
        self.W = W = self.Fi + self.G
        n0, g0 = c.n, len(c.gates)

        # working copies, shifted by g (CNOT only)
        xs = []
        for r in regs:
            x = c.alloc(W)
            copy_view(c, shl(r + [r[-1]] * (W - len(r)), self.g), x)
            xs.append(x)

        # leaf pairs: sign flag, then vectoring
        self.flags, self.dbits = {}, {}
        cur = []
        for p in range(m // 2):
            x, y = xs[2 * p], xs[2 * p + 1]
            f = c.alloc(1)[0]
            c.cx(x[-1], f)
            for j in range(W):
                c.cx(f, x[j]); c.cx(f, y[j])     # (x, y) <- ~(x, y) = -(x, y) - ulp
            self.flags[p] = f
            self.dbits[(L - 1, p)] = cordic_vector(c, x, y, self.I)
            cur.append(x)

        # internal nodes, leaves to root, in place on the children's x registers
        for q in range(L - 2, -1, -1):
            nxt = []
            for p in range(2 ** q):
                self.dbits[(q, p)] = cordic_vector(c, cur[2 * p], cur[2 * p + 1], self.I)
                nxt.append(cur[2 * p])
            cur = nxt
        self.root = cur[0]
        self.gate_range = (g0, len(c.gates))
        self.work_qubits = c.n - n0

    # -- decoding ---------------------------------------------------------------
    def angles(self, sim):
        """Node angles theta = A - 2 sum_i d_i alpha_i (+ pi f at the leaves)."""
        al = np.arctan(2.0 ** -np.arange(self.I))
        th = {}
        for (q, p), bits in self.dbits.items():
            t = sum((1 - 2 * sim.get_bits(d)) * a for d, a in zip(bits, al))
            if q == self.L - 1:
                t = t + np.pi * sim.get_bits(self.flags[p])
            th[(q, p)] = t
        return th

    def amplitudes(self, th, B):
        """Prepared vector: product of cos/sin of the node angles along each path."""
        L, m = self.L, self.m
        amp = np.ones((B, m))
        for l in range(m):
            for q in range(L):
                t = th[(q, l >> (L - q))]
                amp[:, l] *= np.sin(t) if (l >> (L - 1 - q)) & 1 else np.cos(t)
        return amp

    # -- resources --------------------------------------------------------------
    def toffoli(self):
        """Arithmetic stage, compute plus uncompute, counted from the gate list."""
        return _toffoli(self.c.gates[slice(*self.gate_range)])

    def formulas(self):
        """Closed forms of Sec. 5 for the arithmetic stage."""
        I, W, m = self.I, self.W, self.m
        return dict(toffoli=4 * I * (W - 1) * (m - 1),
                    work=m * W + m // 2 + 2 * W
                    + (m - 1) * (I + I * W - I * (I - 1) // 2))

    def rotation_counts(self):
        """Rotation stage (counted, not simulated at gate level).

        Level 0 uses the root decision bits as controls. Level q >= 1 selects
        D_i = XOR_p [t_0..t_{q-1} = p] d_i^(q, p) by unary iteration over the
        2^q prefixes (2^q - 1 temporary ANDs) with one Toffoli per (p, i); D_i
        is uncomputed by measurement, whose phase correction (-1)^D_i repeats
        the unary iteration with CZ gates (another 2^q - 1 ANDs). The leaf
        sign flags are selected the same way and take a Z. Each level applies
        one fixed rotation R(A) and I rotations R(-2 alpha_i) controlled by D_i.
        """
        I, L = self.I, self.L
        tof = sum(2 * (2 ** q - 1) + (I + (q == L - 1)) * 2 ** q
                  for q in range(1, L))
        return dict(toffoli=tof, controlled_ry=L * I, fixed_ry=L)


# ==========================================================================
# drivers
# ==========================================================================
def exact_unit(V):
    V = np.asarray(V, float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def _dirty_bits(sim, n, protected):
    return int(sum(sim.get_bits(q).sum() for q in range(n) if q not in protected))


class TreeCircuit:
    """R alone, on caller registers loaded with given values (rounded to F bits)."""

    def __init__(self, L, F, g=None, G=None, w_in=None):
        c = Circuit()
        self.F = F
        self.w_in = F + 2 if w_in is None else w_in       # |v_i| < 2
        self.v = [c.alloc(self.w_in) for _ in range(2 ** L)]
        self.tree = Tree(c, self.v, F, g, G)
        self.n_arith = len(c.gates)
        c.gates += reverse(c.gates)
        self.c = c

    def simulate(self, V):
        """V: (B, m). Returns prepared amplitudes (B, m) and checks."""
        V = np.asarray(V, float)
        B = V.shape[0]
        Z = np.round(V * 2 ** self.F).astype(np.int64)
        sim = BitSim(self.c.n, B)
        for l, r in enumerate(self.v):
            sim.set_int(r, Z[:, l] & ((1 << self.w_in) - 1))
        sim.run(self.c.gates[:self.n_arith])
        th = self.tree.angles(sim)
        amp = self.tree.amplitudes(th, B)
        sim.run(self.c.gates[self.n_arith:])
        dirty = _dirty_bits(sim, self.c.n, {w for r in self.v for w in r})
        restored = all(np.all(sim.get_int(r) == Z[:, l]) for l, r in enumerate(self.v))
        return amp, dict(work_bits_left_set=dirty, inputs_restored=restored)


class SymbolCircuit:
    """A followed by R for v(k) = (sin(pi k_1/N), ..., sin(pi k_d/N)), the
    gradient symbol of the finite-difference Laplacian up to phases, padded
    with zero registers to m = 2^ceil(log2 d).

    A is CORDIC rotation mode on pi k_j/N (F fractional bits, F iterations).
    For k_j/N >= 1/2 a flag subtracts pi from the angle and starts the rotation
    from (-1/K, 0), so the rotated angle lies in [-pi/2, pi/2).
    """

    def __init__(self, n, d, F, g=None, guardA=4):
        c = Circuit()
        self.n, self.N, self.d, self.F = n, 2 ** n, d, F
        self.k = [c.alloc(n) for _ in range(d)]
        Fz, wz, w = F, F + 1, F + guardA
        K = cordic_gain(F)
        sins = []
        for kj in self.k:
            z = c.alloc(wz)
            for j in range(n):
                c.cx(kj[j], z[Fz - n + j])         # z = k_j / N, in units of pi
            f = c.alloc(1)[0]
            c.cx(z[-1], f); c.cx(z[-2], f)         # |z| >= 1/2
            c.cx(f, z[-1])                         # z <- z + 1 (mod 2)
            x, y = c.alloc(w), c.alloc(w)
            load_select(c, x, to_int(1 / K, F, w), to_int(-1 / K, F, w), f)
            cordic_rotate(c, x, y, z, F, F, Fz)
            sins.append(y)
        self.toffoli_A = _toffoli(c.gates)
        m = 2 ** int(np.ceil(np.log2(d)))
        regs = sins + [c.alloc(w) for _ in range(m - d)]   # zero padding
        self.sins = sins
        self.tree = Tree(c, regs, F, g)
        self.n_arith = len(c.gates)
        c.gates += reverse(c.gates)
        self.c = c

    def simulate(self):
        """Every k != 0. Returns prepared amplitudes, register values of v,
        exact v, and checks."""
        N, d = self.N, self.d
        idx = np.arange(1, N ** d)
        ks = [(idx // N ** (d - 1 - j)) % N for j in range(d)]
        B = len(idx)
        sim = BitSim(self.c.n, B)
        for kj, val in zip(self.k, ks):
            sim.set_int(kj, val)
        sim.run(self.c.gates[:self.n_arith])
        vreg = np.stack([sim.get_int(y) / 2 ** self.F for y in self.sins], 1)
        amp = self.tree.amplitudes(self.tree.angles(sim), B)[:, :d]
        sim.run(self.c.gates[self.n_arith:])
        dirty = _dirty_bits(sim, self.c.n, {q for r in self.k for q in r})
        restored = all(np.all(sim.get_int(kj, False) == val) for kj, val in zip(self.k, ks))
        vex = np.stack([np.sin(np.pi * kk / N) for kk in ks], 1)
        return amp, vreg, vex, dict(work_bits_left_set=dirty, inputs_restored=restored)
