# BinaryToAmplitude

Code for the paper *Multiplexed state preparation of the Fourier blocks of the
periodic Poisson isometry* (K. Suresh, UW-Madison), the companion of *A
Preconditioned Quantum Linear Solver for Periodic Two-Phase Heat Conduction
with Mesh-Independent Condition Number*. Every number in the paper is
reproduced by the scripts below.

The Fourier transform block-diagonalizes the isometry V = W^(1/2) G (K^+)^(1/2)
of the periodic Poisson problem with bilinear elements. For every wavenumber
k != 0, the circuit prepares

    c(k)/||c(k)||,   c = (a, b, zeta/sqrt3, zeta/sqrt3),
    a = sin(phi_1) cos(phi_2),  b = cos(phi_1) sin(phi_2),  zeta = sin(phi_1) sin(phi_2),
    phi_j = pi k_j / N,

on two target qubits, controlled by the 2n qubits of k. It has two parts.

1. **Part 1, entries from the wavenumber bits** (`SymbolVector`): the angles
   phi_1 +- phi_2 are wired from the bits and added, CORDIC rotation mode gives
   their cosines and sines, and sum to product forms a, b, zeta as fixed-point
   numbers.
2. **Part 2, amplitudes from the entries** (`Tree`): two CORDIC vectorings and
   one constant product (the hourglass pair has the fixed angle pi/4); the
   decision bits of each vectoring control the rotations directly.

There is no division, square root, or angle register, and the cost depends on
the number of fractional bits F, not on N. At N = 1024 and F = 45 the circuit
uses 44,424 Toffoli gates and 90 controlled rotations, with 5,625 work qubits.

## Install

    git clone https://github.com/UW-ERSL/BinaryToAmplitude
    cd BinaryToAmplitude
    pip install -e .

Only numpy is required.

## Quick start

```python
import numpy as np
from binary_to_amplitude import poisson_isometry, exact_unit

sv = poisson_isometry(n=6, F=24)                     # N = 64, g = 0 guard bits
amp, vreg, vex, treg, tex, checks = sv.simulate()    # gate-level arithmetic, every k != 0
err = np.linalg.norm(amp - exact_unit(vex), axis=1).max()
print(checks, f"{err:.2e}", sv.toffoli())
```

## Scripts and the paper

| script | reproduces | time |
|---|---|---|
| `example_poisson_isometry.py` | Theorem 1 and Table 1 (cost by stage at N = 1024, F = 45); Sec. 4, the measured error in the 42 cases and the bound of Corollary 6; the check of the whole block against the Q4 assembly | a few minutes |
| `check_rotation.py` | Lemma 2 (rotation mode): the bound and the residual angle on random angles | seconds |
| `check_lemma.py` | Lemma 3 (one vectoring): every inequality of the proof on random integer vectorings | seconds |
| `verify_counts.py` | Sec. 5: Toffoli and work-qubit counts of Parts 1 and 2 asserted against the gate lists; work register returns to zero | about 10 seconds |

The class behind `poisson_isometry` is more general: `SymbolVector` prepares
v(j)/||v(j)|| for v(j) = C t(j), t(j) = (1, cos theta_1, sin theta_1, ...),
theta_r = pi (p_r . j + q_r)/N, and `TreeCircuit` runs Part 2 alone on given
values. These scripts exercise the general class; they are not part of the
paper.

| script | checks |
|---|---|
| `table_cost.py` | Part 2 alone on random vectors of length 2 to 16 |
| `table_examples.py` | whole circuit for the gradient symbol and the stencil rows |
| `example_laplacian_symbol.py` | gradient symbol of the finite-difference Laplacian, d = 2 and 3 |
| `example_stencil_rows.py` | row states of a variable-coefficient stencil |
| `check_options.py` | known-angle nodes and the parameters g, G |

## Options

Two options reduce the cost for a specific application; `poisson_isometry` uses both.

1. **Known-angle nodes:** `known={(q, p): theta_c}` declares that node
   `(q, p)` has the same angle for every index, that is,
   rho_right = tan(theta_c) rho_left, with cos(theta_c) >= 1/sqrt 2 (order the
   children so). The node needs no vectoring: its output is
   (K / cos theta_c) x_left, formed by shifts and additions over the signed
   digits of the constant, and its rotation is the fixed R(theta_c). At the
   leaf level the right register is not read. `signed=[(q, p), ...]` gives
   such a leaf node the sign flag when its left entry may be negative.
2. **Guard and integer bits:** `g` (default ceil(log2 F)) and `G` (default
   ceil(log2(K^L sqrt m)) + 2) are parameters. The simulation reports
   `no_overflow`, which checks every node output.

```python
tc = TreeCircuit(L=2, F=45, g=0, G=4, known={(1, 1): np.pi / 4})
print(tc.tree.toffoli(), tc.tree.rotation_counts())   # 18624, 90 ctrl Ry, 3 fixed, 1 flag Z
```

## Package

- `binary_to_amplitude/poisson.py`: the configuration of the paper
  (`poisson_isometry`): the matrix C of c(k), g = 0, G = 4, and the known-angle
  node (pi/4, 1/sqrt3).
- `binary_to_amplitude/symbol.py`: Part 1 and the whole circuit for
  v(j) = C t(j) (`SymbolVector`), with the closed-form counts and the bounds of
  Lemma 2 and Corollary 6 of the paper.
- `binary_to_amplitude/tree.py`: Part 2, the circuit R (`Tree`), and a driver
  on given values (`TreeCircuit`).
- `binary_to_amplitude/revcirc.py`: reversible fixed-point arithmetic (Gidney
  adders, CORDIC rotation and vectoring modes) and a bit-packed simulator that
  runs all inputs at once, from
  [QuantumHomogenizationPoisson](https://github.com/UW-ERSL/QuantumHomogenizationPoisson).

## Scope

1. **Two-dimensional Q4 Poisson problem:** the paper covers this case only; the
   fixed unitary T, the phase, and the zero mode are in the Poisson paper.
2. **Scalar case:** one vector per index. Vector-valued problems, with several
   orthonormal columns per index (as in elasticity), are not covered.
3. **Rotation stage:** counted (`Tree.rotation_counts`) but not simulated at
   gate level; the simulation applies the angles that the decision bits encode.
4. **Phases:** complex phases are applied separately.
5. **Entries:** Part 1 builds trigonometric polynomials of the index; other
   entries need their own circuit, after which Part 2 applies unchanged.
6. **v = 0:** the caller supplies a flag.
7. **Pebbling:** the work register keeps every shifted copy until the
   uncomputation; no pebbling is implemented.

## License

Apache 2.0.
