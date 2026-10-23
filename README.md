# BinaryToAmplitude

Code for the note *Binary-to-amplitude conversion of a vector by a tree of
CORDIC vectorings* (K. Suresh, UW-Madison). Every number in the note is
reproduced by the scripts below.

Given an index register |j> and fixed-point registers holding an unnormalized
real vector v(j) with m entries, the circuit R prepares v(j)/||v(j)|| on
log2(m) qubits for all j at once, with no division, square root, or angle
register. It runs m - 1 CORDIC vectorings arranged in a binary tree, and the
decision bits of each vectoring control the rotations directly. Its cost
depends on m and the number of fractional bits F, not on the size of the index
register.

## Install

    git clone https://github.com/UW-ERSL/BinaryToAmplitude
    cd BinaryToAmplitude
    pip install -e .

Only numpy is required.

## Quick start

```python
import numpy as np
from binary_to_amplitude import TreeCircuit, exact_unit

F = 32
tc = TreeCircuit(L=2, F=F)                     # m = 4 entries, g = ceil(log2 F) guard bits
V = np.round(np.random.default_rng(0).uniform(-1, 1, (1000, 4)) * 2**F) / 2**F
amp, checks = tc.simulate(V)                   # gate-level arithmetic, every input
err = np.linalg.norm(amp - exact_unit(V), axis=1).max()
print(checks, f"{err:.2e}", tc.tree.toffoli())
```

## Scripts and the note

| script | reproduces | time |
|---|---|---|
| `verify_counts.py` | Sec. 5: Toffoli and work-qubit counts asserted against the gate lists; work register returns to zero | seconds |
| `table_cost.py` | Table 1 | seconds |
| `check_lemma.py` | Lemma 2: every inequality of the proof on random integer vectorings | seconds |
| `example_laplacian_symbol.py` | Sec. 6, Example 1 (gradient symbol of the finite-difference Laplacian, d = 2 and 3) | about 1 minute |
| `example_stencil_rows.py` | Sec. 6, Example 2 (row states of a variable-coefficient stencil) | seconds |
| `check_options.py` | Options: known-angle nodes and the parameters g, G (counts, accuracy, overflow detection) | about 1 minute |

## Options

Two options reduce the cost for a specific application.

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

- `binary_to_amplitude/tree.py`: the circuit R (`Tree`), a driver on given
  values (`TreeCircuit`), and the Laplacian-symbol example with its caller
  circuit A (`SymbolCircuit`).
- `binary_to_amplitude/revcirc.py`: reversible fixed-point arithmetic (Gidney
  adders, CORDIC rotation and vectoring modes) and a bit-packed simulator that
  runs all inputs at once, from
  [QuantumHomogenizationPoisson](https://github.com/UW-ERSL/QuantumHomogenizationPoisson).

## Scope

1. **Rotation stage:** counted (`Tree.rotation_counts`) but not simulated at
   gate level; the simulation applies the angles that the decision bits encode.
2. **Phases:** complex phases are applied separately and are not part of R.
3. **The caller's circuit A:** supplied by the caller; only the Laplacian
   example builds one.
4. **v = 0:** the caller supplies a flag.
5. **Pebbling:** the work register keeps every shifted copy until the
   uncomputation; no pebbling is implemented.

## License

Apache 2.0.
