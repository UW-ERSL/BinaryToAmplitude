"""Secs. 2 and 3, the worked example: k = (3, 1), N = 16, F = 12, registers of
16 bits. Part 1: the cosines and sines of theta_+- and the bit strings of a, b,
zeta. Part 2: the node outputs and decision bits of the tree, the angles they
encode, and the prepared amplitudes against c/||c||."""
import numpy as np
from multiplexed_state_preparation import poisson_isometry
from multiplexed_state_preparation.revcirc import BitSim

n, F, k = 4, 12, (3, 1)
sv, N, u = poisson_isometry(n, F), 2 ** n, 2.0 ** -F
t = sv.tree
sim = BitSim(sv.c.n, 1)
for kj, val in zip(sv.k, k):
    sim.set_int(kj, np.array([val]))
bits = lambda r: "".join(str(int(sim.get_bits(q)[0])) for q in reversed(r))

# ----- Part 1
sim.run(sv.c.gates[:t.gate_range[0]])
f1, f2 = np.pi * k[0] / N, np.pi * k[1] / N
th = [f1 + f2, f1 - f2]
print("Part 1")
for r, (cs, sn) in enumerate(zip(sv.trig[0::2], sv.trig[1::2])):
    print(f"  theta_{'+-'[r]}: (cos, sin) = ({sim.get_int(cs)[0] * u:.5f}, {sim.get_int(sn)[0] * u:.5f}),"
          f" exact ({np.cos(th[r]):.5f}, {np.sin(th[r]):.5f})")
a, b, z = np.sin(f1) * np.cos(f2), np.cos(f1) * np.sin(f2), np.sin(f1) * np.sin(f2)
for name, r, ex in (("a", sv.rows[0], a), ("b", sv.rows[1], b), ("zeta", sv.rows[2], z)):
    print(f"  {name:4s} = {sim.get_int(r)[0] * u:.5f}: {bits(r)}, exact {ex:.5f}")

# ----- Part 2
print("Part 2")
pos = t.gate_range[0]
for end, node, reg in t.marks:
    sim.run(sv.c.gates[pos:end])
    pos = end
    print(f"  node {node}: output {sim.get_int(reg)[0] * u:.5f}: {bits(reg)}")
sim.run(sv.c.gates[pos:t.gate_range[1]])
K = np.prod(np.sqrt(1 + 4.0 ** -np.arange(t.I)))
r_ = np.hypot(a, b)
print(f"  exact K r = {K * r_:.5f}, sqrt(2/3) K zeta = {np.sqrt(2/3) * K * z:.5f},"
      f" K^2 ||c|| = {K * K * np.sqrt(a*a + b*b + 2*z*z/3):.5f}")
ang = t.angles(sim)
exact = {(1, 0): np.arctan2(b, a), (1, 1): np.pi / 4, (0, 0): np.arctan2(np.sqrt(2/3) * z, r_)}
for node, bb in t.dbits.items():
    print(f"  node {node}: decision bits d_0..d_{t.I - 1} = {''.join(str(int(sim.get_bits(q)[0])) for q in bb)}")
for node in sorted(ang):
    print(f"  node {node}: angle {ang[node][0]:.5f}, exact {exact[node]:.5f}")
for f in t.flags.values():
    print(f"  sign flag f = {int(sim.get_bits(f)[0])}")
amp = t.amplitudes(ang, 1)[0]
c = np.array([a, b, z / np.sqrt(3), z / np.sqrt(3)])
print("  amplitudes", np.round(amp, 5), "\n  exact     ", np.round(c / np.linalg.norm(c), 5),
      f"\n  error {np.linalg.norm(amp - c / np.linalg.norm(c)):.1e}")
