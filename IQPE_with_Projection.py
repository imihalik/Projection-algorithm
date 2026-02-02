import numpy as np
from qiskit.circuit.library import PauliEvolutionGate, PhaseGate, TGate, XGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import eigh

from projection_algorithm import driver
from iqpe import iqpe
from utils import bitstring_to_eigenvalue, get_isotropic_1d_heisenberg_hamiltonian


# 1. Textbook examples

iqpe(PhaseGate(2 * np.pi * (1/2**2 + 1/2**3 + 1/2**5)), XGate(), 5)
# 0.40625, [0, 1, 1, 0, 1]

iqpe(TGate(), XGate(), 3)
# 0.125, [0, 0, 1]


# 2. Simple Ising Hamiltonian
H = SparsePauliOp(["ZI", "IZ", "ZZ"], coeffs=[1.42, 2.19, 2.65])
e, v = np.linalg.eig(H)
# e = array([ 6.26+0.j, -3.42+0.j, -1.88+0.j, -0.96+0.j])

def imeasure_H(precision, init_state):
    for n in range(10):
        print(
            iqpe(
                U=PauliEvolutionGate(H, time=2 * np.pi / 2**n),
                initial_state=init_state,
                num_bits=precision,
            )
        )

t = 2 * np.pi / 2**4
iqpe(
    U=PauliEvolutionGate(H, time=t, synthesis=SuzukiTrotter(reps=2)),
    initial_state=v[:, 0],
    num_bits=5,
)
#  [1, 0, 0, 1, 1]
bitstring_to_eigenvalue("10011", t)
# 6.5

iqpe(
    U=PauliEvolutionGate(H, time=t, synthesis=SuzukiTrotter(reps=2)),
    initial_state=v[:, 0],
    num_bits=9,
)
# [1, 0, 0, 1, 1, 1, 0, 0, 0]
bitstring_to_eigenvalue("100111000", t)
# 6.25

t = 2 * np.pi / 2**3
iqpe(
    U=PauliEvolutionGate(H, time=t),
    initial_state=v[:, 1],
    num_bits=5,
)
# num_bits 5: -3.35
# num_bits 9: -3.421875


# 3. Isotropic Heisenberg Hamiltonian
heisenberg_h = get_isotropic_1d_heisenberg_hamiltonian(num_qubits=2, J=1.3)
e, v = eigh(heisenberg_h)
# e = array([-3.9,  1.3,  1.3,  1.3])

t = 2 * np.pi / 2**2
iqpe(
    U=PauliEvolutionGate(heisenberg_h, time=t),
    initial_state=v[:, 0],
    num_bits=5,
)
# num_bits 5: 0.96875 -> -3.875
# num_bits 9: 0.974609375 -> --3.8984375

iqpe(
    U=PauliEvolutionGate(heisenberg_h, time=t),
    initial_state=v[:, 1],
    num_bits=9,
)
# num_bits 5: 0.6875 -> 1.25
# num_bits 9: 0.67578125 -> 1.296875


# 4. Combine with Projection Algorithm.
e, v = driver(heisenberg_h, 1e-20)

iqpe(
    U=PauliEvolutionGate(heisenberg_h, time=t),
    initial_state=v,
    num_bits=9,
)
# 0.974609375 -> -3.8984375
