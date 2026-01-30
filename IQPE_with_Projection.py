import numpy as np
from qiskit.circuit.library import PauliEvolutionGate, PhaseGate, TGate, XGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import eigh

from projection_algorithm import driver
from iqpe import iqpe


# 1. Textbook examples

iqpe(PhaseGate(2 * np.pi * (1/2**2 + 1/2**3 + 1/2**5)), XGate(), 5)
# [1, 0, 1, 1, 0]

iqpe(TGate(), XGate(), 3)
# [1, 0, 0]


# 2. Simple Ising Hamiltonian

H = SparsePauliOp(["ZI", "IZ", "ZZ"], coeffs=[1.42, 2.19, 2.65])
e, v = np.linalg.eig(H)
# e = array([ 6.26+0.j, -3.42+0.j, -1.88+0.j, -0.96+0.j])

iqpe(
    U=PauliEvolutionGate(H, time=2 * np.pi / 2**3, synthesis=SuzukiTrotter(reps=2)),
    initial_state=v[:, 0],
    num_bits=5,
)
# 5: [1, 1, 1, 0, 0]
# 9: [1, 1, 1, 1, 0, 1, 1, 0, 0]

iqpe(
    U=PauliEvolutionGate(H, time=2 * np.pi / 2**2),
    initial_state=v[:, 1],
    num_bits=5,
)
# 5:  [1, 1, 0, 1, 1]
# 9:  [0, 1, 1, 0, 1, 1, 0, 1, 1]


H2 = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
e, v = eigh(H2)
# e = array([-1.5, -0.5,  0.5,  1.5])

iqpe(
    U=PauliEvolutionGate(H2, time=2 * np.pi / 2**2),
    initial_state=v[:, 0],
    num_bits=5,
)
# 5: [0, 0, 1, 1, 0]
# 9: [0, 0, 0, 0, 0, 0, 1, 1, 0]


# 3. Isotropic Heisenberg Hamiltonian

def get_isotropic_1d_heisenberg_hamiltonian(num_qubits, J=1.0):
    """Constructs an Isotropic Heisenberg Hamiltonian for a 1D chain.

    H = J * sum_{i} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    """
    ham_list = []
    # Iterate through adjacent pairs (nearest-neighbors)
    for i in range(num_qubits - 1):
        # Isotropic coupling: same J for XX, YY, and ZZ
        ham_list.append(("XX", [i, i+1], J))
        ham_list.append(("YY", [i, i+1], J))
        ham_list.append(("ZZ", [i, i+1], J))

    return SparsePauliOp.from_sparse_list(ham_list, num_qubits=num_qubits)


heisenberg_h = get_isotropic_1d_heisenberg_hamiltonian(num_qubits=2, J=1.3)
e, v = eigh(heisenberg_h)
# e = array([-3.9,  1.3,  1.3,  1.3])

iqpe(
    U=PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=v[:, 0],
    num_bits=5,
)
# 5: [1, 1, 1, 1, 1]
# 9: [1, 1, 0, 0, 1, 1, 1, 1, 1]

iqpe(
    U=PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=v[:, 1],
    num_bits=9,
)
# 5: [0, 1, 1, 0, 1]
# 9: [0, 1, 0, 1, 1, 0, 1, 0, 1]


# Combine with Projection Algorithm.

e, v = driver(heisenberg_h, 1e-20)

iqpe(
    U=PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=v,
    num_bits=9,
)
# [1, 1, 0, 0, 1, 1, 1, 1, 1]
