import numpy as np
from qiskit.circuit.library import PauliEvolutionGate, PhaseGate, TGate, XGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import eigh

from projection_algorithm import driver
from qpe import qpe
from utils import bitstring_to_eigenvalue, get_isotropic_1d_heisenberg_hamiltonian


####################################################################################################
# Demo

# 1. Textbook examples

qpe(3, PhaseGate(2 * np.pi * (1 / 4 + 1 / 8)), XGate(), to_phase=False)
# {'011': 1.0}

qpe(3, TGate(), XGate(), to_phase=False)
# {'001': 1.0}


# 2. Full workflow with Simple Ising Hamiltonian

H = SparsePauliOp(["ZI", "IZ", "ZZ"], coeffs=[1.42, 2.19, 2.65])
e, v = np.linalg.eig(H)
# e: array([ 6.26+0.j, -3.42+0.j, -1.88+0.j, -0.96+0.j])

def measure_H(precision, init_state):
    for n in range(10):
        print(
            qpe(
                precision,
                PauliEvolutionGate(H, time=2 * np.pi / 2**n),
                initial_state=init_state,
            )
        )

# positive eigenvalue
measure_H(7, list(v[:,0]))
# {'1011111': 0.76904296875, '1011110': 0.11767578125, '1100000': 0.03662109375}
# {'1101111': 0.63134765625, '1110000': 0.2080078125, '1101110': 0.046875}
# {'0111000': 0.71630859375, '0110111': 0.146484375, '0111001': 0.0380859375}
# {'0011100': 0.8984375, '0011011': 0.0380859375, '0011101': 0.02294921875}
# {'1001110': 0.974609375, '1001111': 0.0087890625, '1001101': 0.00830078125}
# {'1100111': 0.99755859375, '1100100': 0.0009765625, '1101100': 0.00048828125}
# {'1110011': 0.44140625, '1110100': 0.3662109375, '1110101': 0.0439453125}
# {'1111010': 0.79638671875, '1111001': 0.09375, '1111011': 0.03564453125}
# {'1111101': 0.93994140625, '1111100': 0.02392578125, '1111110': 0.01318359375}
# {'1111110': 0.5205078125, '1111111': 0.30517578125, '1111101': 0.04345703125}

# As t decreases, bitstring approaches 111..., signifying the phase is negative. Choose anything
# larger than 0.5 in the increasing sequence.

bitstring_to_eigenvalue("1100111", 2 * np.pi / 2**5)
# 6.25

# Similarly for precision=5: 6.25, precision=9: 6.265625

# negative eigenvalue
measure_H(7, list(v[:,1]))
# {'0110110': 0.833984375, '0110101': 0.07373046875, '0110111': 0.03466796875}
# {'1011011': 0.9580078125, '1011010': 0.01513671875, '1011100': 0.01123046875}
# {'1101101': 0.505859375, '1101110': 0.31005859375, '1101100': 0.0498046875}
# {'0110111': 0.7578125, '0110110': 0.123046875, '0111000': 0.0400390625}
# {'0011011': 0.63720703125, '0011100': 0.2060546875, '0011010': 0.044921875}
# {'0001110': 0.6845703125, '0001101': 0.15673828125, '0001111': 0.04296875}
# {'0000111': 0.91064453125, '0000110': 0.041015625, '0001000': 0.017578125}
# {'0000011': 0.52490234375, '0000100': 0.2841796875, '0000010': 0.05029296875}
# {'0000010': 0.75146484375, '0000001': 0.12646484375, '0000011': 0.03564453125}
# {'0000001': 0.9267578125, '0000000': 0.0322265625, '0000010': 0.01513671875}

# As t decreases, bitstring approaches 000..., signifying the phase is positive. Choose anything
# smaller than 0.5 in the decreasing sequence.

bitstring_to_eigenvalue("0110111", 2 * np.pi / 2**3)
# -3.4375

# Similarly for precision=5: -3.375, precision=9: -3.421875


# 3. More examples

H2 = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
eigenvalues, eigenvectors = eigh(H2)
# eigenvalues = -1.5, -0.5, 0.5, 1.5
t = 2 * np.pi / 2**2
qpe(
    6,
    PauliEvolutionGate(H2, time=t, synthesis=SuzukiTrotter(reps=2)),
    initial_state=list(eigenvectors[:,0]),
)
# {0.375: 1.0} -> -1.5

H6 = SparsePauliOp(
    ["IIIIII", "ZZZIII", "IZZZII", "IIZZZI", "IIIZZZ"], coeffs=[-2.2, 0.33, 0.7, 0.52, 0.82]
)
eigenvalues, eigenvectors = eigh(H6)
# eigenvalues = -4.57, ...
t = 2 * np.pi / 2**3
qpe(7, PauliEvolutionGate(H6, time=t), initial_state=list(eigenvectors[:,0]))
# {0.5703125: 0.95751953125, 0.578125: 0.01318359375, 0.5625: 0.01171875} -> -4.5625


# 4. Isotropic Heisenberg Hamiltonian

# QPE correctly estimates the ground state energy -3.9 for 2-qubit Heisenberg model.
heisenberg_h = get_isotropic_1d_heisenberg_hamiltonian(num_qubits=2, J=1.3)
eigenvalues, eigenvectors = eigh(heisenberg_h)
# eigenvalues = -3.9,  1.3,  1.3,  1.3

t = 2 * np.pi / 2**2
qpe(
    5,
    PauliEvolutionGate(heisenberg_h, time=t),
    initial_state=list(eigenvectors[:,0]),
)
# {0.96875: 0.87890625, 0.0: 0.0498046875, 0.9375: 0.0302734375} -> -3.875
qpe(
    9,
    PauliEvolutionGate(heisenberg_h, time=t),
    initial_state=list(eigenvectors[:,0]),
)
# {0.974609375: 0.8857421875, 0.9765625: 0.0478515625, 0.97265625: 0.02783203125} -> -3.8984375

qpe(
    5,
    PauliEvolutionGate(heisenberg_h, time=t),
    initial_state=list(eigenvectors[:,1]),
)
# {0.6875: 0.58154296875, 0.65625: 0.248046875, 0.71875: 0.04638671875} -> 1.25
qpe(
    9,
    PauliEvolutionGate(heisenberg_h, time=t),
    initial_state=list(eigenvectors[:,1]),
)
# {0.67578125: 0.55224609375, 0.673828125: 0.26806640625, 0.677734375: 0.04931640625} -> 1.296875


# 5. Combine with Projection Algorithm.
_, initial_state = driver(heisenberg_h, 1e-20)
t = 2 * np.pi / 2**2
qpe(
    9,
    PauliEvolutionGate(heisenberg_h, time=t),
    initial_state=initial_state,
)
# {0.974609375: 0.8662109375, 0.9765625: 0.06298828125, 0.97265625: 0.0244140625} -> -3.8984375


####################################################################################################
# Q's

# - The above works well. But it requires knowledge of eigenvalues and eigenvectors of the
# Hamiltonian beforehand.
# - Assuming we don't know them beforehand, we should choose t small enough, meaning we'd need
# large enough precision.
# - If t is too small, would it be practically possible to measure?
# - Initial state prepared by Projection Algo seems to be the issue. Still looking.

# Truncation / Bad estimation

# 4-qubit Heisenberg Hamiltonian
heisenberg_h = get_isotropic_1d_heisenberg_hamiltonian(num_qubits=4)
eigenvalues, eigenvectors = eigh(heisenberg_h)
eigenvalues = -6.46410162, -3.82842712, ...
qpe(
    8,
    PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**3),
    initial_state=list(eigenvectors[:,0]),
)
# {'10100000': 1932, '00100000': 116}

# TFIM Hamiltonian
def get_tfim_hamiltonian(num_qubits, J=1.0, h=0.5):
    # (Pauli string, qubit indices, coefficient)
    interactions = [("ZZ", [i, i+1], -J) for i in range(num_qubits - 1)]
    fields = [("X", [i], -h) for i in range(num_qubits)]

    # Combine into a single list and build the operator
    return SparsePauliOp.from_sparse_list(interactions + fields, num_qubits=num_qubits)

h2 = get_tfim_hamiltonian(2)
eigenvalues, eigenvectors = eigh(h2)
eigenvalues = -1.41421356, ...
qpe(6, PauliEvolutionGate(h2, time=2 * np.pi / 2**2), initial_state=list(eigenvectors[:,0]))
# {'100000': 1025, '000000': 1023}


h3 = get_tfim_hamiltonian(3)
eigenvalues, eigenvectors = eigh(h3)
eigenvalues = -2.40321193, ...
qpe(6, PauliEvolutionGate(h3, time=2 * np.pi / 2**2), initial_state=list(eigenvectors[:,0]))
# {'011000': 1292, '111000': 756}
