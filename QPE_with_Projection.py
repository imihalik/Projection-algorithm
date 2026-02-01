import numpy as np
from qiskit.circuit.library import PauliEvolutionGate, PhaseGate, TGate, XGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import eigh

from projection_algorithm import driver
from qpe import qpe
from utils import get_isotropic_1d_heisenberg_hamiltonian, phase_to_eigenvalue


####################################################################################################
# Demo

# 1. Textbook examples

qpe(3, PhaseGate(2 * np.pi * (1 / 4 + 1 / 8)), XGate(), to_phase=False)
# {'011': 1.0}

qpe(3, TGate(), XGate(), to_phase=False)
# {'001': 1.0}


# 2. Simple Ising Hamiltonian
H = SparsePauliOp(["ZI", "IZ", "ZZ"], coeffs=[1.42, 2.19, 2.65])
e, v = np.linalg.eig(H)
# e: array([ 6.26+0.j, -3.42+0.j, -1.88+0.j, -0.96+0.j])

# Works well if t is small enough.
t = 2 * np.pi / 2**3
U = PauliEvolutionGate(H, time=t, synthesis=SuzukiTrotter(reps=2))

qpe(5, U, initial_state=list(v[:,0]))
# {0.21875: 0.994140625, 0.1875: 0.001953125, 0.25: 0.00146484375}
phase_to_eigenvalue(0.21875, t)
# -1.75 (X), 6.25 (O)

qpe(9, U, initial_state=list(v[:,0]))
# {0.216796875: 0.64306640625, 0.21875: 0.20166015625, 0.21484375: 0.0439453125}
# -> -1.734375 (X), 6.265625 (O)

# Negative eigenvalue
t = 2 * np.pi / 2**2
U = PauliEvolutionGate(H, time=t)
qpe(5, U, initial_state=list(v[:,1]))
# {0.84375: 0.640625, 0.875: 0.20263671875, 0.8125: 0.0400390625}
# -> -3.375 (O), 0.625 (X)

qpe(9, U, initial_state=list(v[:,1]))
# {0.85546875: 0.814453125, 0.853515625: 0.099609375, 0.857421875: 0.02880859375}
# -3.4218750000000004 (O), 0.578125 (X)

# If | H t | > 2π, the part of the phase larger than 2π is lost. I.e., either we at least know the
# range of the eigenvalue beforehand, or we choose t small "enough".
t = 2 * np.pi
U = PauliEvolutionGate(H, time=t)
qpe(5, U, initial_state=list(v[:,0]))
# {0.75: 0.703125, 0.71875: 0.162109375, 0.78125: 0.04736328125}
# -0.75 (X), 0.25 (X)

# Mixed state
t = 2 * np.pi / 2**3
U = PauliEvolutionGate(H, time=t)
qpe(5, U, initial_state=list((v[:,0] + v[:,1]) / np.sqrt(2)))
# {0.21875: 0.484375, 0.4375: 0.36669921875, 0.40625: 0.083984375}
# -> 6.25 for the 0 state, -3.5 for the 1 state

# More examples
H2 = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
eigenvalues, eigenvectors = eigh(H2)
# eigenvalues = -1.5, -0.5, 0.5, 1.5
t = 2 * np.pi / 2**2
qpe(6, PauliEvolutionGate(H2, time=t), initial_state=list(eigenvectors[:,0]))
# {0.375: 1.0} -> -1.5

H6 = SparsePauliOp(
    ["IIIIII", "ZZZIII", "IZZZII", "IIZZZI", "IIIZZZ"], coeffs=[-2.2, 0.33, 0.7, 0.52, 0.82]
)
eigenvalues, eigenvectors = eigh(H6)
# eigenvalues = -4.57, ...
t = 2 * np.pi / 2**3
qpe(7, PauliEvolutionGate(H6, time=t), initial_state=list(eigenvectors[:,0]))
# {0.5703125: 0.95751953125, 0.578125: 0.01318359375, 0.5625: 0.01171875} -> -4.5625


# 3. Isotropic Heisenberg Hamiltonian
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


# Combine with Projection Algorithm.
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
