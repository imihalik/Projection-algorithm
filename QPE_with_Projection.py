import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, PhaseGate, TGate, XGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import StatevectorSimulator
from scipy.linalg import eigh

from projection_algorithm import driver
from qpe import qpe


####################################################################################################
# Demo

# 1. Textbook examples

qpe(3, PhaseGate(2 * np.pi * (1 / 4 + 1 / 8)), XGate())
# {'011': 1024}

qpe(3, TGate(), XGate())
# {'001': 1024}

qc = QuantumCircuit(1)
qc.append(XGate(), [0])

simulator = StatevectorSimulator()
statevector = simulator.run(transpile(qc, simulator)).result().get_statevector()
initial_state = statevector / np.linalg.norm(statevector)

qpe(5, PhaseGate(2 * np.pi * (1 / 4 + 1 / 8 + 1 / 16)), initial_state=initial_state)
# {'01110': 1024}


# 2. Simple Ising Hamiltonian

H = SparsePauliOp(["ZI", "IZ", "ZZ"], coeffs=[1.42, 2.19, 2.65])
e, v = np.linalg.eig(H)
# e: array([ 6.26+0.j, -3.42+0.j, -1.88+0.j, -0.96+0.j])

# Works well if t is small enough.
U = PauliEvolutionGate(H, time=2 * np.pi / 2**3, synthesis=SuzukiTrotter(reps=2))

qpe(5, U, initial_state=list(v[:,0]))
# {'01001': 1, '00110': 4, '00100': 1, '01101': 1, '01000': 7, '00111': 2034}
# -> 2**3 * (1 - (1/2**3 + 1/2**4 + 1/2**5)) = 6.25
qpe(9, U, initial_state=list(v[:,0]))
# {'001101111': 1343, ...}
# -> 2**3 * (1 - (1/2**3 + 1/2**4 + 1/2**6 + 1/2**7 + 1/2**8 + 1/2**9)) = 6.265625

# Negative eigenvalue
U = PauliEvolutionGate(H, time=2 * np.pi / 2**2)
qpe(5, U, initial_state=list(v[:,1]))
# {'11011': 1238, ...} -> -2**2 * (1/2 + 1/2**2 + 1/2**4 + 1/2**5) = -3.375
qpe(9, U, initial_state=list(v[:,1]))
# {'110110110': 1701, ...}
# -> -2**2 * (1/2 + 1/2**2 + 1/2**4 + 1/2**5 + 1/2**7 + 1/2**8) = -3.421875

# If | H t | > 2π, the part of the phase larger than 2π is lost. I.e., either we at least know the
# range of the eigenvalue beforehand, or we choose t small "enough".
U = PauliEvolutionGate(H, time=2 * np.pi / 2)
qpe(5, U, initial_state=list(v[:,0]))
# {'11100': 1873, ...} -> 2 * (1 - (1/2 + 1/2**2 + 1/2**3)) = 0.25

# Mixed state
U = PauliEvolutionGate(H, time=2 * np.pi / 2**3)
qpe(5, U, initial_state=list((v[:,0] + v[:,1]) / np.sqrt(2)))
# {'00111': 1035, '01110': 704, ...}

# More examples
H2 = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
eigenvalues, eigenvectors = eigh(H2)
# eigenvalues = -1.5, -0.5, 0.5, 1.5
qpe(6, PauliEvolutionGate(H2, time=2 * np.pi / 2**2), initial_state=list(eigenvectors[:,0]))
# {'011000': 2048} -> -1.5

H3 = SparsePauliOp(["III", "ZZI", "IZZ"], coeffs=[-2, 0.5, 0.5])
eigenvalues, eigenvectors = eigh(H3)
eigenvalues = -3, ...
qpe(6, PauliEvolutionGate(H3, time=2 * np.pi / 2**2), initial_state=list(eigenvectors[:,0]))
# {'110000': 2048} -> -3

H4 = SparsePauliOp(["IIII", "ZZII", "IZZI", "IIZZ"], coeffs=[-2, 0.5, 0.5, 0.5])
eigenvalues, eigenvectors = eigh(H4)
eigenvalues = -3.5, ...
qpe(6, PauliEvolutionGate(H4, time=2 * np.pi / 2**2), initial_state=list(eigenvectors[:,0]))
# {'111000': 2048} -> -3.5

H6 = SparsePauliOp(
    ["IIIIII", "ZZZIII", "IZZZII", "IIZZZI", "IIIZZZ"], coeffs=[-2.2, 0.33, 0.7, 0.52, 0.82]
)
eigenvalues, eigenvectors = eigh(H6)
eigenvalues = -4.57, ...
qpe(7, PauliEvolutionGate(H6, time=2 * np.pi / 2**3), initial_state=list(eigenvectors[:,0]))
# {'1001001': 1943, ...} -> -2**3 * (1/2 + 1/2**4 + 1/2**7) = -4.5625


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

# QPE correctly estimates the ground state energy -3.9 for 2-qubit Heisenberg model.
heisenberg_h = get_isotropic_1d_heisenberg_hamiltonian(num_qubits=2, J=1.3)
eigenvalues, eigenvectors = eigh(heisenberg_h)
# eigenvalues = -3.9,  1.3,  1.3,  1.3

qpe(
    5,
    PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=list(eigenvectors[:,0]),
)
# {'11111': 1794,, ...} -> -2**2 * (1/2 + 1/2**2 + 1/2**3 + 1/2**4 + 1/2**5) = -3.875
qpe(
    9,
    PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=list(eigenvectors[:,0]),
)
# {'111110011': 1776, ...}
# -> -2**2 * (1/2 + 1/2**2 + 1/2**3 + 1/2**4 + 1/2**5 + 1/2**8 + 1/2**9) = -3.8984375

qpe(
    5,
    PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=list(eigenvectors[:,1]),
)
# {'10110': 1130, ...} -> 2**2 * (1 - (1/2 + 1/2**3 + 1/2**4)) = 1.25
qpe(
    9,
    PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=list(eigenvectors[:,1]),
)
# {'101011010': 1156, ...} -> 2**2 * (1 - (1/2 + 1/2**3 + 1/2**5 + 1/2**6 + 1/2**8)) = 1.296875


# Combine with Projection Algorithm.

exp_vals, initial_state = driver(heisenberg_h, 1e-20)
qpe(
    9,
    PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=initial_state,
)
# {'111110011': 1781, ...}
# -> -2**2 * (1/2 + 1/2**2 + 1/2**3 + 1/2**4 + 1/2**5 + 1/2**8 + 1/2**9) = -3.8984375


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
