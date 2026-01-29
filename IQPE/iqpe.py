import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import StatevectorSimulator
from scipy.linalg import eigh

from tests.assets import v3


def iqpe_projected(U, initial_state, num_bits: int, seed=None, verbose=True):
    if isinstance(initial_state, Statevector):
        psi = initial_state
    else:
        psi = Statevector(initial_state)
    n_sys = psi.num_qubits

    bits = []
    omega_coef = 0.0

    # Iterate from most significant bit down to least
    for k in range(num_bits, 0, -1):
        omega_coef /= 2.0
        power = 2 ** (k - 1)

        qc = QuantumCircuit(1 + n_sys)
        # 1. Ancilla in |+>
        qc.h(0)
        # 2. Controlled-U^(2^{k-1})
        qc.append(U.control().power(power), [0] + list(range(1, 1 + n_sys)))
        # 3. Feedback phase
        theta = -2 * np.pi * omega_coef
        # if abs(theta) > 1e-12:
        qc.p(theta, 0)
        # 4. Final H on ancilla
        qc.h(0)

        # 5. Evolve |0>_anc ⊗ |psi>
        full_state = Statevector.from_label("0").tensor(psi)
        full_state = full_state.evolve(qc)

        # 6. Get ancilla marginal probabilities
        probs = np.asarray(full_state.probabilities([0]), dtype=float)
        s = probs.sum()
        if s <= 0 or not np.isfinite(s):
            p0, p1 = 0.5, 0.5
        else:
            p0, p1 = probs / s

        # 7. Majority vote (deterministic)
        bit = 1 if p1 > p0 else 0

        # >>> these two lines were missing <<<
        bits.append(bit)
        omega_coef += bit / 2.0

        if verbose:
            print(f"Iteration k={k}: P(0)={p0:.4f}, P(1)={p1:.4f}, measured={bit}")

    phi_est = omega_coef % 1.0
    return phi_est, bits


def unwrap_phase(phi_wrapped: float) -> float:
    """Map a phase in [0,1) to (-0.5, 0.5] for energy reconstruction."""
    return phi_wrapped if phi_wrapped <= 0.5 else phi_wrapped - 1.0


def energy_from_phase(phi_wrapped: float, t_evolution: float) -> float:
    """
    Given φ in [0,1) for U = exp(-i H t),
    φ ≈ -E t / (2π) mod 1 → E ≈ -2π * φ_unwrapped / t.
    """
    phi_cont = unwrap_phase(phi_wrapped)
    return -2 * np.pi * phi_cont / t_evolution



H = SparsePauliOp(["ZI", "IZ", "ZZ"], coeffs=[1.42, 2.19, 2.65])
e, v = np.linalg.eig(H)
# e = array([ 6.26+0.j, -3.42+0.j, -1.88+0.j, -0.96+0.j])

iqpe_projected(
    U=PauliEvolutionGate(H, time=2 * np.pi / 2**3),
    initial_state=v[:, 0],
    num_bits=5,
    seed=42,
    verbose=True,
)
# 5: [1, 1, 1, 0, 0]
# 9: [1, 1, 1, 1, 0, 1, 1, 0, 0]

iqpe_projected(
    U=PauliEvolutionGate(H, time=2 * np.pi / 2**3),
    initial_state=v[:, 1],
    num_bits=5,
    seed=42,
    verbose=True,
)
# 5:  [0, 0, 1, 1, 1] -> -7
# 9:  [0, 1, 0, 1, 1, 0, 0, 0, 1] -> -4.40625


H2 = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
e, v = eigh(H2)
# e = array([-1.5, -0.5,  0.5,  1.5])

iqpe_projected(
    U=PauliEvolutionGate(H2, time=2 * np.pi / 2**2),
    initial_state=v[:, 0],
    num_bits=5,
    seed=42,
    verbose=True,
)
# 5: [0, 0, 1, 0, 0]
# 9: [0, 0, 0, 0, 0, 0, 1, 0, 0]



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
e, v = eigh(heisenberg_h)
# e = array([-3.9,  1.3,  1.3,  1.3])

iqpe_projected(
    U=PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=v[:, 0],
    num_bits=5,
    seed=42,
    verbose=True,
)
# 5: [1, 1, 1, 1, 1]
# 9: [1, 1, 0, 0, 1, 1, 1, 1, 1]



e, v = driver(heisenberg_h, 1e-20)

iqpe_projected(
    U=PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=v,
    num_bits=5,
    seed=42,
    verbose=True,
)
# 5: [1, 1, 1, 1, 1]
# 9: [1, 1, 0, 0, 1, 1, 1, 1, 1]




def projection_circuit(circuit):
    times =  [np.pi / 2, np.pi / 4, np.pi / 8, np.pi / 16]
    phases = [0, 0, 0, 0]
    aux_qubits = [0, 1, 2, 3]

    num_qubits = circuit.num_qubits

    qc = QuantumCircuit(num_qubits + len(aux_qubits), len(aux_qubits))
    qc.append(circuit, range(len(aux_qubits), len(aux_qubits) + num_qubits))

    for j, (t, delta) in enumerate(zip(times, phases)):
        aux_qubit = aux_qubits[j]
        qc.sdg(aux_qubit)
        qc.h(aux_qubit)

        for i in range(len(aux_qubits), len(aux_qubits) + num_qubits):
            qc.cx(aux_qubit, i)
            qc.rz(2 * t + delta, i)
            qc.cx(aux_qubit, i)

        qc.h(aux_qubit)
        qc.s(aux_qubit)

        qc.barrier()

    return qc


def apply_Jz_projection(circuit, H_matrix):
    num_qubits = circuit.num_qubits
    qc = projection_circuit(circuit)

    simulator = StatevectorSimulator()
    statevector = (
        simulator
        .run(transpile(qc, simulator), shots=1024)
        .result()
        .get_statevector()
        .to_dict()
    )
    data = np.zeros(2 ** num_qubits, dtype=complex)
    for k, v in statevector.items():
        # states where ancillas were measured in |0> states
        if k.endswith('0000'):
            data[int(k[:-4], 2)] = v

    # normalized state after measurement
    psi = Statevector(data)
    normalized_vector = psi / np.linalg.norm(psi)

    new_state = np.array(normalized_vector)
    exp = np.vdot(new_state, H_matrix @ new_state).real

    return normalized_vector, exp


def apply_first_Jz_projection(num_qubits, H_matrix):
    initial_circuit = QuantumCircuit(num_qubits)
    initial_state = np.array(Statevector.from_label("0" * num_qubits))
    return (
        *apply_Jz_projection(initial_circuit, H_matrix),
        # Expectation value with initial state
        np.vdot(initial_state, H_matrix @ initial_state).real,
    )


def apply_Jx_and_Jz_projection(normalized_vector, H_matrix):
    num_qubits = normalized_vector.num_qubits
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(normalized_vector, range(num_qubits))

    # Jx projection
    for qubit in range(num_qubits):
        circuit.rx(np.pi / 2, qubit)

    return apply_Jz_projection(circuit, H_matrix)


def driver(hamiltonian, threshold=1e-6):
    np.set_printoptions()

    num_qubits = hamiltonian.num_qubits
    H_matrix = hamiltonian.to_matrix()

    v_0, e_p, e_0 = apply_first_Jz_projection(num_qubits, H_matrix)
    v_c, e_c = apply_Jx_and_Jz_projection(v_0, H_matrix)
    exp_vals = [e_0, e_p, e_c]
    while (diff := abs((e_c - e_p) / e_p)) > threshold:
        print(f"Energy % difference: {diff}")
        e_p = e_c
        v_c, e_c = apply_Jx_and_Jz_projection(v_c, H_matrix)
        exp_vals.append(e_c)

    return exp_vals, v_c












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
U = PauliEvolutionGate(H, time=2 * np.pi / 2**3)

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



# Combined with Projection Algorithm.

exp_vals, initial_state = driver(heisenberg_h, 1e-20)
qpe(
    8,
    PauliEvolutionGate(heisenberg_h, time=2 * np.pi / 2**2),
    initial_state=initial_state,
)
# {'11111010': 1160, ...} -> - 2**2 * (1/2 + 1/2**2 + 1/2**3 + 1/2**4 + 1/2**5 + 1/2**7) = 3.90625
