import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter


def project_ancilla(state: Statevector, bit: int, system_qubits: int) -> Statevector:
    reshaped = state.data.reshape(2, 2 ** system_qubits)
    projected = reshaped[bit, :]
    norm = np.linalg.norm(projected)
    if norm == 0:
        return Statevector(projected)
    return Statevector(projected / norm)


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
        qc.append(U.control(1).power(power), [0] + list(range(1, n_sys + 1)))

        # 3. Feedback phase
        theta = -2 * np.pi * omega_coef
        if abs(theta) > 1e-12:
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

        # 8. Project system onto post-measurement state
        psi = project_ancilla(full_state, bit, n_sys)

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






H = SparsePauliOp(["ZI", "IZ", "ZZ"], coeffs=[0.33, 3.24, 1.17])
evals, evecs = np.linalg.eigh(H)
array([-4.08, -2.4 ,  1.74,  4.74])

precision_bits = 5

iqpe_projected(
    U=PauliEvolutionGate(H, time=2 * np.pi / 2**precision_bits, synthesis=SuzukiTrotter(reps=2)),
    initial_state=evecs[:, 3],
    num_bits=precision_bits,
    seed=42,
    verbose=True,
)

iqpe_projected(
    U=PauliEvolutionGate(H2, time=2 * np.pi / 2**precision_bits, synthesis=SuzukiTrotter(reps=2)),
    initial_state=eigenvectors[:, 0],
    num_bits=precision_bits,
    seed=42,
    verbose=True,
)