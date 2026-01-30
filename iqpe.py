import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def iqpe(U, initial_state, num_bits: int):
    if isinstance(initial_state, Statevector):
        psi = initial_state
    else:
        psi = Statevector(initial_state)
    n_sys = psi.num_qubits
    state_qubits = list(range(1, 1 + n_sys))

    aer_sim = AerSimulator()

    bits = []
    omega_coef = 0.0

    # Iterate from least significant bit down to most
    for k in range(num_bits, 0, -1):
        omega_coef /= 2.0
        power = 2 ** (k - 1)

        # 0. Initialize circuit
        qc = QuantumCircuit(1 + n_sys, 1)
        qc.initialize(psi, state_qubits)

        # 1. Ancilla in |+>
        qc.h(0)

        # 2. Controlled-U^(2^{k-1})
        qc.append(U.control().power(power), [0] + list(range(1, 1 + n_sys)))

        # 3. Feedback phase
        qc.p(-2 * np.pi * omega_coef, 0)

        # 4. Final H on ancilla
        qc.h(0)
        qc.barrier()

        # 5. Measure
        qc.measure(0, 0)

        # 6. Simulate circuit
        counts = (
            aer_sim
            .run(transpile(qc, aer_sim), shots=2048)
            .result()
            .results[0]
            .data
            .counts
        )

        # 7. Get result
        bit = int(counts.get("0x1", 0) > counts.get("0x0", 0))
        bits.append(bit)
        omega_coef += bit / 2.0

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
