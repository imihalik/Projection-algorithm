import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def iqpe(U, initial_state, num_bits: int) -> tuple[float, list[int]]:
    if isinstance(initial_state, Statevector):
        psi = initial_state
    else:
        psi = Statevector(initial_state)
    n_sys = psi.num_qubits
    state_qubits = list(range(1, 1 + n_sys))

    aer_sim = AerSimulator()

    bits = []
    omega_coef = 0.0

    # Iterate from the least significant bit up to the most
    for k in range(num_bits, 0, -1):
        omega_coef /= 2.0
        power = 2 ** (k - 1)

        # 0. Initialize circuit
        qc = QuantumCircuit(1 + n_sys, 1)
        qc.initialize(psi, state_qubits)

        # 1. Ancilla in |+>
        qc.h(0)

        # 2. Controlled-U^(2^{k-1})
        qc.append(U.control().power(power), [0] + state_qubits)

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

    bits.reverse()
    return omega_coef, bits
