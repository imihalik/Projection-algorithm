import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def qpe(
    precision: int,
    unitary: QuantumCircuit | Instruction,
    initializer: QuantumCircuit | Instruction | None = None,
    initial_state: Statevector | None = None,
) -> dict:
    num_state_qubits = unitary.num_qubits
    state_qubits = list(range(precision, precision + num_state_qubits))
    ancilla_qubits = list(range(precision))

    qpe_circuit = QuantumCircuit(precision + num_state_qubits, precision)
    if initializer:
        qpe_circuit.compose(initializer, state_qubits, inplace=True, front=True)
    elif initial_state:
        qpe_circuit.initialize(initial_state, state_qubits)

    qpe_circuit.h(ancilla_qubits)

    for ancilla_qubit in ancilla_qubits:
        qpe_circuit.compose(
            unitary.control().power(2**ancilla_qubit),
            [ancilla_qubit] + state_qubits,
            inplace=True,
        )

    qpe_circuit.barrier()

    qpe_circuit.compose(QFT(precision, inverse=True), ancilla_qubits, inplace=True)

    qpe_circuit.barrier()

    for digit in ancilla_qubits:
        qpe_circuit.measure(digit, digit)

    aer_sim = AerSimulator()
    return aer_sim.run(transpile(qpe_circuit, aer_sim), shots=2048).result().get_counts()
