import math

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import QFT, PhaseGate, RZGate, TGate, XGate
from qiskit_aer import AerSimulator


def qpe(
    precision: int,
    unitary: QuantumCircuit | Instruction,
    initializer: QuantumCircuit | Instruction | None = None,
) -> dict:
    num_state_qubits = unitary.num_qubits
    state_qubits = list(range(precision, precision + num_state_qubits))
    ancilla_qubits = list(range(precision))

    qpe = QuantumCircuit(precision + num_state_qubits, precision)
    if initializer:
        qpe.compose(initializer, state_qubits, inplace=True)

    qpe.h(ancilla_qubits)

    for ancilla_qubit in ancilla_qubits:
        qpe.compose(
            unitary.control().power(2**ancilla_qubit),
            [ancilla_qubit] + state_qubits,
            inplace=True,
        )

    qpe.barrier()

    qpe.compose(QFT(precision, inverse=True), ancilla_qubits, inplace=True)

    qpe.barrier()

    for digit in ancilla_qubits:
        qpe.measure(digit, digit)

    aer_sim = AerSimulator()
    return aer_sim.run(transpile(qpe, aer_sim), shots=1024).result().get_counts()


qpe(3, PhaseGate(2 * math.pi * (1 / 4 + 1 / 8)), XGate())
# {'011': 1024}

qpe(3, TGate(), XGate())
# {'001': 1024}
