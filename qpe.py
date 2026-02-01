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
    top_n: int | None = 3,
    sim_shots: int = 2048,
    to_phase: bool = True,
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
    distribution = sorted(
        aer_sim.run(transpile(qpe_circuit, aer_sim), shots=sim_shots).result().get_counts().items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        bits_to_phase(k) if to_phase else k: v / sim_shots
        for i, (k, v) in enumerate(distribution)
        if i < (top_n or 2**precision)
    }


def bits_to_phase(bits: str) -> float:
    phase = 0.0
    for i, bit in enumerate(bits, start=1):
        phase += int(bit) / (2 ** i)
    return phase
