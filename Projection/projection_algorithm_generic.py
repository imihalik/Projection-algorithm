"""
Created on Thu Aug  1 15:08:13 2024
Last Revised on Mon Jan 20 2026

@author: mihalikova
@contributer: minjoon-park

qiskit==2.2.3
qiskit-aer==0.17.2
"""

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator


def projection_circuit(circuit):
    times =  [np.pi / 2, np.pi / 4, np.pi / 8, np.pi / 16]
    phases = [0, 0, 0, 0]
    aux_qubits = [0, 1, 2, 3]

    num_qubits = circuit.num_qubits

    qc = QuantumCircuit(num_qubits + len(aux_qubits), len(aux_qubits))
    qc.append(circuit, range(len(aux_qubits), num_qubits + len(aux_qubits)))

    for j, (t, delta) in enumerate(zip(times, phases)):
        aux_qubit = aux_qubits[j]
        qc.sdg(aux_qubit)
        qc.h(aux_qubit)

        for i in range(len(aux_qubits), num_qubits + len(aux_qubits)):
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


def apply_Jx_and_Jz_projection(normalized_vector, num_qubits, H_matrix):
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
    v_c, e_c = apply_Jx_and_Jz_projection(v_0, num_qubits, H_matrix)
    exp_vals = [e_0, e_p, e_c]
    while (diff := abs((e_c - e_p) / e_p)) > threshold:
        print(f"Energy % difference: {diff}")
        e_p = e_c
        v_c, e_c = apply_Jx_and_Jz_projection(v_c, num_qubits, H_matrix)
        exp_vals.append(e_c)

    return exp_vals, v_c
