"""
Created on Thu Aug  1 15:08:13 2024
Revised on Mon Oct 27 2025

@author: mihalikova
@contributer: minjoon-park

qiskit==2.2.2
qiskit-aer==0.17.2
qiskit_ibm_runtime==0.43.1
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import StatevectorSimulator
from scipy.linalg import eigh

num_qubits = 4 * 3

# 4x3 Heisenberg lattice
hamiltonian = SparsePauliOp(
    [
        'XIIXIIIIIIII', 'YIIYIIIIIIII', 'ZIIZIIIIIIII', 'XXIIIIIIIIII',
        'YYIIIIIIIIII', 'ZZIIIIIIIIII', 'XIIIIIIIXIII', 'YIIIIIIIYIII',
        'ZIIIIIIIZIII', 'XIIIXIIIIIII', 'YIIIYIIIIIII', 'ZIIIZIIIIIII',
        'IIIIXIIXIIII', 'IIIIYIIYIIII', 'IIIIZIIZIIII', 'IIIIXXIIIIII',
        'IIIIYYIIIIII', 'IIIIZZIIIIII', 'IIIIXIIIXIII', 'IIIIYIIIYIII',
        'IIIIZIIIZIII', 'IIIIIIIIXIIX', 'IIIIIIIIYIIY', 'IIIIIIIIZIIZ',
        'IIIIIIIIXXII', 'IIIIIIIIYYII', 'IIIIIIIIZZII', 'IXXIIIIIIIII',
        'IYYIIIIIIIII', 'IZZIIIIIIIII', 'IXIIIIIIIXII', 'IYIIIIIIIYII',
        'IZIIIIIIIZII', 'IXIIIXIIIIII', 'IYIIIYIIIIII', 'IZIIIZIIIIII',
        'IIIIIXXIIIII', 'IIIIIYYIIIII', 'IIIIIZZIIIII', 'IIIIIXIIIXII',
        'IIIIIYIIIYII', 'IIIIIZIIIZII', 'IIIIIIIIIXXI', 'IIIIIIIIIYYI',
        'IIIIIIIIIZZI', 'IIXXIIIIIIII', 'IIYYIIIIIIII', 'IIZZIIIIIIII',
        'IIXIIIIIIIXI', 'IIYIIIIIIIYI', 'IIZIIIIIIIZI', 'IIXIIIXIIIII',
        'IIYIIIYIIIII', 'IIZIIIZIIIII', 'IIIIIIXXIIII', 'IIIIIIYYIIII',
        'IIIIIIZZIIII', 'IIIIIIXIIIXI', 'IIIIIIYIIIYI', 'IIIIIIZIIIZI',
        'IIIIIIIIIIXX', 'IIIIIIIIIIYY', 'IIIIIIIIIIZZ', 'IIIXIIIIIIIX',
        'IIIYIIIIIIIY', 'IIIZIIIIIIIZ', 'IIIXIIIXIIII', 'IIIYIIIYIIII',
        'IIIZIIIZIIII', 'IIIIIIIXIIIX', 'IIIIIIIYIIIY', 'IIIIIIIZIIIZ',
    ],
    coeffs=np.array(
        [
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
        ]
    ),
)
H_matrix = hamiltonian.to_matrix()

times =  [np.pi / 2, np.pi / 4, np.pi / 8, np.pi / 16]
phases = [0,0,0,0]


def initialize_circuit(state1: str, state2: str):
    n = len(state1)

    assert n == len(state2), "States must have the same length"

    qc = QuantumCircuit(n)

    s1 = state1[::-1]
    s2 = state2[::-1]

    for i, bit in enumerate(s1):
        if bit == "1":
            qc.x(i)

    control_qubits = [i for i in range(n) if s1[i] != s2[i]]

    control = control_qubits[0]
    qc.h(control)

    for i in control_qubits[1:]:
        qc.cx(control, i)

    return qc


def projection_circuit(circuit, times, phases, measure=False):
    aux_qubits = [0, 1, 2, 3]

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

        if measure:
            qc.measure_all()

    return qc


def apply_Jz_projection(circuit):
    qc = projection_circuit(circuit, times, phases, measure=False)

    simulator = StatevectorSimulator()
    statevector = simulator.run(transpile(qc, simulator)).result().get_statevector().to_dict()
    np.set_printoptions(threshold=5)
    data = np.zeros(2 ** num_qubits, dtype=complex)
    for k, v in statevector.items():
        # states where ancillas were measured in |0> states
        if k.endswith('0000'):
            data[int(k[:-4], 2)] = v

    # normalized state after measurement
    psi = Statevector(data)
    normalized_vector = psi / np.linalg.norm(psi)

    np.set_printoptions()
    new_state = np.array(normalized_vector)
    exp = np.vdot(new_state, H_matrix @ new_state).real

    return normalized_vector, exp


def apply_first_Jz_projection():
    neal_state = ["101001011010", "010110100101"]
    initial_circuit = initialize_circuit(*neal_state)
    # print(initial_circuit.decompose())
    initial_state =  np.array(Statevector.from_label('0' * 12).evolve(initial_circuit))
    return (
        *apply_Jz_projection(initial_circuit),
        # Expectation value with initial state
        np.vdot(initial_state, H_matrix @ initial_state).real,
    )


def apply_Jx_and_Jz_projection(normalized_vector):
    circuit = QuantumCircuit(12)
    circuit.initialize(normalized_vector, range(num_qubits))

    # Jx projection
    for qubit in range(num_qubits):
        circuit.rx(np.pi / 2, qubit)

    return apply_Jz_projection(circuit)


def driver():
    v_0, e_p, e_0 = apply_first_Jz_projection()
    v_c, e_c = apply_Jx_and_Jz_projection(v_0)
    exp_vals = [e_0, e_p, e_c]
    while (diff := abs((e_c - e_p) / e_p)) > 1e-6:
        print(f"Energy % difference: {diff}")
        e_p = e_c
        v_c, e_c = apply_Jx_and_Jz_projection(v_c)
        exp_vals.append(e_c)

    return exp_vals, v_c


def plot(exp_vals):
    n = len(exp_vals)
    x = list(range(n))

    # Compute actual ground state energy.
    # eigenvalues, _ = eigh(H_matrix)
    # actual = eigenvalues[0]
    # Precomputed, as it takes time.
    actual = -58.94574155307309

    plt.ylim(actual - 2, exp_vals[0] + 2)
    plt.xticks(x)
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(direction="in", top=True, right=True)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.plot(x, exp_vals, "bo", label="J$^2$ Projection")
    plt.plot([0, n - 1], [actual, actual], "k--", label="Ground state")

    plt.show()


# exp_vals, final_state = driver()
# plot(exp_vals)
