#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:49:37 2026

@author: im

pennylane>=0.40.0

"""

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from matplotlib.ticker import MaxNLocator


NUM_SYSTEM_QUBITS = 12          # 4×3 lattice
NUM_AUX = 4                     # ancilla qubits
NUM_TOTAL = NUM_SYSTEM_QUBITS + NUM_AUX

times  = [np.pi / 2, np.pi / 4, np.pi / 8, np.pi / 16]
phases = [0, 0, 0, 0]


pauli_strings_qiskit = [
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
]

# reverse Qiskit -> PennyLane 
pauli_strings_pl = [s[::-1] for s in pauli_strings_qiskit]
coeffs = [2.0] * len(pauli_strings_pl)


def make_hamiltonian():
    ops = []
    for pstr in pauli_strings_pl:
        terms = []
        for wire, ch in enumerate(pstr):
            if ch != 'I':
                terms.append(getattr(qml, ch)(wire))
        if len(terms) == 1:
            ops.append(terms[0])
        else:
            prod = terms[0]
            for t in terms[1:]:
                prod = prod @ t
            ops.append(prod)
    return qml.Hamiltonian(coeffs, ops)

H = make_hamiltonian()
H_matrix = qml.matrix(H, wire_order=list(range(NUM_SYSTEM_QUBITS)))


dev_total  = qml.device("default.qubit", wires=NUM_TOTAL)
dev_system = qml.device("default.qubit", wires=NUM_SYSTEM_QUBITS)


AUX_WIRES    = list(range(NUM_AUX))              
SYSTEM_WIRES = list(range(NUM_AUX, NUM_TOTAL))     

SYSTEM_WIRES_LOCAL = list(range(NUM_SYSTEM_QUBITS)) 

# superposition of Neel states
def initialize_state(state1: str, state2: str, wires: list):

    n = len(state1)
    assert n == len(state2) == len(wires)
 
    s1 = state1[::-1]   
    s2 = state2[::-1]
 
    for i, bit in enumerate(s1):
        if bit == '1':
            qml.PauliX(wires=wires[i])
 
    control_qubits = [i for i in range(n) if s1[i] != s2[i]]
    ctrl = control_qubits[0]
    qml.Hadamard(wires=wires[ctrl])
    for i in control_qubits[1:]:
        qml.CNOT(wires=[wires[ctrl], wires[i]])
 
 
def projection_gates(state_vector=None, times=times, phases=phases, apply_rx=False):
    if state_vector is not None:
        qml.StatePrep(state_vector, wires=SYSTEM_WIRES, normalize=True)
    else:
        initialize_state("101001011010", "010110100101", SYSTEM_WIRES)
 
    if apply_rx:
        for w in SYSTEM_WIRES:
            qml.RX(np.pi / 2, wires=w)
 
    for j, (t, delta) in enumerate(zip(times, phases)):
        aux = AUX_WIRES[j]
 
        qml.adjoint(qml.S)(wires=aux)
        qml.Hadamard(wires=aux)
 
        for sys_w in SYSTEM_WIRES:
            qml.CNOT(wires=[aux, sys_w])
            qml.RZ(2 * t + delta, wires=sys_w)
            qml.CNOT(wires=[aux, sys_w])
 
        qml.Hadamard(wires=aux)
        qml.S(wires=aux)
 
 
@qml.qnode(dev_total)
def get_full_statevector(state_vector=None, apply_rx=False):
    projection_gates(state_vector=state_vector, apply_rx=apply_rx)
    return qml.state()
 
 
def apply_Jz_projection(state_vector=None, apply_rx=False):
    full_sv = np.array(get_full_statevector(state_vector=state_vector,
                                              apply_rx=apply_rx))
 

    dim_sys = 2 ** NUM_SYSTEM_QUBITS   
    dim_aux = 2 ** NUM_AUX             
 
    sv_reshaped = full_sv.reshape(dim_aux, dim_sys)
 
    raw = sv_reshaped[0, :]
 
    norm = np.linalg.norm(raw)
    psi = raw / norm
 
    exp = np.vdot(psi, H_matrix @ psi).real
    return psi, exp
 
 
def apply_first_Jz_projection():

    @qml.qnode(dev_system)
    def initial_sv():
        initialize_state("101001011010", "010110100101", SYSTEM_WIRES_LOCAL)
        return qml.state()
 
    initial_state = np.array(initial_sv())
    e_0 = np.vdot(initial_state, H_matrix @ initial_state).real
 
    psi, e_p = apply_Jz_projection(state_vector=None, apply_rx=False)
    return psi, e_p, e_0
 
 
def apply_Jx_and_Jz_projection(state_vector):

    return apply_Jz_projection(state_vector=state_vector, apply_rx=True)
 

def driver():
    v0, e_p, e_0 = apply_first_Jz_projection()
    v_c, e_c = apply_Jx_and_Jz_projection(v0)
    exp_vals = [e_0, e_p, e_c]
 
    while (diff := abs((e_c - e_p) / e_p)) > 1e-6:
        print(f"Energy % difference: {diff:.2e}   E = {e_c:.6f}")
        e_p = e_c
        v_c, e_c = apply_Jx_and_Jz_projection(v_c)
        exp_vals.append(e_c)
 
    print(f"Converged. Final energy: {e_c:.6f}")
    return exp_vals, v_c
 
 

def plot(exp_vals):
    actual = -58.94574155307309 
 
    n = len(exp_vals)
    x = list(range(n))
 
    fig, ax = plt.subplots()
    ax.set_ylim(actual - 2, exp_vals[0] + 2)
    ax.set_xticks(x)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.plot(x, exp_vals, "bo", label=r"$J^2$ Projection")
    ax.plot([0, n - 1], [actual, actual], "k--", label="Ground state")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

exp_vals, final_state = driver()
plot(exp_vals)