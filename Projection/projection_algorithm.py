# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:08:13 2024

@author: mihalikova

qiskit==0.46.0
qiskit-terra==0.46.0
qiskit-aer==0.14.2
"""

from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
#%% 
# 4x3 Heisenberg lattice
hamiltonian = SparsePauliOp(['XIIXIIIIIIII', 'YIIYIIIIIIII', 'ZIIZIIIIIIII', 'XXIIIIIIIIII', 'YYIIIIIIIIII', 'ZZIIIIIIIIII', 'XIIIIIIIXIII', 'YIIIIIIIYIII', 'ZIIIIIIIZIII', 'XIIIXIIIIIII', 'YIIIYIIIIIII', 'ZIIIZIIIIIII', 'IIIIXIIXIIII', 'IIIIYIIYIIII', 'IIIIZIIZIIII', 'IIIIXXIIIIII', 'IIIIYYIIIIII', 'IIIIZZIIIIII', 'IIIIXIIIXIII', 'IIIIYIIIYIII', 'IIIIZIIIZIII', 'IIIIIIIIXIIX', 'IIIIIIIIYIIY', 'IIIIIIIIZIIZ', 'IIIIIIIIXXII', 'IIIIIIIIYYII', 'IIIIIIIIZZII', 'IXXIIIIIIIII', 'IYYIIIIIIIII', 'IZZIIIIIIIII', 'IXIIIIIIIXII', 'IYIIIIIIIYII', 'IZIIIIIIIZII', 'IXIIIXIIIIII', 'IYIIIYIIIIII', 'IZIIIZIIIIII', 'IIIIIXXIIIII', 'IIIIIYYIIIII', 'IIIIIZZIIIII', 'IIIIIXIIIXII', 'IIIIIYIIIYII', 'IIIIIZIIIZII', 'IIIIIIIIIXXI', 'IIIIIIIIIYYI', 'IIIIIIIIIZZI', 'IIXXIIIIIIII', 'IIYYIIIIIIII', 'IIZZIIIIIIII', 'IIXIIIIIIIXI', 'IIYIIIIIIIYI', 'IIZIIIIIIIZI', 'IIXIIIXIIIII', 'IIYIIIYIIIII', 'IIZIIIZIIIII', 'IIIIIIXXIIII', 'IIIIIIYYIIII', 'IIIIIIZZIIII', 'IIIIIIXIIIXI', 'IIIIIIYIIIYI', 'IIIIIIZIIIZI', 'IIIIIIIIIIXX', 'IIIIIIIIIIYY', 'IIIIIIIIIIZZ', 'IIIXIIIIIIIX', 'IIIYIIIIIIIY', 'IIIZIIIIIIIZ', 'IIIXIIIXIIII', 'IIIYIIIYIIII', 'IIIZIIIZIIII', 'IIIIIIIXIIIX', 'IIIIIIIYIIIY', 'IIIIIIIZIIIZ'],              
coeffs= [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])


H_matrix = hamiltonian.to_matrix()

from scipy.linalg import eigh
eigenvalues, eigenvectors = eigh(H_matrix)

print(eigenvalues[0:15])

#%%

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


#Neel states
state1 = "101001011010"
state2 = "010110100101"


s1 = state1[::-1]
s2 = state2[::-1]

n = len(s1)  # number of qubits
qc = QuantumCircuit(n)


for i, bit in enumerate(s1):
    if bit == '1':
        qc.x(i)


control_qubits = []
for i in range(n):
    if s1[i] != s2[i]:
        control_qubits.append(i)


control = control_qubits[0]
qc.h(control)


for i in control_qubits[1:]:
    qc.cx(control, i)

state = Statevector.from_label('0' * 12)
full_initial_state =  np.array(state.evolve(qc))

#Expectation value with our initial state
expectation_value_classical = np.vdot(full_initial_state, H_matrix @ full_initial_state)
print("Classical expectation value:", expectation_value_classical.real)


#%% initialization of our circuit

def initialize_circuit(state1: str, state2: str):

    assert len(state1) == len(state2), "States must have the same length"
    n = len(state1)
    qc = QuantumCircuit(n)

    s1 = state1[::-1]
    s2 = state2[::-1]

    for i, bit in enumerate(s1):
        if bit == '1':
            qc.x(i)

    control_qubits = []
    target_qubit = None
    for i in range(n):
        if s1[i] != s2[i]:
            control_qubits.append(i)

    control = control_qubits[0]
    qc.h(control)

    for i in control_qubits[1:]:
        qc.cx(control, i)
    return qc
    
initial_circuit = initialize_circuit("101001011010", "010110100101")
#print(initial_circuit.decompose())


def projection_circuit(initial_circuit, times, phases, measure=False):
    num_qubits = 12
    aux_qubits = [0, 1, 2, 3]
    qc = QuantumCircuit(num_qubits + len(aux_qubits), len(aux_qubits))
    initial_circuit_shifted = initial_circuit.copy().to_gate(label="InitialState")
    qc.append(initial_circuit_shifted, range(len(aux_qubits), num_qubits + len(aux_qubits)))
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
#%% first Jz projection

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from qiskit import transpile, Aer, execute



times =  [np.pi/2, np.pi/4, np.pi/8, np.pi/16]
phases = [0,0,0,0]


qc = projection_circuit(initial_circuit, times, phases, measure=False)


simulator = Aer.get_backend('statevector_simulator')
trans = transpile(qc, simulator)
result = execute(trans, simulator).result()
statevector = result.get_statevector()
np.set_printoptions(threshold=5)


statevector_dict = statevector.to_dict()
statevector_dict_0000 = {key: value for key, value in statevector_dict.items() if key.endswith('0000')} #states where ancillas were measured in |0> states
statevector_dict_new = {key[:-4] if key.endswith('0000') else key: value for key, value in statevector_dict_0000.items()} #removing zeros

num_qubits = 12

data = np.zeros(2 ** num_qubits, dtype=complex)

for key in statevector_dict_new:
    data[int(key, 2)] = statevector_dict_new[key]

psi = Statevector(data)
norm = np.linalg.norm(psi)
normalized_vector = psi / norm
np.set_printoptions(threshold=np.inf)


full_initial_state = np.array(normalized_vector) #normalized state after measurement
exp = np.vdot(full_initial_state, H_matrix @ full_initial_state)
print("First Jz projection:")
print(exp)

#%% Jx and Jz projections

init = QuantumCircuit(12)
init.initialize(normalized_vector, range(12))

#Jx projection
circuit=init
for qubit in range(12):
    circuit.rx(np.pi/2, qubit)

#Jz projection
def projection_circuit(initial_circuit, times, phases, measure=False):
    num_qubits = 12
    aux_qubits = [0, 1, 2, 3]

    qc = QuantumCircuit(num_qubits + len(aux_qubits), len(aux_qubits))

    initial_circuit_shifted = initial_circuit
    qc.append(initial_circuit_shifted, range(len(aux_qubits), num_qubits + len(aux_qubits)))
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



qc = projection_circuit(init, times, phases, measure=False)
simulator = Aer.get_backend('statevector_simulator')
trans = transpile(qc, simulator)
result = execute(trans, simulator).result()
statevector = result.get_statevector()
np.set_printoptions(threshold=5)
statevector_dict = statevector.to_dict()
statevector_dict_0000 = {key: value for key, value in statevector_dict.items() if key.endswith('0000')} #states where ancillas were measured in |0> states
statevector_dict_new = {key[:-4] if key.endswith('0000') else key: value for key, value in statevector_dict_0000.items()} #removing zeros


data = np.zeros(2 ** num_qubits, dtype=complex)

for key in statevector_dict_new:
    data[int(key, 2)] = statevector_dict_new[key]

psi = Statevector(data)
norm = np.linalg.norm(psi)

normalized_vector = psi / norm
np.set_printoptions(threshold=np.inf)


full_initial_state = np.array(normalized_vector) #normalized state after measurement
exp = np.vdot(full_initial_state, H_matrix @ full_initial_state)
print("Jx and Jz projection (run this cell multiple times):")
print(exp)

#%% results

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

projected = [-32, -39.27272727272725, -43.48647697684815, -44.86390325116048, -45.21860389314227, -45.30525818486656, -45.326413968519994, -45.33161829816402, -45.332906705227366, -45.333226978774405, -45.33330678738977]

x = [0,1,2,3,4,5,6,7,8,9,10]

yy = [-32,-32]
xx = [0,10]
yyy=[-58.94574155307309,-58.94574155307309]

plt.plot(xx,yy,"r--",label="Initial state")
plt.plot(x,projected,"bo",label="J$^2$ Projection")
plt.plot(xx,yyy,"k--",label="Ground state")
plt.ylim(-60,-23)
plt.xticks(x)
plt.legend(loc='upper right') 
ax = plt.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.tick_params(direction='in', top=True, right=True)
plt.xlabel('Step')
plt.ylabel('Energy')

plt.show()




