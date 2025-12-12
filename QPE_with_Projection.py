import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, PhaseGate, XGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import StatevectorSimulator

from qpe import qpe

# Confirm `initial_state` works.
qc = QuantumCircuit(1)
qc.append(XGate(), [0])

simulator = StatevectorSimulator()
statevector = simulator.run(transpile(qc, simulator)).result().get_statevector()
initial_state = statevector / np.linalg.norm(statevector)

qpe(5, PhaseGate(2 * np.pi * (1 / 4 + 1 / 8 + 1 / 16)), initial_state=initial_state)
# {'01110': 1024}


# Simple Ising Hamiltonian
H = SparsePauliOp(["ZI", "IZ", "ZZ"], coeffs=[0.33, 3.24, 1.17])
e, v = np.linalg.eig(H)
# e: array([ 4.74+0.j, -4.08+0.j,  1.74+0.j, -2.4 +0.j])
# v: array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
#          [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
#          [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
#          [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])


# Works well if t is small enough.
U = PauliEvolutionGate(H, time=2 * np.pi / 2**3, synthesis=SuzukiTrotter(reps=2))

qpe(5, U, initial_state=list(v[0]))
# {'11011': 1, '01110': 1, '01100': 4, '10111': 1, '01101': 1017}
# -> 2**3 * (1 - (1 / 2**2 + 1 / 2**3 + 1 / 2**5)) = 4.75

qpe(9, U, initial_state=list(v[0]))
# {'011010001': 675, ...}
# -> 2**3 * (1 - (1 / 2**2 + 1 / 2**3 + 1 / 2**5 + 1 / 2**9)) = 4.734375


# If | H t | > 2π, the part of the phase larger than 2π is lost. I.e., either we at least know the
# range of the eigenvalue beforehand, or we choose t small "enough".
U = PauliEvolutionGate(H, time=2 * np.pi / 2, synthesis=SuzukiTrotter(reps=2))
qpe(5, U, initial_state=list(v[0]))
# {'10100': 945, ...}
# -> 2 * (1 - (1 / 2 + 1 / 2**3)) = 0.75


# Negative eigenvalue
U = PauliEvolutionGate(H, time=2 * np.pi / 2**3, synthesis=SuzukiTrotter(reps=2))
qpe(5, U, initial_state=list(v[1]))
# {'10000': 720, ...}
# -> -2**3 * (1 / 2) = -4.0


# Mixed state
U = PauliEvolutionGate(H, time=2 * np.pi / 2**3, synthesis=SuzukiTrotter(reps=2))
qpe(5, U, initial_state=list((v[0] + v[1]) / np.sqrt(2)))
# {'01101': 505, '10000': 385, ...}


# Q's
# - The above works well. But it requires knowledge of eigenvalues and eigenvectors of the
# Hamiltonian beforehand.
# - Assuming we don't know them beforehand, we should choose t small enough, meaning we'd need
# large enough precision.
# - If t is too small, would it be practically possible to measure?
# - Initial state prepared by Projection Algo seems to be the issue. Still looking.


####################################################################################################

See https://darveshiyat.medium.com/implementing-quantum-phase-estimation-algorithm-using-qiskit-e808e8167d32.
I ported it into qiskit 2.2.3, and confirmed what he wrote in the blog. It implements `my_qpe`,
and interprets the results in the following way:

`my_qpe`
- employs e^{-i H t} as the unitary operator
- returns a dictionary of counts of bitstrings measured from the ancilla qubits, e.g.,
{'1010': 512, '0110': 256, ... }.
- picks up the bitstring with the highest counts, e.g., '1010'.
- calculate the phase = -H t = 2 * pi * decimal('1010') / 2^n, where n is the number of ancilla
qubits. In a non-trivial example of a simple Ising Hamiltonian, he chose t = 1, and obtained
H = 4.71238, whereas the exact eigenvalue is 4.74. So, not bad.

But is this right? The phase works only by mod 2\pi, so when arbitrarily choosing t = 1,
what if -H * 1 > 2\pi?

In contrast, in our 4x3 Heisenberg lattice, I chose t to be small enough to make sure
- H t < 2\pi. E.g., t = 2\pi / 2^7, and got the most frequent bitstring, e.g., '0110000',
to obtain H = -48, whereas the exact ground state energy is -58.94574155.
The initial state obtained by projection algorithm has the enrgy of -45.33332670286026,

In [6]: qpe(7, PauliEvolutionGate(hamiltonian, time=2 * np.pi / 2**7), initial_state=init_state)
Out[6]:
{'1110000': 13,
 '0001000': 35,
 '1100000': 10,
 '1111000': 9,
 '1000000': 78,
 '1011000': 15,
 '0101000': 155,
 '0010000': 55,
 '0000000': 16,
 '0111000': 140,
 '0100000': 132,
 '1101000': 13,
 '1010000': 19,
 '1001000': 40,
 '0110000': 202,
 '0011000': 92}

To see the trend, increase the precision with t fixed:

In [7]: qpe(8, PauliEvolutionGate(hamiltonian, time=2 * np.pi / 2**7), initial_state=init_state)
Out[7]:
{'11110000': 6,
 '11000000': 21,
 '00000000': 27,
 '10100000': 9,
 '00110000': 85,
 '10010000': 42,
 '11100000': 8,
 '10000000': 58,
 '00100000': 74,
 '01010000': 152,
 '01000000': 139,
 '10110000': 8,
 '01100000': 191,
 '11010000': 14,
 '00010000': 47,
 '01110000': 143}

I.e., no change: '01100000' is still the most frequent bitstring.

Most of less significant bits are 0, so precision sucks.
