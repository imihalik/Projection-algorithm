import numpy as np
from qiskit.circuit.library import PauliEvolutionGate

from Projection.projection_algorithm_qiskit_2 import driver, hamiltonian
from qpe import qpe

# Get the initial state from the projection algorithm.
_, init_state = driver()

# Use QPE to estimate the eigenvalue of the Hamiltonian.
# `PauliEvolutionGate` gives e^{-i t H} for the Hamiltonian H.
# From `Projection.projection_algorithm.eigenvalues`, we know the ground state energy is around -59.
# Choose t = 2π / 2^n, where n >= 6, so that as long as -H is less than 64, -H / 2^n will be less
# than 1, enabling QPE to estimate it.

# First choose n > 6, e.g., 7. Phases whose MSB is 1 are negligible.
# 0110000000, 0101000000, 0111000000, 0100000000 are dominant.
In [1]: qpe(10, PauliEvolutionGate(hamiltonian, time=2 * np.pi / 2**7), initial_state=init_state)
Out[1]:
{'1110000000': 4,
 '1010000000': 18,
 '1100000000': 22,
 '1011000000': 10,
 '1101000000': 12,
 '0100000000': 127,
 '1000000000': 61,
 '1001000000': 45,
 '0000000000': 20,
 '0001000000': 49,
 '0010000000': 66,
 '1111000000': 12,
 '0110000000': 200,
 '0111000000': 130,
 '0101000000': 168,
 '0011000000': 80}

# Now n = 6. All the states are equally probable, so 11100000 is likely to be the ground state, w/
# energy ~ -56.
In [4]: qpe(8, PauliEvolutionGate(hamiltonian, time=2 * np.pi / 2**6), initial_state=init_state)
Out[4]:
{'00100000': 77,
 '10000000': 134,
 '01000000': 84,
 '01100000': 94,
 '00000000': 101,
 '11100000': 159,
 '10100000': 162,
 '11000000': 213}

# Currently debugging why the less significant bits are all 0.
