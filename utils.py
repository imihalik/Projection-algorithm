import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper


def bitstring_to_eigenvalue(bits: str, t_evolution: float, is_positive: bool) -> float:
    """Convert the measured bitstring to the eigenvalue of the Hamiltonian."""
    phi = sum(int(bit) * 2**(-i) for i, bit in enumerate(bits, start=1))
    return -2 * np.pi * ((phi - 1) if is_positive else phi) / t_evolution


def get_isotropic_1d_heisenberg_hamiltonian(num_qubits, J=1.0):
    """Constructs an Isotropic Heisenberg Hamiltonian for a 1D chain.

    H = J * sum_{i} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    """
    ham_list = []
    # Iterate through adjacent pairs (nearest-neighbors)
    for i in range(num_qubits - 1):
        # Isotropic coupling: same J for XX, YY, and ZZ
        ham_list.append(("XX", [i, i+1], J))
        ham_list.append(("YY", [i, i+1], J))
        ham_list.append(("ZZ", [i, i+1], J))

    return SparsePauliOp.from_sparse_list(ham_list, num_qubits=num_qubits)


def get_fermi_hubbard_hamiltonian_on_line_lattice(
    n: int,
    uniform_interaction: complex = -1.0,
    uniform_onsite_potential: complex = 0.0,
    onsite_interaction: complex = 5.0,
) -> SparsePauliOp:
    """Constructs a simple Fermi-Hubbard Hamiltonian for a n-site system."""
    fermi_hubbard_model = FermiHubbardModel(
        LineLattice(num_nodes=n, boundary_condition=BoundaryCondition.OPEN)
        .uniform_parameters(
            uniform_interaction=uniform_interaction,
            uniform_onsite_potential=uniform_onsite_potential
        ),
        onsite_interaction=onsite_interaction,
    )
    return JordanWignerMapper().map(fermi_hubbard_model.second_q_op())
