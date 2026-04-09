from operator_min import *

h2_name = 'H2_BK'

with open(hamiltonians_path + h2_name + '.txt', 'r') as f:
    h2_hamiltonian = read_hamiltonian_from_file(h2_name)

paulis = SparsePauliOp.from_list([(t, 1.0) for t in h2_hamiltonian])
k = 3
pauli_grouping_info, result = pauli_grouping(paulis, C, D, k, reps, optimization_level, api_key)

print(result)
print()

print("\nBest solution:")
print(result.prettyprint())

print()
print(result.x)
print_grouping_summary(pauli_grouping_info)
