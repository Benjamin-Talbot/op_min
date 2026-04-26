from operator_min import *

lih_name = 'LiH'

with open(hamiltonians_path + lih_name + '.txt', 'r') as f:
    lih_hamiltonian = read_hamiltonian_from_file(lih_name)

paulis = SparsePauliOp.from_list([(t, 1.0) for t in lih_hamiltonian])
k = 25
pauli_grouping_info, result = pauli_grouping(paulis, C, D, k, reps, optimization_level, api_key, simulation=simulation)

print(result)
print()

print("\nBest solution:")
print(result.prettyprint())

print()
print(result.x)
