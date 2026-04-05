from operator_min import *

beh2_name = 'BeH2_BK'

with open(hamiltonians_path + beh2_name + '.txt', 'r') as f:
    beh2_hamiltonian = read_hamiltonian_from_file(beh2_name)

paulis = SparsePauliOp.from_list([(t, 1.0) for t in beh2_hamiltonian])
k = 172
pauli_grouping_info, result = pauli_grouping(paulis, C, D, k, reps, optimization_level, api_key)

print(result)
print()

print("\nBest solution:")
print(result.prettyprint())

print()
print(result.x)