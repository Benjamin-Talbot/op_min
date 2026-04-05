from operator_min import *

h2o_name = 'H2O_JW'

with open(hamiltonians_path + h2o_name + '.txt', 'r') as f:
    h2o_hamiltonian = read_hamiltonian_from_file(h2o_name)

paulis = SparsePauliOp.from_list([(t, 1.0) for t in h2o_hamiltonian])
k = 322
pauli_grouping_info, result = pauli_grouping(paulis, C, D, k, reps, optimization_level, api_key)

print(result)
print()

print("\nBest solution:")
print(result.prettyprint())

print()
print(result.x)