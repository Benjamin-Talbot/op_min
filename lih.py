from operator_min import *

lih_name = 'LiH'

with open(hamiltonians_path + lih_name + '.txt', 'r') as f:
    lih_hamiltonian = read_hamiltonian_from_file(lih_name)

paulis = SparsePauliOp.from_list([(t, 1.0) for t in lih_hamiltonian])
k = 25
ham_repr = True
pauli_grouping_info, result = pauli_grouping(paulis, C, D, k, reps, optimization_level, api_key, ham_repr=ham_repr)

if ham_repr:
    print('Result:', result.best_measurement['bitstring'])
    print('Objective value:', result.best_measurement['value'])
    print('Cost function evaluations:', result.cost_function_evals)
    print_grouping_summary(pauli_grouping_info)
else:
    print("Number of samples:", len(result.samples))
    print()

    print("\nBest solution:")
    print(result.prettyprint())

    print()
    print(result.x)
    print_grouping_summary(pauli_grouping_info)
