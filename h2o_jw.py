from operator_min import *

h2o_name = 'H2O_JW'

with open(hamiltonians_path + h2o_name + '.txt', 'r') as f:
    h2o_hamiltonian = read_hamiltonian_from_file(h2o_name)

paulis = SparsePauliOp.from_list([(t, 1.0) for t in h2o_hamiltonian])
k = 322
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
