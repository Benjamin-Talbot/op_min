from operator_min import *

paulis_list = [
    'ZZZZ',
    'ZIII',
    'ZZZI',
    'ZZII',
    'IIXX',
    'YYXX',
    'YIXX'
]

paulis = SparsePauliOp.from_list([(p, 1.0) for p in paulis_list])

k = 2
ham_repr = True
pauli_grouping_info, result = pauli_grouping(paulis, C, D, k, reps, optimization_level, api_key, ham_repr=ham_repr)

if ham_repr:
    print('Result:', result.best_measurement['bitstring'])
    print('Cost value:', result.cost_function_evals)
else:
    print("Number of samples:", len(result.samples))
    print()

    print("\nBest solution:")
    print(result.prettyprint())

    print()
    print(result.x)