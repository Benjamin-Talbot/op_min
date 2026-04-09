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
pauli_grouping_info, result = pauli_grouping(paulis, C, D, k, reps, optimization_level, api_key)

print(result)
print("Number of samples:", len(result.samples))
print()

print("\nBest solution:")
print(result.prettyprint())

print()
print(result.x)
print_grouping_summary(pauli_grouping_info)
