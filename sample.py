from operator_min import *

paulis_list = [
    'ZZZZ',
    'ZIII',
    'ZZZI',
    'ZZII',
    'IIXX',
    'YYXX',
    'YIXX',
    'IIII'
]

paulis = SparsePauliOp.from_list([(p, 1.0) for p in paulis_list])

k = 2
pauli_grouping_info, result = pauli_grouping(paulis, C, D, k, reps, optimization_level, api_key, simulation=simulation)

print(result)
print("Number of samples:", len(result.samples))
print()

print("\nBest solution:")
print(result.prettyprint())

print()
print(result.x)

print("\n\n\n")
print("Optimized vertices:", pauli_grouping_info.active_vertices)
print("Skipped fully-commuting vertices:", pauli_grouping_info.fully_commuting_vertices)
print()

print("Skipped terms:")
for i in pauli_grouping_info.fully_commuting_vertices or []:
    print(i, pauli_grouping_info.paulis.paulis[i].to_label(), pauli_grouping_info.paulis.coeffs[i])

print()
print("Raw colours:", pauli_grouping_info.raw_colours)
print("Repaired colours:", pauli_grouping_info.repaired_colours)
print("Invalid vertices:", pauli_grouping_info.invalid_vertices)
print("Raw conflicts:", pauli_grouping_info.raw_conflicts)
print("Repaired conflicts:", pauli_grouping_info.repaired_conflicts)
print()

print("Groups by colour:")
for colour, group in enumerate(pauli_grouping_info.groups_by_colour or []):
    terms = [pauli_grouping_info.paulis.paulis[i].to_label() for i in group]
    print(f"Colour {colour}: indices={group}, terms={terms}")
