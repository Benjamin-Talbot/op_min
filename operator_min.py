api_key = ''
# hamiltonians_path = '/home/btalbot/projects/def-stijn/btalbot/op_min/hamiltonians/' # Siku
# hamiltonians_path = '/home/btalbot/links/projects/def-stijn/btalbot/op_min/hamiltonians/' # Trillium
hamiltonians_path = '/home/btalbot/scratch/op_min/hamiltonians/' # Nibi
# hamiltonians_path = './hamiltonians/' # local

simulation = True

####################################
# Function and Problem Definitions #
####################################

# Imports

from itertools import combinations, product
from collections import defaultdict
from dataclasses import dataclass
import math
import os
from random import randint
from types import SimpleNamespace
import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import COBYLA, SPSA
from qiskit_optimization.utils import algorithm_globals

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler


# Graph colouring Qubo Quadratic Program

def adjacency_matrix_to_list(A):
    graph = {}
    n = len(A)
    for i in range(n):
        graph[i] = [j for j in range(n) if A[i][j] == 1]
    return graph

def color_bit_width(num_colors: int) -> int:
    if num_colors <= 1:
        return 1
    return math.ceil(math.log2(num_colors))

def int_to_bits(x, m):
    return [(x >> i) & 1 for i in range(m)]

def bits_to_int(bits):
    value = 0
    for i, bit in enumerate(bits):
        value |= int(bit) << i
    return value

def color_to_gray(color):
    return color ^ (color >> 1)

def gray_to_color(gray_value):
    color = 0
    while gray_value:
        color ^= gray_value
        gray_value >>= 1
    return color

def color_to_bits(color, m):
    return int_to_bits(color_to_gray(color), m)

def bits_to_color(bits):
    return gray_to_color(bits_to_int(bits))

def graph_edges(graph):
    if isinstance(graph, dict):
        for u, neighbours in graph.items():
            for v in neighbours:
                if u < v:
                    yield u, v
        return

    for u in range(len(graph)):
        for v in range(u + 1, len(graph)):
            if graph[u][v] == 1:
                yield u, v

def space_efficient_qp(A, num_colors, C=10.0, D=1.0):
    qp = QuadraticProgram()
    graph = adjacency_matrix_to_list(A)

    nodes = list(graph.keys())
    m = color_bit_width(num_colors)

    # --- Variables: b_{v,l}
    b = {}
    for v in nodes:
        for l in range(m):
            name = f"b_{v}_{l}"
            qp.binary_var(name)
            b[(v, l)] = name

    linear = {}
    quadratic = {}
    constant = 0


    def add_linear(var, coeff):
        linear[var] = linear.get(var, 0) + coeff

    def add_quadratic(v1, v2, coeff):
        key = tuple(sorted([v1, v2]))
        quadratic[key] = quadratic.get(key, 0) + coeff

    for u in nodes:
        for v in graph[u]:
            if u >= v:
                continue

            # Start with constant term
            terms = [({}, 1.0)]  # (monomial dict, coeff)

            for l in range(m):
                bu = b[(u, l)]
                bv = b[(v, l)]

                new_terms = []

                for mon, coeff in terms:
                    # multiply by (1 - bu - bv + 2 bu bv)

                    # 1
                    new_terms.append((mon.copy(), coeff))

                    # -bu
                    mon1 = mon.copy()
                    mon1[bu] = mon1.get(bu, 0) + 1
                    new_terms.append((mon1, -coeff))

                    # -bv
                    mon2 = mon.copy()
                    mon2[bv] = mon2.get(bv, 0) + 1
                    new_terms.append((mon2, -coeff))

                    # +2 bu bv
                    mon3 = mon.copy()
                    mon3[bu] = mon3.get(bu, 0) + 1
                    mon3[bv] = mon3.get(bv, 0) + 1
                    new_terms.append((mon3, 2 * coeff))

                terms = new_terms

            for mon, coeff in terms:
                coeff *= C

                vars_list = []
                for var, power in mon.items():
                    vars_list.extend([var] * power)

                if len(vars_list) == 0:
                    constant += coeff

                elif len(vars_list) == 1:
                    add_linear(vars_list[0], coeff)

                elif len(vars_list) == 2:
                    add_quadratic(vars_list[0], vars_list[1], coeff)

                else:
                    # introduce auxiliary variable
                    aux = "_aux_" + "_".join(vars_list)
                    qp.binary_var(aux)

                    k = len(vars_list)

                    # aux <= each var
                    for var in vars_list:
                        qp.linear_constraint({aux: 1, var: -1}, "<=", 0)

                    # aux >= sum(vars) - (k-1)
                    qp.linear_constraint(
                        {aux: 1, **{v: -1 for v in vars_list}},
                        ">=",
                        -(k - 1)
                    )

                    add_linear(aux, coeff)

    if num_colors < 2**m:
        for v in nodes:
            coeffs = {b[(v, l)]: 2**l for l in range(m)}
            qp.linear_constraint(coeffs, "<=", num_colors - 1)

    qp.minimize(linear=linear, quadratic=quadratic, constant=constant)
    return qp

def graph_colouring_qubo_qp(A: list[list[int]], k: int, C: int, D: int, name:str="graph_colouring_qubo"):
    """
    Build the QuadraticProgram for

        C * sum_v (1 - sum_i x[v,i])^2
      + D * sum_{v,w} sum_i A[v][w] x[v,i] x[w,i]

    Parameters
    ----------
    A : list[list[float]] or 2D array
        n x n adjacency matrix.
        For an undirected graph, A should typically be symmetric with zeros on the diagonal.
    k : int
        Number of colours.
    C, D : float
        Penalty weights.

    Returns
    -------
    qp : QuadraticProgram
    """
    n = len(A)
    qp = QuadraticProgram(name)

    # Create binary variables x_{v,i}
    x = {}
    for v in range(n):
        for i in range(k):
            var_name = f"x_{v}_{i}"
            qp.binary_var(var_name)
            x[(v, i)] = var_name

    linear = defaultdict(float)
    quadratic = defaultdict(float)

    # Constant part from C * sum_v 1
    constant = C * n

    # First term:
    # C * sum_v (1 - sum_i x_{v,i})^2
    #
    # For binary x:
    # (1 - sum_i x_i)^2 = 1 - sum_i x_i + 2 * sum_{i<j} x_i x_j
    for v in range(n):
        # linear: -C * x_{v,i}
        for i in range(k):
            linear[x[(v, i)]] += -C

        # quadratic: +2C * x_{v,i} x_{v,j} for i<j
        for i in range(k):
            for j in range(i + 1, k):
                quadratic[(x[(v, i)], x[(v, j)])] += 2.0 * C

    # Second term:
    # D * sum_{v,w} sum_i A[v][w] x_{v,i} x_{w,i}
    #
    # Since QuadraticProgram stores each pair once, we combine (v,w) and (w,v).
    # For symmetric A, this reproduces the full double sum exactly.
    for v in range(n):
        for w in range(v + 1, n):
            edge_weight = A[v][w] + A[w][v]
            if edge_weight != 0:
                coeff = D * edge_weight
                for i in range(k):
                    quadratic[(x[(v, i)], x[(w, i)])] += coeff

    qp.minimize(
        constant=constant,
        linear=dict(linear),
        quadratic=dict(quadratic),
    )
    return qp


# Graph Generation from Pauli Words

def qwc(word1, word2):
    '''
    Qubit-wise commutativity (QWC)
    [P_i, P_j]_qw
        = 0 if sigma_i and sigma_i' commute on each qubit, i.e.  [sigma_i, sigma_i'] = 0 for all i
        = 1 otherwise
    '''
    len1 = len(word1)
    len2 = len(word2)
    length = min(len1, len2)

    for i in range(length):
        # if neither are I and they are different, then they do not commute
        pauli1 = word1[i].to_label()
        pauli2 = word2[i].to_label()
        if pauli1 != 'I' and pauli2 != 'I' and pauli1 != pauli2:
            return 1
    return 0

def generate_qwc_graph(paulis):
    '''
    Generate the QWC graph for a SparsePauliOp object.
    Returns an adjacency matrix where A[i][j] = 1 if paulis[i] and paulis[j] commute, and 0 otherwise.
    '''

    n = len(paulis)
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            A[i][j] = 1 - qwc(paulis.paulis[i], paulis.paulis[j])
            A[j][i] = A[i][j]
    
    return A

def complement_graph(A):
    '''
    Given an adjacency matrix A, return the adjacency matrix of the complement graph.
    '''
    
    n = len(A)
    A_comp = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                A_comp[i][j] = 1 - A[i][j]
    
    return A_comp

def build_commutation_graph(paulis):
    return np.array(generate_qwc_graph(paulis), dtype=int)

def sparse_pauli_subset(paulis: SparsePauliOp, indices: list[int]) -> SparsePauliOp:
    return SparsePauliOp.from_list(
        [(paulis.paulis[i].to_label(), paulis.coeffs[i]) for i in indices]
    )

def split_universally_commuting_terms(paulis: SparsePauliOp):
    qwc_graph = generate_qwc_graph(paulis)
    universally_commuting = [
        index for index, row in enumerate(qwc_graph) if sum(row) == len(qwc_graph) - 1
    ]
    universally_commuting_set = set(universally_commuting)
    active_indices = [
        index for index in range(len(qwc_graph)) if index not in universally_commuting_set
    ]
    return qwc_graph, active_indices, universally_commuting

import itertools
import math

def projector_term(n_qubits, qubit_indices, bitstring):
    """
    Build projector onto a specific bitstring using Z operators.
    """
    terms = []

    for signs in itertools.product([0, 1], repeat=len(bitstring)):
        coeff = 1.0
        pauli = ["I"] * n_qubits

        for i, (bit, s) in enumerate(zip(bitstring, signs)):
            q = qubit_indices[i]

            if s == 0:
                coeff *= 0.5
            else:
                coeff *= 0.5 * (1 if bit == 0 else -1)
                # SparsePauliOp labels are ordered q_{n-1} ... q_0.
                pauli[n_qubits - 1 - q] = "Z"

        terms.append(("".join(pauli), coeff))

    return SparsePauliOp.from_list(terms)

def space_efficient_coloring_hamiltonian(graph, k, invalid_penalty=0.0, conflict_penalty=1.0):
    if not isinstance(graph, dict):
        graph = adjacency_matrix_to_list(graph)

    n = len(graph)
    m = color_bit_width(k)
    n_qubits = n * m

    H = SparsePauliOp.from_list([("I" * n_qubits, 0.0)])

    # When the mixer preserves the valid-color subspace these terms are
    # unnecessary, but keeping them optional makes the Hamiltonian usable
    # outside the constrained ansatz as well.
    if invalid_penalty != 0.0 and k < 2**m:
        for v in range(n):
            qubits = [v * m + i for i in range(m)]

            for c in range(k, 2**m):
                P = projector_term(n_qubits, qubits, color_to_bits(c, m))
                H += invalid_penalty * P

    # --- EDGE PENALTY ---
    for u, v in graph_edges(graph):
        qubits_u = [u * m + i for i in range(m)]
        qubits_v = [v * m + i for i in range(m)]

        for c in range(k):
            bits = color_to_bits(c, m)
            P_u = projector_term(n_qubits, qubits_u, bits)
            P_v = projector_term(n_qubits, qubits_v, bits)
            H += conflict_penalty * (P_u @ P_v)

    return H.simplify()

def binary_coloring_hamiltonian(graph, k, A=10.0, B=1.0):
    return space_efficient_coloring_hamiltonian(
        graph,
        k,
        invalid_penalty=A,
        conflict_penalty=B,
    )

def build_cost_hamiltonian(paulis, k, invalid_penalty=0.0, conflict_penalty=1.0):
    qwc_graph = generate_qwc_graph(paulis)
    conflict_graph = complement_graph(qwc_graph)
    return space_efficient_coloring_hamiltonian(
        conflict_graph,
        k,
        invalid_penalty=invalid_penalty,
        conflict_penalty=conflict_penalty,
    )



# Cost Function

def cost(bitstring: str, A: list[list[int]], vertices: list[int], colours: list[int], degrees: list[int], C: int, D: int):
    k = len(colours)

    sv = 0 # sum over vertices
    for v in vertices:
        sc = 0 # sum over colours
        for c in colours:
            bit = v * k + c
            sc += int(bitstring[bit])
        sv += (1 - sc)**2
    sc = C * sv # sum for C term
    
    sv = 0
    for v,w in combinations(vertices, 2):
        if A[v][w] == 1:
            for i in colours:
                vi = v * k + i
                wi = w * k + i
                sv += int(bitstring[vi]) * int(bitstring[wi])
    sd = D * sv # sum for D term
    
    cost = sc + sd

    return cost


# Misc

def get_modes(counts):
    max_val = counts[max(counts, key=counts.get)]
    max_counts = []
    for count in counts:
        if counts[count] == max_val:
            max_counts.append(count)
    return max_counts

def get_smallest_cost(costs):
    min_val = min(costs.values())
    for cost in costs:
        if costs[cost] == min_val:
            return cost
        
def read_hamiltonian_from_file(molecule_name):
    hamiltonian = []
    try:
        with open(hamiltonians_path + molecule_name + '.txt', 'r') as file:
            hamiltonian = [line.strip() for line in file]
    except FileNotFoundError:
        print(f"Hamiltonian file for {molecule_name} not found.")
    return hamiltonian
        

########
# QAOA #
########

class PauliGroupingInfoBase:
    def __init__(self, paulis: SparsePauliOp, A: list[list[int]], A_comp: list[list[int]], k: int, C: int, D: int):
        self.paulis = paulis
        self.A = A
        self.A_comp = A_comp
        self.n = len(A)
        self.degrees = [sum(row) for row in A_comp]
        self.colours  = [i for i in range(k)]
        self.vertices = [v for v in range(self.n)]
        self.C = C
        self.D = D

@dataclass
class PauliGroupingInfo:
    paulis: SparsePauliOp
    A: list[list[int]]
    A_comp: list[list[int]]
    n: int
    degrees: list[int]
    colours: list[int]
    vertices: list[int]
    C: int
    D: int
    active_paulis: SparsePauliOp | None = None
    active_A: list[list[int]] | None = None
    active_A_comp: list[list[int]] | None = None
    active_vertices: list[int] | None = None
    universally_commuting_vertices: list[int] | None = None
    raw_colours: list[int] | None = None
    repaired_colours: list[int] | None = None
    groups: list[list[int]] | None = None
    groups_by_colour: list[list[int]] | None = None
    raw_conflicts: int | None = None
    repaired_conflicts: int | None = None
    invalid_vertices: list[int] | None = None

def count_colour_conflicts(A, colours, num_colors):
    conflicts = 0
    for v, w in combinations(range(len(colours)), 2):
        if A[v][w] == 1 and 0 <= colours[v] < num_colors and colours[v] == colours[w]:
            conflicts += 1
    return conflicts

def local_conflicts(A, colours, vertex, colour, num_colors):
    conflicts = 0
    for neighbour, edge in enumerate(A[vertex]):
        if neighbour == vertex or edge != 1:
            continue
        neighbour_colour = colours[neighbour]
        if 0 <= neighbour_colour < num_colors and neighbour_colour == colour:
            conflicts += 1
    return conflicts

def repair_colouring(A, colours, num_colors):
    repaired = list(colours)
    degrees = [sum(row) for row in A]
    order = sorted(range(len(repaired)), key=lambda vertex: degrees[vertex], reverse=True)

    for vertex in order:
        if 0 <= repaired[vertex] < num_colors:
            continue
        repaired[vertex] = min(
            range(num_colors),
            key=lambda colour: local_conflicts(A, repaired, vertex, colour, num_colors),
        )

    for _ in range(max(1, len(repaired))):
        improved = False
        for vertex in order:
            current = repaired[vertex]
            best_colour = current
            best_conflicts = local_conflicts(A, repaired, vertex, current, num_colors)

            for colour in range(num_colors):
                if colour == current:
                    continue
                candidate_conflicts = local_conflicts(A, repaired, vertex, colour, num_colors)
                if candidate_conflicts < best_conflicts:
                    best_colour = colour
                    best_conflicts = candidate_conflicts

            if best_colour != current:
                repaired[vertex] = best_colour
                improved = True

        if not improved:
            break

    return repaired

def colouring_to_groups(colours, num_colors):
    return [group for group in colouring_to_groups_by_colour(colours, num_colors) if group]

def colouring_to_groups_by_colour(colours, num_colors):
    groups = defaultdict(list)
    for vertex, colour in enumerate(colours):
        if 0 <= colour < num_colors:
            groups[colour].append(vertex)
    return [groups[colour] for colour in range(num_colors)]

def decode_bitstring(bitstring, n, k):
    m = color_bit_width(k)
    little_endian_bits = bitstring[::-1]
    colours = []
    invalid_vertices = []

    for vertex in range(n):
        bits = [int(little_endian_bits[vertex * m + offset]) for offset in range(m)]
        colour = bits_to_color(bits)
        colours.append(colour)
        if colour >= k:
            invalid_vertices.append(vertex)

    return colours, invalid_vertices

def decode_assignment_bits(values, n, k):
    m = color_bit_width(k)
    colours = []
    invalid_vertices = []

    for vertex in range(n):
        bits = [int(round(values[vertex * m + offset])) for offset in range(m)]
        colour = bits_to_color(bits)
        colours.append(colour)
        if colour >= k:
            invalid_vertices.append(vertex)

    return colours, invalid_vertices

def populate_grouping_summary(pauli_grouping_info: PauliGroupingInfo, colours, invalid_vertices):
    active_vertices = pauli_grouping_info.active_vertices
    if active_vertices is None:
        active_vertices = pauli_grouping_info.vertices

    active_A_comp = pauli_grouping_info.active_A_comp
    if active_A_comp is None:
        active_A_comp = pauli_grouping_info.A_comp

    universally_commuting = pauli_grouping_info.universally_commuting_vertices or []
    num_colors = len(pauli_grouping_info.colours)
    repaired_active_colours = repair_colouring(active_A_comp, colours, num_colors)

    raw_colours = [-1] * pauli_grouping_info.n
    repaired_colours = [-1] * pauli_grouping_info.n

    for local_vertex, original_vertex in enumerate(active_vertices):
        raw_colours[original_vertex] = colours[local_vertex]
        repaired_colours[original_vertex] = repaired_active_colours[local_vertex]

    for original_vertex in universally_commuting:
        raw_colours[original_vertex] = 0
        repaired_colours[original_vertex] = 0

    universally_commuting_set = set(universally_commuting)
    groups_without_universal_terms = colouring_to_groups_by_colour(
        [
            colour if vertex not in universally_commuting_set else -1
            for vertex, colour in enumerate(repaired_colours)
        ],
        num_colors,
    )
    if universally_commuting:
        groups_without_universal_terms[0].extend(universally_commuting)

    pauli_grouping_info.raw_colours = raw_colours
    pauli_grouping_info.repaired_colours = repaired_colours
    pauli_grouping_info.groups_by_colour = groups_without_universal_terms
    pauli_grouping_info.groups = [group for group in groups_without_universal_terms if group]
    pauli_grouping_info.raw_conflicts = count_colour_conflicts(pauli_grouping_info.A_comp, raw_colours, num_colors)
    pauli_grouping_info.repaired_conflicts = count_colour_conflicts(pauli_grouping_info.A_comp, repaired_colours, num_colors)
    pauli_grouping_info.invalid_vertices = [active_vertices[index] for index in invalid_vertices]

def print_grouping_summary(pauli_grouping_info: PauliGroupingInfo):
    if pauli_grouping_info.repaired_colours is None:
        print("No decoded grouping summary available.")
        return

    if pauli_grouping_info.universally_commuting_vertices:
        print(
            "Universally commuting terms removed before QAOA and added back to colour 0:",
            pauli_grouping_info.universally_commuting_vertices,
        )

    print("\nDecoded colouring:")
    print(pauli_grouping_info.raw_colours)
    print("Conflicting non-QWC edges before repair:", pauli_grouping_info.raw_conflicts)

    if pauli_grouping_info.invalid_vertices:
        print("Invalid colour encodings repaired at vertices:", pauli_grouping_info.invalid_vertices)

    print("\nGreedy-repaired colouring:")
    print(pauli_grouping_info.repaired_colours)
    print("Conflicting non-QWC edges after repair:", pauli_grouping_info.repaired_conflicts)
    print("Measurement groups:", pauli_grouping_info.groups)

def apply_color_transition(qc, qubits, source_bits, target_bits, beta):
    diff = [i for i in range(len(qubits)) if source_bits[i] != target_bits[i]]
    if len(diff) != 1:
        raise ValueError("Adjacent Gray codewords must differ by exactly one bit.")

    target = qubits[diff[0]]
    controls = []

    for i in range(len(qubits)):
        if i == diff[0]:
            continue

        if source_bits[i] == 1:
            controls.append(qubits[i])
        else:
            qc.x(qubits[i])
            controls.append(qubits[i])

    if controls:
        qc.mcrx(2 * beta, controls, target)
    else:
        qc.rx(2 * beta, target)

    for i in range(len(qubits)):
        if i != diff[0] and source_bits[i] == 0:
            qc.x(qubits[i])

def gray_code_mixer(n, k, beta):
    m = color_bit_width(k)
    qc = QuantumCircuit(n * m)

    for v in range(n):
        qubits = [v * m + i for i in range(m)]

        for c in range(k - 1):
            apply_color_transition(
                qc,
                qubits,
                color_to_bits(c, m),
                color_to_bits(c + 1, m),
                beta,
            )

    return qc

def binary_coloring_mixer(n, k, beta):
    return gray_code_mixer(n, k, beta)

def binary_initial_state(n, k):
    m = color_bit_width(k)
    qc = QuantumCircuit(n * m)

    for v in range(n):
        bits = color_to_bits(v % k, m)
        for i, bit in enumerate(bits):
            if bit == 1:
                qc.x(v * m + i)

    return qc

def validate_direct_qaoa_dimensions(H, n, k, initial_state=None, mixer=None):
    expected_qubits = n * color_bit_width(k)
    if H.num_qubits != expected_qubits:
        raise ValueError(
            f"Direct Hamiltonian qubit mismatch: expected {expected_qubits}, got {H.num_qubits}. "
            "This usually means a QUBO-derived operator was passed instead of the direct space-efficient Hamiltonian."
        )

    if initial_state is not None and initial_state.num_qubits != expected_qubits:
        raise ValueError(
            f"Initial state qubit mismatch: expected {expected_qubits}, got {initial_state.num_qubits}."
        )

    if mixer is not None and mixer.num_qubits != expected_qubits:
        raise ValueError(
            f"Mixer qubit mismatch: expected {expected_qubits}, got {mixer.num_qubits}."
        )

def run_qaoa(qp: QuadraticProgram, reps: int, n: int, k: int, optimization_level: int, service: QiskitRuntimeService | None, simulation: bool = True):
    algorithm_globals.random_seed = 42
    result = None
    if simulation == False:
        backend = service.least_busy(simulator=False)
        print(backend)

        sampler = Sampler(mode=backend)
        optimizer = SPSA(maxiter=100)
        # optimizer = COBYLA()

        # Build the hybrid QAOA solver
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            translation_method='translator',
            routing_method='sabre'
        )
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=reps,
            pass_manager=pm
        )

        # Wrap QAOA as an optimizer for QuadraticProgram
        solver = MinimumEigenOptimizer(qaoa)

        # Solve
        result = solver.solve(qp)
    
    else:
        qubo = QuadraticProgramToQubo().convert(qp)
        num_qubits = len(qubo.variables)
        expected_qubits = n * color_bit_width(k)
        if num_qubits != expected_qubits:
            raise ValueError(
                f"The QUBO formulation expanded to {num_qubits} variables, but the custom initial state and mixer are "
                f"built for the direct space-efficient encoding with {expected_qubits} qubits. "
                "If you intended to use the direct Hamiltonian, call pauli_grouping(..., ham_repr=True)."
            )
        initial_point = [0.1] * (2 * reps)

        if num_qubits <= 20:
            qaoa = QAOA(
                sampler=StatevectorSampler(),
                optimizer=COBYLA(maxiter=200),
                reps=reps,
                initial_state=binary_initial_state(n, k),
                mixer=gray_code_mixer(n, k, beta=Parameter("beta")),
                initial_point=initial_point,
            )
            optimizer = MinimumEigenOptimizer(qaoa)
            return optimizer.solve(qubo)

        backend_options = dict(
            method="matrix_product_state",
            max_parallel_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            matrix_product_state_max_bond_dimension=32,
            matrix_product_state_truncation_threshold=1e-8,
            max_memory_mb=700000,
        )

        spsa = SPSA(
            maxiter=300,
            learning_rate=0.05,
            perturbation=0.1,
            blocking=True,
            allowed_increase=0.01
        )

        qaoa = QAOA(
            sampler=AerSampler(
                run_options={'shots': 256},
                backend_options=backend_options
            ),
            optimizer=spsa,
            reps=reps,
            initial_state=binary_initial_state(n, k),
            mixer=gray_code_mixer(n, k, beta=Parameter("beta")),
            initial_point=initial_point,
        )

        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qubo)
    return result

def run_qaoa_H(H, n, k, reps=2, service: QiskitRuntimeService | None = None, optimization_level: int = 2, simulation: bool = True):
    algorithm_globals.random_seed = 42
    initial_point = [0.1] * (2 * reps)
    initial_state = binary_initial_state(n, k)
    mixer = gray_code_mixer(n, k, beta=Parameter("beta"))
    validate_direct_qaoa_dimensions(H, n, k, initial_state=initial_state, mixer=mixer)
    print('Number of qubits:', H.num_qubits)

    if simulation == False:
        backend = service.least_busy(simulator=False)
        print(backend)

        sampler = Sampler(mode=backend)
        optimizer = SPSA(maxiter=100)
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            translation_method='translator',
            routing_method='sabre'
        )

        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=reps,
            pass_manager=pm,
            initial_state=initial_state,
            mixer=mixer,
            initial_point=initial_point,
        )
        return qaoa.compute_minimum_eigenvalue(H)

    if H.num_qubits <= 20:
        qaoa = QAOA(
            sampler=StatevectorSampler(),
            optimizer=COBYLA(maxiter=200),
            reps=reps,
            initial_state=initial_state,
            mixer=mixer,
            initial_point=initial_point,
        )
        return qaoa.compute_minimum_eigenvalue(H)

    backend_options = dict(
        method="matrix_product_state",
        max_parallel_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        matrix_product_state_max_bond_dimension=32,
        matrix_product_state_truncation_threshold=1e-8,
        max_memory_mb=700000,
    )

    spsa = SPSA(
        maxiter=300,
        learning_rate=0.05,
        perturbation=0.1,
        blocking=True,
        allowed_increase=0.01
    )

    qaoa = QAOA(
        sampler=AerSampler(
            run_options={'shots': 256},
            backend_options=backend_options
        ),
        optimizer=spsa,
        reps=reps,
        initial_state=initial_state,
        mixer=mixer,
        initial_point=initial_point,
    )

    return qaoa.compute_minimum_eigenvalue(H)

def get_results(pauli_grouping_info: PauliGroupingInfo, job_result: dict | None = None, service: QiskitRuntimeService | None = None, job_id: str | None = None):
    if service is not None and job_id is not None:
        job = service.job(job_id)
        job_result = job.result()
    
    if job_result is None:
        raise ValueError("Either job_result or both service and job_id must be provided.")

    counts = job_result[0].data.meas.get_counts()
    highest_counts = get_modes(counts)

    costs = {}
    for count in highest_counts:
        # print('Bitstring:', count)
        cost_val = cost(count, pauli_grouping_info.A, pauli_grouping_info.vertices, pauli_grouping_info.colours, pauli_grouping_info.degrees, pauli_grouping_info.C, pauli_grouping_info.D)
        costs[count] = cost_val
        # print('Cost:', cost_val)

    opt_sol_found = get_smallest_cost(costs)
    print('Optimal solution found:', opt_sol_found)
    print('Cost:', costs[opt_sol_found])

    return (opt_sol_found, costs[opt_sol_found])

#########################
# Running QAOA on a QPU #
#########################

def pauli_grouping(paulis: SparsePauliOp, C: int, D: int, k: int, reps: int, optimization_level: int, api_key: str, simulation: bool = True, ham_repr: bool = False):
    '''
    Group the Pauli words in a SparsePauliOp object into QWC groups.
    Returns a list of lists, where each inner list contains the indices of the Pauli words that belong to the same QWC group.
    '''

    A, active_vertices, universally_commuting_vertices = split_universally_commuting_terms(paulis)
    A_comp = complement_graph(A)
    active_paulis = sparse_pauli_subset(paulis, active_vertices) if active_vertices else None
    active_A = [[A[i][j] for j in active_vertices] for i in active_vertices]
    active_A_comp = complement_graph(active_A)

    pauli_grouping_info = PauliGroupingInfo(
        paulis=paulis,
        A=A,
        A_comp=A_comp,
        n=len(A),
        degrees=[sum(row) for row in A_comp],
        colours=[i for i in range(k)],
        vertices=[v for v in range(len(A))],
        C=C,
        D=D,
        active_paulis=active_paulis,
        active_A=active_A,
        active_A_comp=active_A_comp,
        active_vertices=active_vertices,
        universally_commuting_vertices=universally_commuting_vertices,
    )
    
    service = None
    if simulation == False:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=api_key
        )

    if not active_vertices:
        populate_grouping_summary(pauli_grouping_info, [], [])
        if ham_repr:
            return (
                pauli_grouping_info,
                SimpleNamespace(
                    best_measurement={"bitstring": "", "value": 0.0},
                    cost_function_evals=0,
                ),
            )
        return (
            pauli_grouping_info,
            SimpleNamespace(
                samples=[],
                x=[],
                prettyprint=lambda: "All Pauli terms commute with one another; no QAOA run was required.",
            ),
        )
    
    if ham_repr:
        H = build_cost_hamiltonian(
            active_paulis,
            k,
            invalid_penalty=0.0,
            conflict_penalty=D,
        )
        result = run_qaoa_H(
            H,
            len(active_vertices),
            k,
            reps,
            service=service,
            optimization_level=optimization_level,
            simulation=simulation,
        )

        best_measurement = getattr(result, "best_measurement", None)
        if best_measurement is not None:
            colours, invalid_vertices = decode_bitstring(
                best_measurement["bitstring"],
                len(active_vertices),
                k,
            )
            populate_grouping_summary(pauli_grouping_info, colours, invalid_vertices)
    else:
        # Keep the QUBO path isolated from the direct-Hamiltonian path so the
        # operator ansatz cannot accidentally inherit the quadratized problem size.
        qp = space_efficient_qp(active_A_comp, num_colors=k)
        result = run_qaoa(qp, reps, len(active_vertices), k, optimization_level, service)
        assignment = getattr(result, "x", None)
        if assignment is not None:
            colours, invalid_vertices = decode_assignment_bits(
                assignment,
                len(active_vertices),
                k,
            )
            populate_grouping_summary(pauli_grouping_info, colours, invalid_vertices)

    return (pauli_grouping_info, result)
    
    # # Group indices by colour
    # groups = defaultdict(list)
    # for i, colour in enumerate(colours):
    #     groups[colour].append(i)

    # return list(groups.values())

def substitute_solution(qp, subs):
    # subs = {
    #     'x_0_0': 1,
    #     'x_0_1': 0,
    #     'x_1_0': 1,
    #     'x_1_1': 0,
    #     'x_2_0': 1,
    #     'x_2_1': 0,
    #     # etc.
    # }

    print('Optimal solution:', qp.substitute_variables(subs))


# Variable Definitions

C = 10
D = 1
reps = 3
optimization_level = 2
