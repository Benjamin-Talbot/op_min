api_key = ''
# hamiltonians_path = '/home/btalbot/projects/def-stijn/btalbot/op_min/hamiltonians/' # Siku
# hamiltonians_path = '/home/btalbot/links/projects/def-stijn/btalbot/op_min/hamiltonians/' # Trillium
# hamiltonians_path = '/home/btalbot/scratch/honours_code/one-hot-encoding/hamiltonians/' # Fir
hamiltonians_path = './hamiltonians/'

simulation = False

####################################
# Function and Problem Definitions #
####################################

# Imports

from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass
from random import randint
from types import SimpleNamespace

from qiskit import transpile, QuantumCircuit
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

def subgraph_from_indices(A, indices):
    return [[A[i][j] for j in indices] for i in indices]

def split_fully_commuting_terms(paulis: SparsePauliOp):
    A = generate_qwc_graph(paulis)
    fully_commuting_vertices = [
        index for index, row in enumerate(A) if sum(row) == len(A) - 1
    ]
    fully_commuting_set = set(fully_commuting_vertices)
    active_vertices = [
        index for index in range(len(A)) if index not in fully_commuting_set
    ]
    return A, active_vertices, fully_commuting_vertices


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
    active_vertices: list[int] | None = None
    active_A_comp: list[list[int]] | None = None
    fully_commuting_vertices: list[int] | None = None
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
        if vertex == neighbour or edge != 1:
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

def extract_variable_values(x, variable_names):
    return {
        name: int(round(value))
        for name, value in zip(variable_names, x)
    }

def decode_one_hot_colouring(variable_values, n, num_colors):
    colours = []
    invalid_vertices = []

    for vertex in range(n):
        active_colours = [
            colour
            for colour in range(num_colors)
            if variable_values.get(f"x_{vertex}_{colour}", 0) == 1
        ]

        if len(active_colours) == 1:
            colours.append(active_colours[0])
        else:
            colours.append(-1)
            invalid_vertices.append(vertex)

    return colours, invalid_vertices

def postprocess_one_hot_result(
    result,
    qubo: QuadraticProgram,
    full_A_comp,
    active_A_comp,
    active_vertices,
    fully_commuting_vertices,
    num_colors,
):
    variable_names = [variable.name for variable in qubo.variables]
    candidate_vectors = []

    if getattr(result, "samples", None):
        candidate_vectors.extend(sample.x for sample in result.samples)
    if getattr(result, "x", None) is not None:
        candidate_vectors.append(result.x)

    best_summary = None
    for vector in candidate_vectors:
        variable_values = extract_variable_values(vector, variable_names)
        raw_active_colours, invalid_active_vertices = decode_one_hot_colouring(
            variable_values,
            len(active_vertices),
            num_colors,
        )
        repaired_active_colours = repair_colouring(active_A_comp, raw_active_colours, num_colors)

        raw_colours = [-1] * len(full_A_comp)
        repaired_colours = [-1] * len(full_A_comp)

        for local_vertex, original_vertex in enumerate(active_vertices):
            raw_colours[original_vertex] = raw_active_colours[local_vertex]
            repaired_colours[original_vertex] = repaired_active_colours[local_vertex]

        for original_vertex in fully_commuting_vertices:
            raw_colours[original_vertex] = 0
            repaired_colours[original_vertex] = 0

        fully_commuting_set = set(fully_commuting_vertices)
        groups_by_colour = colouring_to_groups_by_colour(
            [
                colour if vertex not in fully_commuting_set else -1
                for vertex, colour in enumerate(repaired_colours)
            ],
            num_colors,
        )
        if fully_commuting_vertices:
            groups_by_colour[0].extend(fully_commuting_vertices)

        summary = {
            "raw_colours": raw_colours,
            "repaired_colours": repaired_colours,
            "groups_by_colour": groups_by_colour,
            "groups": [group for group in groups_by_colour if group],
            "raw_conflicts": count_colour_conflicts(full_A_comp, raw_colours, num_colors),
            "repaired_conflicts": count_colour_conflicts(full_A_comp, repaired_colours, num_colors),
            "invalid_vertices": [active_vertices[index] for index in invalid_active_vertices],
        }
        rank = (
            summary["repaired_conflicts"],
            len(summary["invalid_vertices"]),
            summary["raw_conflicts"],
        )
        if best_summary is None or rank < best_summary[0]:
            best_summary = (rank, summary)

    return best_summary[1] if best_summary is not None else None

def xy_mixer(n, k, beta):
    n_qubits = n * k
    qc = QuantumCircuit(n_qubits)
    
    for v in range(n):
        # apply XY interactions between all colour pairs
        for c1, c2 in combinations(range(k), 2):
            q1 = v * k + c1
            q2 = v * k + c2
            
            qc.append(XXPlusYYGate(2 * beta), [q1, q2])
    
    return qc

def initial_state(n: int, k: int):
    qc = QuantumCircuit(n * k)
    
    for v in range(n):
        c = v % k  # simple valid assignment
        qc.x(v * k + c)
    
    return qc

def run_qaoa(qp: QuadraticProgram, reps: int, n: int, k: int, optimization_level: int, service: QiskitRuntimeService | None, simulation: bool = True):
    algorithm_globals.random_seed = 42
    qubo = QuadraticProgramToQubo().convert(qp)
    result = None
    if simulation == False:
        if service is None:
            raise ValueError("A QiskitRuntimeService instance is required when simulation=False.")

        backend = service.least_busy(simulator=False, operational=True, min_num_qubits=n * k)
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
            initial_state=initial_state(n, k),
            mixer=xy_mixer(n, k, beta=0.5),
            pass_manager=pm
        )

        # Wrap QAOA as an optimizer for QuadraticProgram
        solver = MinimumEigenOptimizer(qaoa)

        # Solve
        result = solver.solve(qubo)
    
    else:
        backend_options = dict(
            method="matrix_product_state",
            max_parallel_threads=64,
            matrix_product_state_max_bond_dimension=32,
            matrix_product_state_truncation_threshold=1e-8,
            max_memory_mb=0,
        )
        
        qaoa = QAOA(
            sampler=AerSampler(
                run_options={'shots': 256},
                backend_options=backend_options
            ),
            optimizer=SPSA(),
            reps=reps,
            initial_state=initial_state(n, k),
            mixer=xy_mixer(n, k, beta=0.5)
        )

        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qubo)
    return result, qubo

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

def pauli_grouping(paulis: SparsePauliOp, C: int, D: int, k: int, reps: int, optimization_level: int, api_key: str, simulation: bool = True):
    '''
    Group the Pauli words in a SparsePauliOp object into QWC groups.
    Returns a list of lists, where each inner list contains the indices of the Pauli words that belong to the same QWC group.
    '''

    A, active_vertices, fully_commuting_vertices = split_fully_commuting_terms(paulis)
    A_comp = complement_graph(A)
    active_A_comp = complement_graph(subgraph_from_indices(A, active_vertices))

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
        active_vertices=active_vertices,
        active_A_comp=active_A_comp,
        fully_commuting_vertices=fully_commuting_vertices,
    )

    service = None
    if simulation == False:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=api_key
        )

    if not active_vertices:
        groups_by_colour = [[] for _ in range(k)]
        if k > 0:
            groups_by_colour[0] = list(fully_commuting_vertices)
        pauli_grouping_info.raw_colours = [0] * len(A)
        pauli_grouping_info.repaired_colours = [0] * len(A)
        pauli_grouping_info.groups_by_colour = groups_by_colour
        pauli_grouping_info.groups = [group for group in groups_by_colour if group]
        pauli_grouping_info.raw_conflicts = 0
        pauli_grouping_info.repaired_conflicts = 0
        pauli_grouping_info.invalid_vertices = []
        return (
            pauli_grouping_info,
            SimpleNamespace(
                samples=[],
                x=[],
                prettyprint=lambda: "All Pauli terms commute with one another; no optimization was required.",
            ),
        )

    qp = graph_colouring_qubo_qp(active_A_comp, k, C, D)
    result, qubo = run_qaoa(qp, reps, len(active_vertices), k, optimization_level, service, simulation=simulation)

    postprocessed = postprocess_one_hot_result(
        result,
        qubo,
        A_comp,
        active_A_comp,
        active_vertices,
        fully_commuting_vertices,
        k,
    )
    if postprocessed is not None:
        pauli_grouping_info.raw_colours = postprocessed["raw_colours"]
        pauli_grouping_info.repaired_colours = postprocessed["repaired_colours"]
        pauli_grouping_info.groups_by_colour = postprocessed["groups_by_colour"]
        pauli_grouping_info.groups = postprocessed["groups"]
        pauli_grouping_info.raw_conflicts = postprocessed["raw_conflicts"]
        pauli_grouping_info.repaired_conflicts = postprocessed["repaired_conflicts"]
        pauli_grouping_info.invalid_vertices = postprocessed["invalid_vertices"]

    return (pauli_grouping_info, result)
    
    # # Group indices by colour
    # groups = defaultdict(list)
    # for i, colour in enumerate(colours):
    #     groups[colour].append(i)

    # return list(groups.values())


# Variable Definitions

C = 100
D = 10
reps = 3
optimization_level = 2
