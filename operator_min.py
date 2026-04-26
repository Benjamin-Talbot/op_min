api_key = ''
# hamiltonians_path = '/home/btalbot/projects/def-stijn/btalbot/op_min/hamiltonians/' # Siku
# hamiltonians_path = '/home/btalbot/links/projects/def-stijn/btalbot/op_min/hamiltonians/' # Trillium
# hamiltonians_path = '/home/btalbot/scratch/honours_code/one-hot-encoding/hamiltonians/' # Fir
hamiltonians_path = './hamiltonians/'

simulation = True
# Set to False to keep fully commuting terms in the QAOA optimization problem.
remove_fully_commuting_terms = True
print_qaoa_circuit_diagram = False
save_qaoa_circuit_diagram = True
qaoa_circuit_output_dir = "results/qaoa_circuit_diagrams"

####################################
# Function and Problem Definitions #
####################################

# Imports

from itertools import combinations
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from random import randint
from types import SimpleNamespace
import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import QAOAAnsatz, XXPlusYYGate
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
    best_bitstring: str | None = None
    best_measurement_weight: float | None = None
    qubo_objective: float | None = None
    result_source: str | None = None

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

def bitstring_to_vector(bitstring: str, num_variables: int):
    clean = bitstring.replace(" ", "")
    if len(clean) != num_variables:
        raise ValueError(
            f"Expected a {num_variables}-bit string, but received {len(clean)} bits: {bitstring!r}"
        )

    # Qiskit count strings are displayed with qubit 0 on the right.
    return [int(bit) for bit in reversed(clean)]

def normalize_measurement_map(measurements, num_variables: int):
    if measurements is None:
        return None

    if hasattr(measurements, "binary_probabilities"):
        try:
            measurements = measurements.binary_probabilities(num_variables)
        except TypeError:
            measurements = measurements.binary_probabilities()

    if not isinstance(measurements, Mapping):
        return None

    if "bitstring" in measurements and isinstance(measurements["bitstring"], str):
        weight = measurements.get("probability", measurements.get("count", 1))
        return {measurements["bitstring"].replace(" ", ""): weight}

    normalized = {}
    for key, value in measurements.items():
        if isinstance(key, str):
            bitstring = key.replace(" ", "")
        elif isinstance(key, int):
            bitstring = format(key, f"0{num_variables}b")
        else:
            continue

        if len(bitstring) != num_variables:
            continue

        normalized[bitstring] = value

    return normalized or None

def extract_measurement_map(payload, num_variables: int, _visited=None):
    if payload is None:
        return None

    if _visited is None:
        _visited = set()

    payload_id = id(payload)
    if payload_id in _visited:
        return None
    _visited.add(payload_id)

    normalized = normalize_measurement_map(payload, num_variables)
    if normalized is not None:
        return normalized

    if hasattr(payload, "get_counts"):
        try:
            normalized = normalize_measurement_map(payload.get_counts(), num_variables)
        except TypeError:
            normalized = None
        if normalized is not None:
            return normalized

    if hasattr(payload, "data"):
        data = getattr(payload, "data")
        for attribute in ("meas", "c", "bits"):
            register = getattr(data, attribute, None)
            normalized = extract_measurement_map(register, num_variables, _visited)
            if normalized is not None:
                return normalized

    if hasattr(payload, "min_eigen_solver_result"):
        min_eigen_solver_result = getattr(payload, "min_eigen_solver_result")
        for attribute in ("eigenstate", "best_measurement", "quasi_distribution"):
            normalized = extract_measurement_map(
                getattr(min_eigen_solver_result, attribute, None),
                num_variables,
                _visited,
            )
            if normalized is not None:
                return normalized

    if isinstance(payload, (list, tuple)):
        for item in payload:
            normalized = extract_measurement_map(item, num_variables, _visited)
            if normalized is not None:
                return normalized
        return None

    try:
        first_item = payload[0]
    except Exception:
        return None
    return extract_measurement_map(first_item, num_variables, _visited)

def decode_measurement_results(measurement_map, qubo: QuadraticProgram, pauli_grouping_info: PauliGroupingInfo):
    best_summary = None

    for bitstring, weight in measurement_map.items():
        vector = bitstring_to_vector(bitstring, len(qubo.variables))
        result_like = SimpleNamespace(x=vector, samples=[SimpleNamespace(x=vector)])
        summary = postprocess_one_hot_result(
            result_like,
            qubo,
            pauli_grouping_info.A_comp,
            pauli_grouping_info.active_A_comp,
            pauli_grouping_info.active_vertices,
            pauli_grouping_info.fully_commuting_vertices,
            len(pauli_grouping_info.colours),
        )
        if summary is None:
            continue

        summary["best_bitstring"] = bitstring
        summary["best_measurement_weight"] = float(weight)
        summary["qubo_objective"] = qubo.objective.evaluate(vector)
        rank = (
            summary["repaired_conflicts"],
            len(summary["invalid_vertices"]),
            summary["raw_conflicts"],
            summary["qubo_objective"],
            -summary["best_measurement_weight"],
        )
        if best_summary is None or rank < best_summary[0]:
            best_summary = (rank, summary)

    return best_summary[1] if best_summary is not None else None

def apply_postprocessed_summary(pauli_grouping_info: PauliGroupingInfo, summary, source: str):
    if summary is None:
        return

    pauli_grouping_info.raw_colours = summary["raw_colours"]
    pauli_grouping_info.repaired_colours = summary["repaired_colours"]
    pauli_grouping_info.groups_by_colour = summary["groups_by_colour"]
    pauli_grouping_info.groups = summary["groups"]
    pauli_grouping_info.raw_conflicts = summary["raw_conflicts"]
    pauli_grouping_info.repaired_conflicts = summary["repaired_conflicts"]
    pauli_grouping_info.invalid_vertices = summary["invalid_vertices"]
    pauli_grouping_info.best_bitstring = summary.get("best_bitstring")
    pauli_grouping_info.best_measurement_weight = summary.get("best_measurement_weight")
    pauli_grouping_info.qubo_objective = summary.get("qubo_objective")
    pauli_grouping_info.result_source = source

def resolve_remove_fully_commuting_terms(remove_fully_commuting_terms_enabled: bool | None = None):
    if remove_fully_commuting_terms_enabled is None:
        return remove_fully_commuting_terms
    return remove_fully_commuting_terms_enabled

def prepare_grouping_problem(
    paulis: SparsePauliOp,
    k: int,
    C: int,
    D: int,
    simulation: bool,
    remove_fully_commuting_terms_enabled: bool | None = None,
):
    if resolve_remove_fully_commuting_terms(remove_fully_commuting_terms_enabled):
        A, active_vertices, fully_commuting_vertices = split_fully_commuting_terms(paulis)
    else:
        A = generate_qwc_graph(paulis)
        active_vertices = [index for index in range(len(A))]
        fully_commuting_vertices = []

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

    qp = graph_colouring_qubo_qp(active_A_comp, k, C, D)
    return pauli_grouping_info, qp

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

def build_qaoa_components(n: int, k: int, reps: int):
    mixer_parameter = Parameter("beta")
    mixer_circuit = xy_mixer(n, k, beta=mixer_parameter)

    # Avoid SPSA auto-calibration, which can produce NaNs when the initial
    # gradient estimate is numerically flat on small/noisy instances.
    optimizer = SPSA(
        maxiter=100,
        learning_rate=0.05,
        perturbation=0.05,
        trust_region=True,
    )
    initial_point = np.array([0.1] * (2 * reps), dtype=float)

    return optimizer, mixer_circuit, initial_point

def export_qaoa_circuit_diagram(
    qubo: QuadraticProgram,
    reps: int,
    initial_state_circuit: QuantumCircuit,
    mixer_circuit: QuantumCircuit,
    output_dir: Path | str = qaoa_circuit_output_dir,
    print_diagram: bool = print_qaoa_circuit_diagram,
    save_diagram: bool = save_qaoa_circuit_diagram,
    transpiled_circuit: QuantumCircuit | None = None,
):
    if not print_diagram and not save_diagram:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cost_operator, _ = qubo.to_ising()
    circuit = QAOAAnsatz(
        cost_operator=cost_operator,
        reps=reps,
        initial_state=initial_state_circuit,
        mixer_operator=mixer_circuit,
    ).decompose(reps=10)

    saved_paths = []

    if save_diagram:
        logical_path = output_dir / f"{qubo.name}_qaoa_reps_{reps}_logical.png"
        circuit.draw(output="mpl", filename=str(logical_path), fold=-1)
        saved_paths.append(logical_path)

        if print_diagram:
            print(f"Saved logical QAOA circuit diagram to {logical_path.resolve()}")

        if transpiled_circuit is not None:
            transpiled_path = output_dir / f"{qubo.name}_qaoa_reps_{reps}_transpiled.png"
            transpiled_circuit.draw(output="mpl", filename=str(transpiled_path), fold=-1)
            saved_paths.append(transpiled_path)

    if saved_paths:
        print("Saved QAOA circuit diagram(s):")
        for path in saved_paths:
            print(path.resolve())

    return circuit

def run_qaoa(qp: QuadraticProgram, reps: int, n: int, k: int, optimization_level: int, service: QiskitRuntimeService | None, simulation: bool = True):
    algorithm_globals.random_seed = 42
    qubo = QuadraticProgramToQubo().convert(qp)
    result = None
    optimizer, mixer_circuit, initial_point = build_qaoa_components(n, k, reps)
    initial_state_circuit = initial_state(n, k)
    if simulation == False:
        if service is None:
            raise ValueError("A QiskitRuntimeService instance is required when simulation=False.")

        backend = service.least_busy(simulator=False, operational=True, min_num_qubits=n * k)
        print(backend)

        sampler = Sampler(mode=backend)

        # Build the hybrid QAOA solver
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            translation_method='translator',
            routing_method='sabre'
        )
        transpiled_preview = pm.run(
            QAOAAnsatz(
                cost_operator=qubo.to_ising()[0],
                reps=reps,
                initial_state=initial_state_circuit,
                mixer_operator=mixer_circuit,
            )
        )
        export_qaoa_circuit_diagram(
            qubo,
            reps,
            initial_state_circuit,
            mixer_circuit,
            transpiled_circuit=transpiled_preview,
        )
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=reps,
            initial_state=initial_state_circuit,
            mixer=mixer_circuit,
            initial_point=initial_point,
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
            optimizer=optimizer,
            reps=reps,
            initial_state=initial_state_circuit,
            mixer=mixer_circuit,
            initial_point=initial_point,
        )
        export_qaoa_circuit_diagram(
            qubo,
            reps,
            initial_state_circuit,
            mixer_circuit,
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

    qp = graph_colouring_qubo_qp(
        pauli_grouping_info.active_A_comp,
        len(pauli_grouping_info.colours),
        pauli_grouping_info.C,
        pauli_grouping_info.D,
    )
    qubo = QuadraticProgramToQubo().convert(qp)

    measurement_map = extract_measurement_map(job_result, len(qubo.variables))
    if measurement_map is None:
        raise TypeError("Could not extract measurement counts or probabilities from the IBM result object.")

    postprocessed = decode_measurement_results(measurement_map, qubo, pauli_grouping_info)
    if postprocessed is None:
        raise ValueError("Unable to decode the IBM result into a one-hot colouring.")

    apply_postprocessed_summary(pauli_grouping_info, postprocessed, source="ibm_measurements")

    print('Best measured bitstring:', pauli_grouping_info.best_bitstring)
    print('Measurement weight:', pauli_grouping_info.best_measurement_weight)
    print('QUBO objective:', pauli_grouping_info.qubo_objective)

    return (pauli_grouping_info.best_bitstring, pauli_grouping_info.qubo_objective)


#########################
# Running QAOA on a QPU #
#########################

def pauli_grouping(
    paulis: SparsePauliOp,
    C: int,
    D: int,
    k: int,
    reps: int,
    optimization_level: int,
    api_key: str,
    simulation: bool = True,
    remove_fully_commuting_terms_enabled: bool | None = None,
):
    '''
    Group the Pauli words in a SparsePauliOp object into QWC groups.
    Returns a list of lists, where each inner list contains the indices of the Pauli words that belong to the same QWC group.
    '''
    pauli_grouping_info, qp = prepare_grouping_problem(
        paulis,
        k,
        C,
        D,
        simulation,
        remove_fully_commuting_terms_enabled=remove_fully_commuting_terms_enabled,
    )

    if not pauli_grouping_info.active_vertices:
        groups_by_colour = [[] for _ in range(k)]
        if k > 0:
            groups_by_colour[0] = list(pauli_grouping_info.fully_commuting_vertices)
        pauli_grouping_info.raw_colours = [0] * pauli_grouping_info.n
        pauli_grouping_info.repaired_colours = [0] * pauli_grouping_info.n
        pauli_grouping_info.groups_by_colour = groups_by_colour
        pauli_grouping_info.groups = [group for group in groups_by_colour if group]
        pauli_grouping_info.raw_conflicts = 0
        pauli_grouping_info.repaired_conflicts = 0
        pauli_grouping_info.invalid_vertices = []
        pauli_grouping_info.result_source = "shortcut"
        return (
            pauli_grouping_info,
            SimpleNamespace(
                samples=[],
                x=[],
                prettyprint=lambda: "All Pauli terms commute with one another; no optimization was required.",
            ),
        )

    service = None
    if simulation == False:
        service_kwargs = {"channel": "ibm_quantum_platform"}
        if api_key:
            service_kwargs["token"] = api_key
        service = QiskitRuntimeService(**service_kwargs)

    result, qubo = run_qaoa(
        qp,
        reps,
        len(pauli_grouping_info.active_vertices),
        k,
        optimization_level,
        service,
        simulation=simulation,
    )

    measurement_map = extract_measurement_map(result, len(qubo.variables))
    if measurement_map is not None:
        postprocessed = decode_measurement_results(measurement_map, qubo, pauli_grouping_info)
        apply_postprocessed_summary(pauli_grouping_info, postprocessed, source="ibm_measurements" if simulation == False else "measurements")
    else:
        postprocessed = postprocess_one_hot_result(
            result,
            qubo,
            pauli_grouping_info.A_comp,
            pauli_grouping_info.active_A_comp,
            pauli_grouping_info.active_vertices,
            pauli_grouping_info.fully_commuting_vertices,
            k,
        )
        apply_postprocessed_summary(pauli_grouping_info, postprocessed, source="optimizer_samples")

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
