"""
Fetch and decode an IBM Runtime result for the one-hot graph-colouring workflow.

Example:
    python fetch_ibm_results.py --job-id <JOB_ID> --hamiltonian H2_BK --k 3 --api-key <TOKEN>
"""

from __future__ import annotations

import argparse

from qiskit_ibm_runtime import QiskitRuntimeService

from operator_min import (
    C as DEFAULT_C,
    D as DEFAULT_D,
    SparsePauliOp,
    api_key as DEFAULT_API_KEY,
    apply_postprocessed_summary,
    decode_measurement_results,
    extract_measurement_map,
    hamiltonians_path,
    prepare_grouping_problem,
    read_hamiltonian_from_file,
)


def print_report(pauli_grouping_info, top_measurements: int, measurement_map):
    print("Best measured bitstring:", pauli_grouping_info.best_bitstring)
    print("Measurement weight:", pauli_grouping_info.best_measurement_weight)
    print("QUBO objective:", pauli_grouping_info.qubo_objective)
    print("Raw colours:", pauli_grouping_info.raw_colours)
    print("Repaired colours:", pauli_grouping_info.repaired_colours)
    print("Invalid vertices:", pauli_grouping_info.invalid_vertices)
    print("Raw conflicts:", pauli_grouping_info.raw_conflicts)
    print("Repaired conflicts:", pauli_grouping_info.repaired_conflicts)
    print()

    print("Groups by colour:")
    for colour, group in enumerate(pauli_grouping_info.groups_by_colour or []):
        terms = [pauli_grouping_info.paulis.paulis[index].to_label() for index in group]
        print(f"Colour {colour}: indices={group}, terms={terms}")

    print()
    print(f"Top {min(top_measurements, len(measurement_map))} measured bitstrings:")
    for bitstring, weight in sorted(measurement_map.items(), key=lambda item: (-item[1], item[0]))[:top_measurements]:
        print(f"{bitstring}: {weight}")


def parse_args():
    parser = argparse.ArgumentParser(description="Decode graph-colouring QAOA results from an IBM Runtime job.")
    parser.add_argument("--job-id", required=True, help="IBM Runtime job ID to fetch and decode.")
    parser.add_argument(
        "--hamiltonian",
        default="H2_BK",
        help="Hamiltonian filename stem inside the hamiltonians directory, for example H2_BK.",
    )
    parser.add_argument("--k", type=int, required=True, help="Number of colours used in the one-hot encoding.")
    parser.add_argument("--C", type=int, default=DEFAULT_C, help="Penalty coefficient for one-hot validity.")
    parser.add_argument("--D", type=int, default=DEFAULT_D, help="Penalty coefficient for colour conflicts.")
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="IBM Quantum Platform API key. If omitted, Qiskit will use a saved account if one exists.",
    )
    parser.add_argument(
        "--channel",
        default="ibm_quantum_platform",
        help="Runtime channel to use when connecting to IBM Quantum.",
    )
    parser.add_argument(
        "--top-counts",
        type=int,
        default=10,
        help="How many of the most frequent measured bitstrings to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hamiltonian = read_hamiltonian_from_file(args.hamiltonian)
    if not hamiltonian:
        raise FileNotFoundError(
            f"No Hamiltonian terms were loaded for {args.hamiltonian!r} from {hamiltonians_path!r}."
        )

    paulis = SparsePauliOp.from_list([(term, 1.0) for term in hamiltonian])
    pauli_grouping_info, qp = prepare_grouping_problem(paulis, args.k, args.C, args.D, simulation=False)

    service_kwargs = {"channel": args.channel}
    if args.api_key:
        service_kwargs["token"] = args.api_key
    service = QiskitRuntimeService(**service_kwargs)

    job = service.job(args.job_id)
    job_result = job.result()

    from qiskit_optimization.converters import QuadraticProgramToQubo
    qubo = QuadraticProgramToQubo().convert(qp)

    measurement_map = extract_measurement_map(job_result, len(qubo.variables))
    if measurement_map is None:
        raise TypeError("Could not extract measurement counts or probabilities from the IBM result object.")

    postprocessed = decode_measurement_results(measurement_map, qubo, pauli_grouping_info)
    if postprocessed is None:
        raise ValueError("Unable to decode the IBM result into a one-hot colouring.")

    apply_postprocessed_summary(pauli_grouping_info, postprocessed, source="ibm_measurements")
    print_report(pauli_grouping_info, args.top_counts, measurement_map)


if __name__ == "__main__":
    main()
