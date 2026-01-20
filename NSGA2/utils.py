import numpy as np
import csv
import math
import os
import json

from pathlib import Path
from typing import Tuple
from NSGA2.src.random_initialization import RandomInitializer
from pymoo.indicators.gd import GD
from pymoo.util.misc import cdist
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from plot import plot_pareto_front_2d, plot_pareto_front_3d_matplotlib
from scheduling_environment.jobShop import JobShop


def select_best_solution(
        pareto_front: np.ndarray,
        pareto_solutions: np.ndarray,
        method: str = "min_time_diff"
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements different strategies for selecting a single solution from
    the Pareto front based on different optimization priorities.

    Args:
        pareto_front: Array of objective values for each solution
        pareto_solutions: Array of decision variables for each solution
        method: Selection method: 'min_makespan', 'min_time_diff',
                'min_assign_diff', or 'balanced'

    Returns:
        Tuple containing selected solution and its objective values
    """
    if method == "min_makespan":
        best_idx = np.argmin(pareto_front[:, 2])
    elif method == "min_time_diff":
        best_idx = np.argmin(pareto_front[:, 0])
    elif method == "min_assign_diff":
        best_idx = np.argmin(pareto_front[:, 1])
    elif method == "balanced":
        # Normalize objectives and select solution with minimum sum
        normalized_front = pareto_front / np.max(pareto_front, axis=0)
        scores = np.sum(normalized_front, axis=1)
        best_idx = np.argmin(scores)
    else:
        best_idx = 0

    return pareto_solutions[best_idx], pareto_front[best_idx]


def calculate_start_completion_times(jobshop, machine_assignments, current_time):
    """
    Computes S (start times) and C (completion times) values similar to Gurobi model,
    considering machine and job constraints.

    Args:
        jobshop: Jobshop problem instance
        machine_assignments: Dictionary mapping operation IDs to machine IDs
        current_time: Current simulation time

    Returns:
        Tuple containing:
            - S_values: Dictionary of start times for each operation
            - C_values: Dictionary of completion times for each operation
    """
    S_values = {}  # Start times
    C_values = {}  # Completion times

    # Group operations by job and sort by natural order
    job_operations = {}
    for job in jobshop.jobs:
        job_ops = [op for op in job.operations if op.operation_id in machine_assignments]
        job_operations[job.job_id] = sorted(job_ops, key=lambda op: job.operations.index(op))

    # Track machine and job availability
    machine_available = {m.machine_id: current_time for m in jobshop.machines}
    job_available = {j.job_id: current_time for j in jobshop.jobs}

    # Schedule operations job by job, operation by operation
    for job in jobshop.jobs:
        job_ops = job_operations[job.job_id]

        for operation in job_ops:
            op_id = operation.operation_id
            machine_id = machine_assignments[op_id]
            processing_time = operation.processing_times[machine_id]

            # Start time = max(job available, machine available)
            start_time = max(job_available[job.job_id], machine_available[machine_id])
            end_time = start_time + processing_time

            S_values[op_id] = start_time
            C_values[op_id] = end_time

            # Update availability
            job_available[job.job_id] = end_time
            machine_available[machine_id] = end_time

    return S_values, C_values


def calculate_metrics(pareto_front):
    """
    Computes several metrics to evaluate the quality and characteristics
    of the Pareto front:
        number of Pareto solutions
        spacing
        diversity
        MMID

    Args:
        pareto_front: NumPy array of shape (n_solutions, n_objectives)

    Returns:
        Dictionary containing:
            - n_solutions: Number of Pareto solutions
            - spacing: Spacing metric (lower is better)
            - diversity: Diversity metric (higher is better)
            - MMID: Mean Ideal Distance metric
    """
    # Initialize dictionary
    metrics = {}

    # Number of Pareto solutions
    metrics['n_solutions'] = len(pareto_front)

    # Spacing
    d_min = []
    for i in range(len(pareto_front)):
        min_distance=float('inf')
        for k in range(len(pareto_front)):
            if i != k:
                manhattan_dist = np.sum(np.abs(pareto_front[i] - pareto_front[k]))
                if manhattan_dist < min_distance:
                    min_distance = manhattan_dist
        d_min.append(min_distance)

    d_mean = np.mean(d_min)

    spacing_sum = 0.0
    for d_val in d_min:
        spacing_sum += (d_val - d_mean) ** 2

    metrics['spacing'] = np.sqrt(spacing_sum / (len(pareto_front) - 1))

    # Diversity
    n_obj = pareto_front.shape[1]
    f_diff = []
    for i in range(n_obj):
        f_max = np.max(pareto_front[:, i])
        f_min = np.min(pareto_front[:, i])
        f_diff.append((f_max-f_min)**2)

    f_sum = np.sum(f_diff)
    metrics['diversity'] = math.sqrt(f_sum)

    # MMID
    norm_dis = []
    for i in range(len(pareto_front)):
        f_values = []
        for m in range(n_obj):
            f_max = np.max(pareto_front[:, m])
            f_min = np.min(pareto_front[:, m])
            f = pareto_front[i, m]
            if f_max == f_min:
                f_values.append(0)
            else:
                f_values.append(((f-f_min)/(f_max-f_min))**2)
        norm_dis.append(math.sqrt(np.sum(f_values)))
    metrics['MMID'] = np.mean(norm_dis)

    return metrics


def calculate_GD(pareto_front, optimal_pareto_front):
    """
    Computes the generational distance 
    """
    indicator = GD(optimal_pareto_front)
    gd_value = indicator(pareto_front)
    return gd_value


def calculate_hypervolumes(pareto_front, optimal_front):
    ref_point = np.array([1.2, 1.2, 1.2])
    ind = HV(ref_point)
    HV_pf = ind(pareto_front)
    HV_of = ind(optimal_front)
    return HV_pf, HV_of


def compute_distances_full(df):
    """
    Normalize the Pareto front and the "optimal bounds" found from it.
    Then, the distance of each points from the three "optimal points" (one
    for each objective) is computed.
    Args:
        - df: Pareto front dataframe
    Returns:
        - df_out: normalized Pareto front dataframe containing the distaces as well
    """

    # Pareto front points
    original_front = df[
        ["Time Differences", "Assignment Differences", "Makespan"]
    ].values

    # Bound points (3 for each individual)
    optimal_td = np.column_stack([
        df["Optimal Time Differences"],
        df["Assignment Differences"],
        df["Makespan"]
    ])

    optimal_ad = np.column_stack([
        df["Time Differences"],
        df["Optimal Assignment Differences"],
        df["Makespan"]
    ])

    optimal_ms = np.column_stack([
        df["Time Differences"],
        df["Assignment Differences"],
        df["Optimal Makespan"]
    ])

    optimal_front = np.vstack([optimal_td, optimal_ad, optimal_ms])

    # Normalization (taking into consideration both the sets)
    all_fronts = np.vstack([original_front, optimal_front])

    min_vals = np.min(all_fronts, axis=0)
    max_vals = np.max(all_fronts, axis=0)

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    original_norm = (original_front - min_vals) / ranges

    optimal_td_norm = (optimal_td - min_vals) / ranges
    optimal_ad_norm = (optimal_ad - min_vals) / ranges
    optimal_ms_norm = (optimal_ms - min_vals) / ranges

    # 3D Heuclidean distances
    dist_td = np.linalg.norm(original_norm - optimal_td_norm, axis=1)
    dist_ad = np.linalg.norm(original_norm - optimal_ad_norm, axis=1)
    dist_ms = np.linalg.norm(original_norm - optimal_ms_norm, axis=1)

    # Normalized df
    df_out = df.copy()

    df_out[["Time differences",
            "Assignment differences",
            "Makespan"]] = original_norm

    df_out["Optimal Time Differences"] = optimal_td_norm[:, 0]
    df_out["Optimal Assignment Differences"] = optimal_ad_norm[:, 1]
    df_out["Optimal Makespan"] = optimal_ms_norm[:, 2]

    df_out["Distance Time differences"] = dist_td
    df_out["Distance assignment differences"] = dist_ad
    df_out["Distance makespan"] = dist_ms

    return df_out


def compute_distances(pareto_front, optimal_pareto_front):
    D = cdist(pareto_front, optimal_pareto_front)

    # Minimum distance for each point
    distances = np.min(D, axis=1)

    # Index of the closest point of the optimal pareto front
    nearest_indices = np.argmin(D, axis=1)

    return distances, nearest_indices


def update_environment_with_solution(jobshop, solution: np.ndarray, decoder, initializer: RandomInitializer,
                                     current_time: int):
    """
    Applies a solution to the jobshop environment, scheduling both ongoing operations and
    new operations according to the solution's machine assignments and operation sequence.

    Args:
        jobshop: Jobshop problem instance
        solution: Solution array containing MS and OS chromosomes
        decoder: Solution decoder instance
        initializer: Random initializer instance
        current_time: Current simulation time

    Returns:
        Updated jobshop environment
    """
    ongoing_operations_info = []
    for op_id in initializer.operations_ongoing:
        operation = jobshop.get_operation(op_id)
        ongoing_operations_info.append({
            'op_id': op_id,
            'job_id': operation.job_id,
            'machine_id': operation.scheduled_machine,
            'start_time': operation.scheduled_start_time,
            'end_time': operation.scheduled_end_time
        })

    # Decode solution
    ms = solution[:decoder.nr_op]
    os = solution[decoder.nr_op:]

    machine_assignments = decoder._decode_ms(ms)
    operation_sequence = decoder._decode_os(os)

    # Reset environment
    jobshop.reset()

    # Track machine availability
    machine_times = {machine.machine_id: current_time for machine in jobshop.machines}

    # Track job progress - last operation completion time for each job
    job_last_completion = {job.job_id: current_time for job in jobshop.jobs}

    for op_info in ongoing_operations_info:
        op_id = op_info['op_id']
        machine_id = op_info['machine_id']
        start_time = op_info['start_time']
        end_time = op_info['end_time']

        operation = jobshop.get_operation(op_id)
        machine = jobshop.get_machine(machine_id)
        processing_time = end_time - start_time

        machine.add_operation_to_schedule_at_time(
            operation, start_time, processing_time, 0
        )

        machine_times[machine_id] = end_time
        job_last_completion[op_info['job_id']] = end_time

    # Track which operations have been scheduled for each job
    job_scheduled_count = {job.job_id: 0 for job in jobshop.jobs}

    # Get all operations for each job that need to be rescheduled
    job_operations_to_schedule = {}
    for job in jobshop.jobs:
        job_ops = [op for op in job.operations if op.operation_id in decoder.operations_toprocess]
        job_operations_to_schedule[job.job_id] = job_ops

    # Schedule operations in the EXACT order given by OS chromosome
    scheduled_operations = set()

    for op_id in operation_sequence:
        if op_id in scheduled_operations:
            continue

        operation = jobshop.get_operation(op_id)
        job_id = operation.job_id
        machine_id = machine_assignments[op_id]

        # Check if this is the next operation for its job
        job_ops = job_operations_to_schedule[job_id]
        current_index = job_scheduled_count[job_id]

        if current_index >= len(job_ops):
            continue

        next_op_for_job = job_ops[current_index]

        if op_id != next_op_for_job.operation_id:
            # Skip and try later
            continue

        # This operation is ready, schedule it
        processing_time = operation.processing_times[machine_id]

        # Start time = max(machine available, previous operation in job finished)
        start_time = max(job_last_completion[job_id], machine_times[machine_id])
        end_time = start_time + processing_time

        # Schedule the operation
        machine = jobshop.get_machine(machine_id)
        machine.add_operation_to_schedule_at_time(
            operation, start_time, processing_time, 0
        )

        # Update tracking
        machine_times[machine_id] = end_time
        job_last_completion[job_id] = end_time
        job_scheduled_count[job_id] += 1
        scheduled_operations.add(op_id)

    return jobshop


def print_pareto_statistics(pareto_front: np.ndarray):
    """
    Displays summary statistics including number of solutions and range of objective values.

    Args:
        pareto_front: NumPy array of shape (n_solutions, 3) containing objective values
    """
    print("\n Pareto Front Statistics:")
    print(f"  Number of solutions: {len(pareto_front)}")
    print(f"  Time Differences: {pareto_front[:, 0].min():.1f} - {pareto_front[:, 0].max():.1f}")
    print(f"  Assignment Differences: {pareto_front[:, 1].min():.1f} - {pareto_front[:, 1].max():.1f}")
    print(f"  Makespan: {pareto_front[:, 2].min():.1f} - {pareto_front[:, 2].max():.1f}")

    print(f"\n  Best in each objective:")
    print(f"    Min Time Differences: {pareto_front[:, 0].min():.1f}")
    print(f"    Min Assignment Differences: {pareto_front[:, 1].min():.1f}")
    print(f"    Min Makespan: {pareto_front[:, 2].min():.1f}")


def save_results(res, decoder, output_dir: str = "results"):
    """
    Saves Pareto solutions in both CSV and JSON formats, along with calculated metrics and
    visualizations.

    Args:
        res: Optimization result object
        decoder: Solution decoder instance
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pareto_front = res.F  # Objective matrix
    objective_names = ["Time Differences", "Assignment Differences", "Makespan"]

    # Combined CSV: MS, OS, objectives
    csv_path = output_dir / "pareto_solutions_full.csv"

    # Open file for writing
    with open(csv_path, "w") as f:
        # Header
        f.write("MS,OS," + ",".join(objective_names) + "\n")

        # Write rows
        for solution, objectives in zip(res.X, pareto_front):
            ms = solution[:decoder.nr_op].tolist()
            os_seq = solution[decoder.nr_op:].tolist()

            # MS and OS as string without spaces, separated by -
            ms_str = "-".join(map(str, ms))
            os_str = "-".join(map(str, os_seq))

            # Objective values
            obj_str = ",".join(f"{v:.5f}" for v in objectives)

            f.write(f"{ms_str},{os_str},{obj_str}\n")

    print(f"Combined Pareto solutions and objectives saved to CSV: {csv_path}")

    solutions_list = []
    for solution in res.X:
        ms = solution[:decoder.nr_op].tolist()
        os_seq = solution[decoder.nr_op:].tolist()
        solutions_list.append({"MS": ms, "OS": os_seq})

    json_path = os.path.join(output_dir, "pareto_solutions.json")
    with open(json_path, "w") as f:
        f.write("[\n")
        for i, sol in enumerate(solutions_list):
            line = f'  {json.dumps(sol, separators=(",", ":"))}'
            if i < len(solutions_list) - 1:
                line += ","
            line += "\n"
            f.write(line)
        f.write("]\n")

    print(f"Pareto solutions saved to JSON: {json_path}")

    """# Calculate and save metrics
    metrics = calculate_metrics(res.F)
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved to: {metrics_path}")"""


def save_normalized_pareto_distance_log(
    file_path,
    normalized_pareto_front: np.ndarray,
    normalized_optimal_front: np.ndarray,
    distances: np.ndarray,
    nearest_indices: np.ndarray = None
):
    """
    Saves a log with:
    - Pareto front points normalized
    - Pareto optimal point normalized closest
    - Heuclidean distance

    Args:
        - file_path: Path to the log file
        - normalized_pareto_front: Normalized pareto front points
        - normalized_optimal_front: Normalized optimal front points
        - distances: Distances between pareto front points
        - nearest_indices: Nearest indices
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "point_id",
            "pf_f1_norm", "pf_f2_norm", "pf_f3_norm",
            "nearest_opt_id",
            "opt_f1_norm", "opt_f2_norm", "opt_f3_norm",
            "euclidean_distance"
        ])

        for i in range(len(normalized_pareto_front)):
            if nearest_indices is not None:
                opt_idx = nearest_indices[i]
                opt_point = normalized_optimal_front[opt_idx]
            else:
                opt_idx = ""
                opt_point = ["", "", ""]

            writer.writerow([
                i,
                *normalized_pareto_front[i],
                opt_idx,
                *opt_point,
                distances[i]
            ])


def save_optimal_pareto_front_log(
    file_path,
    optimal_pareto_front: np.ndarray
):
    """
    Saves the optimal pareto front in a csv file.
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "point_id",
            "obj1_time_diff",
            "obj2_assign_diff",
            "obj3_makespan"
        ])

        for i, point in enumerate(optimal_pareto_front):
            writer.writerow([i, *point])


def debug_pareto_front(res, pareto_front):
    """
    Debug function to understand why Pareto front always has 100 solutions
    """
    print("=== PARETO FRONT DEBUG ===")
    print(f"Total population size: {len(res.X)}")
    print(f"Pareto front size: {len(pareto_front)}")

    # Check if all solutions are really non-dominated
    nds = NonDominatedSorting()
    fronts = nds.do(res.F)

    print(f"Number of fronts: {len(fronts)}")
    print(f"Front 0 size: {len(fronts[0])}")
    print(f"Front 1 size: {len(fronts[1]) if len(fronts) > 1 else 0}")

    # Check objective ranges
    print(f"Objective ranges:")
    for i in range(res.F.shape[1]):
        print(f"  Obj {i}: min={res.F[:, i].min():.3f}, max={res.F[:, i].max():.3f}")

    # Check if all solutions in first front are truly non-dominated
    if len(fronts[0]) == len(res.F):
        print("WARNING: All solutions are in the first front!")
        print("   This suggests a problem with dominance relations")

    return fronts


def visualize_all_pareto_fronts(pareto_front: np.ndarray, output_dir: str = "results"):
    """
    Creates and saves multiple visualization types for the Pareto front
        including 2D plots and 3D interactive plots.

    Args:
        pareto_front: NumPy array of shape (n_solutions, 3) containing objective values
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating Pareto front visualizations...")

    # 2D plot
    plot_pareto_front_2d(pareto_front, save_path=f"{output_dir}/pareto_2d.png")

    # 3D interactive plot
    plot_pareto_front_3d_matplotlib(pareto_front, save_path=f"{output_dir}/pareto_3d.html")

    print("All visualizations generated!")


def analyze_possible_combinations(decoder, operators, initializer, n_samples=1000):
    """
    Analyze how many combinations of solutions are possible.
    """
    print("\n" + "=" * 60)
    print("POSSIBLE COMBINATIONS ANALYSIS")
    print("=" * 60)

    unique_combinations = set()
    all_objectives = []

    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Testing {i}/{n_samples}...")

        sol = initializer.create_individual()
        obj = decoder.decode(sol)

        # Round to identify "identical" combinations
        rounded_obj = tuple(round(x, 2) for x in obj)
        unique_combinations.add(rounded_obj)
        all_objectives.append(obj)

    all_objectives = np.array(all_objectives)

    print(f"\nResults after {n_samples} random solutions:")
    print(f"Unique objective combinations: {len(unique_combinations)}")
    print(f"Time Differences unique values: {len(np.unique(all_objectives[:, 0]))}")
    print(f"Assignment Differences unique values: {len(np.unique(all_objectives[:, 1]))}")
    print(f"Makespan unique values: {len(np.unique(all_objectives[:, 2]))}")

    # Show the found combinations
    print(f"\nSample of unique combinations found:")
    for i, combo in enumerate(list(unique_combinations)[:10]):
        print(f"  {i}: Time={combo[0]:.1f}, Assign={combo[1]:.1f}, Makespan={combo[2]:.1f}")

    return len(unique_combinations)


def find_max_ms_needed(operators):
    """
    Find the maximum number of valid machines for any operation.
    This determines the superior bound for the MS part of the chromosome.

    Args:
        operators: Instance of GeneticOperators

    Returns:
        int: Max MS necessary value
    """
    max_needed = 0
    for job in operators.env.jobs:
        for operation in job.operations:
            if operation.operation_id in operators.operations_toprocess:
                valid_machines = [m for m in operation.optional_machines_id
                                  if m != operators.broken_machine]
                max_needed = max(max_needed, len(valid_machines))

    # -1 because indexes start from 0
    return max_needed - 1 if max_needed > 0 else 0
