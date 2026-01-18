import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Resolve directory structure
PROJECT_ROOT = "/home/bazzo/Scrivania/fjsp_rescheduling"
SIMULATIONS_DIR = os.path.join(PROJECT_ROOT, "simulations")
NSGA2_DIR = os.path.join(PROJECT_ROOT, "NSGA2")
SRC_DIR = os.path.join(NSGA2_DIR, "src")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_SCH_DIR = os.path.join(PROJECT_ROOT, "results_scheduling")
SAVE_LOGS = os.path.join(SIMULATIONS_DIR, "logs", "NSGA2")

# Extend Python path to allow cross-module imports
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SIMULATIONS_DIR)
sys.path.insert(0, NSGA2_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SAVE_DIR)
sys.path.insert(0, RESULTS_SCH_DIR)


from simulations.experiment_logging import (
    setup_logging_directory, setup_experiment_logging, close_experiment_logging, log_experiment_start,
    log_experiment_end, load_environment, log_environment_info
)

from MILP.model_MILP_reschedule_changed import FJS_reschedule

from NSGA2.nsga_ii import run_nsga2_with_my_operators, MySampling
from NSGA2.utils import (plot_pareto_front_3d_matplotlib, select_best_solution,
    update_environment_with_solution, save_results
    )
from NSGA2.src.random_initialization import RandomInitializer
from NSGA2.src.decode import Decoder
from NSGA2.src.schedule_manager import ScheduleManager
from scheduling_environment.jobShop import JobShop
from visualization.gantt_chart import plot


def plot_comparison_3d(nsga2_points, milp_points, distances, save_path=None):
    """
    Plots the NSGA2 points and MILP in 3D with the distances between couples.

    Args:
        nsga2_points: NSGA2 points (numpy array (n,3))
        milp_points: MILP points (numpy array (n,3))
        distances: distances between the couples (n, )
        save_path: optional path to save the graph (str, optional)

    Returns:
        plt
    """
    fig = plt.figure(figsize=(15,10))

    ax = fig.add_subplot(121, projection='3d')

    ax.scatter(nsga2_points[:, 0], nsga2_points[:,1], nsga2_points[:,2],
                c='blue', s=100, marker='o', label='NSGA2', alpha=0.8, edgecolors='black')

    ax.scatter(milp_points[:, 0], milp_points[:,1], milp_points[:,2],
                c='red', s=100, marker='s', label='MILP', alpha=0.8, edgecolors='black')

    for i in range(len(nsga2_points)):
        x_line = [nsga2_points[i, 0], milp_points[i, 0]]
        y_line = [nsga2_points[i, 1], milp_points[i, 1]]
        z_line = [nsga2_points[i, 2], milp_points[i, 2]]

        # Color based on the distance (normalized between 0 and 1)
        if np.max(distances) > 0:
            norm_dist = distances[i] / np.max(distances)
        else:
            norm_dist = 0

        # Use colormap (viridi) for distances
        color = plt.cm.viridis(norm_dist)

        # Draws the line
        ax.plot(x_line, y_line, z_line,
                 color=color, linewidth=2, alpha=0.6)

    # Configure the axes
    ax.set_xlabel('Time Differences\n(Quadratic Delay)', fontsize=11, labelpad=10)
    ax.set_ylabel('Assignment Changes', fontsize=11, labelpad=10)
    ax.set_zlabel('Makespan (Cmax)', fontsize=11, labelpad=10)
    ax.set_title('NSGA2 vs MILP Solutions\nwith Pairwise Distances', fontsize=14, pad=20)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(distances), vmax=max(distances)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Distance between NSGA2 and MILP', fontsize=10)

    if np.max(distances) > 0:
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=np.min(distances),
                                                      vmax=np.max(distances)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label('Distance between NSGA2 and MILP', fontsize=10)

    ax.view_init(elev=20, azim=45)

    # Add the information to the graph
    info_text = f'Number of points: {len(nsga2_points)}'
    info_text += f'\nAverage distance: {np.mean(distances):.2f}'
    info_text += f'\nMin distance: {np.min(distances):.2f}'
    info_text += f'\nMax distance: {np.max(distances):.2f}'

    plt.figtext(0.02, 0.02, info_text, fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save the graph if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato in: {save_path}")

    return fig, ax


if __name__ == "__main__":

    # Example NSGA 2 + MILP model to find the optimum Pareto Front

    file_path = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl")
    jobshop = load_environment(file_path)

    broken_machine_id = 0
    disruption_time = 1

    res, decoder, operators = run_nsga2_with_my_operators(
        jobshop=jobshop,
        broken_machine=broken_machine_id,
        current_time=disruption_time,
        pop_size=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        termination_method="improvement"
    )

    # Get results
    paretofront_objectives = res.F
    paretofront_solutions = res.X

    initializer = RandomInitializer(jobshop, disruption_time, broken_machine_id)

    # Select best solution
    best_solution, best_objectives = select_best_solution(
        paretofront_objectives, paretofront_solutions, method="balanced"
    )

    results_dir = os.path.join(SAVE_DIR, "NSGA2")

    pareto_path = os.path.join(results_dir, f"pareto_front_ex_mixed.pdf")
    plot_pareto_front_3d_matplotlib(paretofront_objectives, save_path=pareto_path)

    nsga2_points = []
    milp_points = []
    distances = []

    i=0

    # Iteration on every individual that is part of the found Pareto Front
    for individual in paretofront_solutions:
        # NSGA2 point
        time_differences, assignment_differences, makespan = decoder.decode(individual)
        nsga2_points.append([time_differences, assignment_differences, makespan])

        # Extracts variables for MILP
        X, Y, S, C, E, Z, Z_aux = decoder.extract_sets(individual)

        # Solve with MILP
        reschedule = FJS_reschedule(jobshop, broken_machine_id, disruption_time)
        reschedule.extract_info()
        reschedule.create_model()
        reschedule.run_model(600, X, S, C, Y, E, Z, Z_aux)
        reschedule.print_results()

        # MILP point
        time_differences1 = reschedule.results['quadratic_delay']
        assignment_differences1 = reschedule.results['assignment_changes']
        makespan1 = reschedule.results['cmax']
        milp_points.append([time_differences1, assignment_differences1, makespan1])

        # Calculate the heuclidean distance between the two points
        dist = np.sqrt(
            (time_differences - time_differences1)**2 +
            (assignment_differences - assignment_differences1)**2+
            (makespan-makespan1)**2
        )
        distances.append(dist)
        print(f"Point {i + 1}:")
        print(f"  NSGA2: ({time_differences:.2f}, {assignment_differences:.2f}, {makespan:.2f})")
        print(f"  MILP:  ({time_differences1:.2f}, {assignment_differences1:.2f}, {makespan1:.2f})")
        print(f"  Heuclidean distance: {dist:.2f}")
        print("-" * 40)

    nsga2_array = np.array(nsga2_points)
    milp_array = np.array(milp_points)
    distances_array = np.array(distances)

    fig, ax = plot_comparison_3d(
        nsga2_array,
        milp_points,
        distances_array,
        save_path=os.path.join(results_dir, "pareto_front.pdf")
    )
    plt.show()



