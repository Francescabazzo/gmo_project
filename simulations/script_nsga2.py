# Import libraries
import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

SIMULATIONS_DIR = PROJECT_ROOT / "simulations"
NSGA2_DIR = PROJECT_ROOT / "NSGA2"
SRC_DIR = NSGA2_DIR / "src"
SAVE_DIR = PROJECT_ROOT / "results"
RESULTS_SCH_DIR = PROJECT_ROOT / "results_scheduling"
SAVE_LOGS = SIMULATIONS_DIR / "logs" / "NSGA2"
MILP_DIR = PROJECT_ROOT / "MILP"

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"SIMULATIONS_DIR: {SIMULATIONS_DIR}")
print(f"NSGA2_DIR: {NSGA2_DIR}")
print(f"SRC_DIR: {SRC_DIR}")
print(f"MILP_DIR: {MILP_DIR}")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SIMULATIONS_DIR))
sys.path.insert(0, str(NSGA2_DIR))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MILP_DIR))

from NSGA2.utils import (calculate_GD, compute_distances_full, calculate_hypervolumes
                         )
from NSGA2.src.decode import Decoder
from NSGA2.src.random_initialization import RandomInitializer
from NSGA2.src.schedule_manager import ScheduleManager
from simulations.experiment_logging import load_environment
from simulations.run_experiments_mo_nolin import get_data_break


def run_experiment(exp_num: int, n_pop: int, file_schedule_path):
    """
    Computes the following
    """
    print("                                                                                 ")
    print(f"================ EXPERIMENT {exp_num}, POPULATION {n_pop} ================")
    experiment_dir = os.path.join(SAVE_DIR, "NSGA2", f"EX{exp_num}", f"Pop_{n_pop}")
    optimal_points_found = pd.read_csv(os.path.join(experiment_dir, "optimal_points_found.csv"))

    jobshop = load_environment(file_schedule_path)
    data_break = get_data_break()
    broken_machine_id = data_break[f"EX{exp_num}"]["broken_machine_id"]
    disruption_time = data_break[f"EX{exp_num}"]["disruption_time"]

    initializer = RandomInitializer(jobshop, disruption_time, broken_machine_id)
    schedule_manager= ScheduleManager(jobshop)
    decoder = Decoder(jobshop, broken_machine_id, disruption_time, initializer, schedule_manager)

    normalized_distances = compute_distances_full(optimal_points_found)

    columns = [
        "Time differences",
        "Assignment differences",
        "Makespan",
        "Optimal Time Differences",
        "Distance Time differences",
        "Optimal Assignment Differences",
        "Distance assignment differences",
        "Optimal Makespan",
        "Distance Makespan"
    ]

    normalized_distances[columns].to_csv(
        os.path.join(experiment_dir, "optimal_points_found_distances.csv"),
        index=False
    )

    pareto_front_full = pd.read_csv(os.path.join(experiment_dir, "pareto_solutions_full.csv"))
    pareto_front_full = pareto_front_full.drop(['MS', 'OS'], axis=1)

    optimal_filtered_front = pd.read_csv(os.path.join(experiment_dir, "optimal_pareto_front.csv"))
    optimal_filtered_front = optimal_filtered_front.drop(['point_id'], axis=1)

    print(f"Pareto front shape: {pareto_front_full.shape}")
    print(f"Optimal Pareto front shape: {optimal_filtered_front.shape}")

    normalized_pareto_front, normalized_filtered_front, stats_normalization = decoder.normalize_pareto_front(
        pareto_front_full,
        optimal_filtered_front
    )

    print("normalized_pareto_front type:", type(normalized_pareto_front))
    print("normalized_pareto_front shape:", np.asarray(normalized_pareto_front).shape)

    print("normalized_filtered_front type:", type(normalized_filtered_front))
    print("normalized_filtered_front shape:", np.asarray(normalized_filtered_front).shape)

    normalized_pareto_front = normalized_pareto_front.to_numpy()
    normalized_filtered_front = normalized_filtered_front.to_numpy()

    GD = calculate_GD(normalized_pareto_front, normalized_filtered_front)

    HV, HV_opt = calculate_hypervolumes(normalized_pareto_front, normalized_filtered_front)

    metrics = {
        "GD": float(GD),
        "Hypervolume": float(HV),
        "Hypervolume_opt": float(HV_opt)
    }

    metrics_path = os.path.join(experiment_dir, "metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_path}")


def get_resch_obj(exp_num: int, n_pop: int):
    """

    """
    experiment_dir = os.path.join(SAVE_DIR, "NSGA2", f"EX{exp_num}", f"Pop_{n_pop}")
    file_path = os.path.join(experiment_dir, f"rescheduled_environment_ex{exp_num}.pkl")
    old_envir = load_environment(os.path.join(RESULTS_SCH_DIR, f"schedule_simulation_{exp_num}.pkl"))
    resch_envir = load_environment(file_path)
    obj_time_delays = 0
    for job in resch_envir.jobs:
        job_id = job.job_id
        old_job = old_envir.get_job(job_id)
        obj_time_delays += (job.scheduled_end_time-old_job.scheduled_end_time)**2
    obj_mach_assignments = 0
    for operation in resch_envir.operations:
        new_machine = operation.scheduled_machine
        operation_id = operation.operation_id
        old_operation = old_envir.get_operation(operation_id)
        old_machine = old_operation.scheduled_machine
        if new_machine != old_machine:
            obj_mach_assignments += 1
    obj_cmax = resch_envir.makespan
    objectives = {
        "Total Squared Completion Time Delays": obj_time_delays,
        "Total Machine Assignment Changes": obj_mach_assignments,
        "Makespan": obj_cmax
    }
    obj_path = os.path.join(experiment_dir, "objectives_resch_env.json")
    with open(obj_path, "w") as f:
        json.dump(objectives, f, indent=4)


if __name__ == "__main__":
    # Experiment 2
    """run_experiment(2, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))
    run_experiment(2, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))
    run_experiment(2, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))
    run_experiment(2, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))

    # Experiment 3
    run_experiment(3, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))
    run_experiment(3, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))
    run_experiment(3, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))
    run_experiment(3, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))

    # Experiment 4
    run_experiment(4, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))
    run_experiment(4, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))
    run_experiment(4, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))
    run_experiment(4, 200)


    """
    # Experiment 5
    run_experiment(5, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    run_experiment(5, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    run_experiment(5, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    run_experiment(5, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))

    # Experiment 6
    run_experiment(6, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))
    run_experiment(6, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))
    run_experiment(6, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))
    run_experiment(6, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))

    # Experiment 7
    run_experiment(7, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))
    run_experiment(7, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))
    run_experiment(7, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))
    run_experiment(7, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))

    """

    # Experiment 8
    run_experiment(8, 20)
    run_experiment(8, 50)
    run_experiment(8, 100)
    run_experiment(8, 200)

    # Experiment 9
    run_experiment(9, 20)
    run_experiment(9, 50)
    run_experiment(9, 100)
    run_experiment(9, 200)

    # Experiment 10
    run_experiment(10, 20)
    run_experiment(10, 50)
    run_experiment(10, 100)
    run_experiment(10, 200)

    # Experiment 11
    run_experiment(11, 20)
    run_experiment(11, 50)
    run_experiment(11, 100)
    run_experiment(11, 200)

    # Experiment 12
    run_experiment(12, 20)
    run_experiment(12, 50)
    run_experiment(12, 100)
    run_experiment(12, 200)

    # Experiment 13
    run_experiment(13, 20)
    run_experiment(13, 50)
    run_experiment(13, 100)
    run_experiment(13, 200)

    # Experiment 14
    run_experiment(14, 20)
    run_experiment(14, 50)
    run_experiment(14, 100)
    run_experiment(14, 200)

    # Experiment 15
    run_experiment(15, 20)
    run_experiment(15, 50)
    run_experiment(15, 100)
    run_experiment(15, 200)

    # Experiment 16
    run_experiment(16, 20)
    run_experiment(16, 50)
    run_experiment(16, 100)
    run_experiment(16, 200)

    # Experiment 17
    run_experiment(17, 20)
    run_experiment(17, 50)
    run_experiment(17, 100)
    run_experiment(17, 200)

    # Experiment 18
    run_experiment(18, 20)
    run_experiment(18, 50)
    run_experiment(18, 100)
    run_experiment(18, 200)

    # Experiment 19
    run_experiment(19, 20)
    run_experiment(19, 50)
    run_experiment(19, 100)
    run_experiment(19, 200)

    # Experiment 20
    run_experiment(20, 20)
    run_experiment(20, 50)
    run_experiment(20, 100)
    run_experiment(20, 200)

    # Experiment 21
    run_experiment(21, 20)
    run_experiment(21, 50)
    run_experiment(21, 100)
    run_experiment(21, 200)

    # Experiment 22
    run_experiment(22, 20)
    run_experiment(22, 50)
    run_experiment(22, 100)
    run_experiment(22, 200)

    # Experiment 23
    run_experiment(23, 20)
    run_experiment(23, 50)
    run_experiment(23, 100)
    run_experiment(23, 200)

    # Experiment 24
    run_experiment(24, 20)
    run_experiment(24, 50)
    run_experiment(24, 100)
    run_experiment(24, 200)

    # Experiment 25
    run_experiment(25, 20)
    run_experiment(25, 50)
    run_experiment(25, 100)
    run_experiment(25, 200)

    # Experiment 26
    run_experiment(26, 20)
    run_experiment(26, 50)
    run_experiment(26, 100)
    run_experiment(26, 200)

    # Experiment 28
    run_experiment(28, 20)
    run_experiment(28, 50)
    run_experiment(28, 100)
    run_experiment(28, 200)

    # Experiment 30
    run_experiment(30, 20)
    run_experiment(30, 50)
    run_experiment(30, 100)
    run_experiment(30, 200)"""