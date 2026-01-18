import sys
import os
import datetime
import io
import traceback
import hashlib
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
matplotlib.use('Agg')   # backend non interattivo
matplotlib.rcParams['interactive'] = False  # Disabilita modalità interattiva

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


from simulations.experiment_logging import (
    setup_experiment_logging, close_experiment_logging, log_experiment_start,
    log_experiment_end, load_environment, log_environment_info
)

from NSGA2.nsga_ii import run_nsga2_with_my_operators, MySampling, assess_pareto_front
from NSGA2.utils import (plot_pareto_front_3d_matplotlib, select_best_solution,
                         update_environment_with_solution, save_results,
                         compute_distances, compute_distances_full, save_optimal_pareto_front_log,
                         save_normalized_pareto_distance_log, calculate_GD, calculate_hypervolumes
                         )
from NSGA2.src.random_initialization import RandomInitializer
from NSGA2.src.decode import Decoder
from NSGA2.src.schedule_manager import ScheduleManager
from scheduling_environment.jobShop import JobShop
from visualization.gantt_chart import plot
from simulations.run_experiments_mo_nolin import get_data_break


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def log_nsga2_results(logger, res, best_objectives, selection_method):
    """Log NSGA-II specific results"""
    logger.info("NSGA-II Optimization Results:")
    logger.info(f"  Number of generations completed: {res.algorithm.n_gen}")
    logger.info(f"  Number of function evaluations: {res.algorithm.evaluator.n_eval}")
    logger.info(f"  Pareto front size: {len(res.F)}")

    logger.info(f"Best solution selected ({selection_method}):")
    logger.info(f"  Time Differences: {best_objectives[0]:.2f}")
    logger.info(f"  Assignment Differences: {best_objectives[1]:.2f}")
    logger.info(f"  Makespan: {best_objectives[2]:.2f}")

    # Log Pareto front statistics
    if len(res.F) > 0:
        logger.info("Pareto Front Statistics:")
        logger.info(f"  Time Differences range: {res.F[:, 0].min():.2f} - {res.F[:, 0].max():.2f}")
        logger.info(f"  Assignment Differences range: {res.F[:, 1].min():.2f} - {res.F[:, 1].max():.2f}")
        logger.info(f"  Makespan range: {res.F[:, 2].min():.2f} - {res.F[:, 2].max():.2f}")


def run_nsga2_experiment(exp_num: int, n_pop: int, file_path, n_gen: int):
    """
    Run NSGA-II experiment, assess the Pareto front and save all results in the specified directories


    """
    seed_str = f"{exp_num}_{n_pop}_{n_gen}"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2 ** 32)

    # Directory dove vogliamo salvare i log
    actual_log_dir = SAVE_LOGS / f"EX{exp_num}" / f"Pop_{n_pop}" / "test1"
    os.makedirs(actual_log_dir, exist_ok=True)

    # DEBUG: Stampa i percorsi
    print(f"Actual log directory: {actual_log_dir}")

    # CHIAMATA MODIFICATA: Passiamo SAVE_LOGS come base_dir, ma specifichiamo log_filename con percorso completo
    logger = setup_experiment_logging(
        log_dir=SAVE_LOGS,  # Directory base (come si aspetta la funzione)
        exp_num=exp_num,
        log_filename=str(actual_log_dir / f"experiment_{exp_num}.txt")  # Percorso completo del file
    )

    terminal_output_file = actual_log_dir / f"terminal_output_ex{exp_num}.txt"

    start_time = datetime.datetime.now()

    try:

        # Create experiment directories
        results_dir = os.path.join(SAVE_DIR, "NSGA2", f"EX{exp_num}", f"Pop_{n_pop}", f"test1")
        os.makedirs(results_dir, exist_ok=True)

        with open(terminal_output_file, 'w', encoding='utf-8') as term_file:

            tee_out = Tee(sys.__stdout__, term_file)
            tee_err = Tee(sys.__stderr__, term_file)

            with redirect_stdout(tee_out), redirect_stderr(tee_err):

                term_file.write(f"=== EXPERIMENT {exp_num} STARTED AT {start_time} ===\n")
                term_file.write(f"Population size: {n_pop}\n")
                term_file.write(f"Data file: {file_path}\n")
                term_file.write("-" * 80 + "\n\n")
                term_file.flush()

                try:
                    # Load environment
                    jobShopEnv = load_environment(file_path, logger)
                    data_break = get_data_break()

                    broken_machine_id = data_break[f"EX{exp_num}"]["broken_machine_id"]
                    disruption_time = data_break[f"EX{exp_num}"]["disruption_time"]

                    print("Jobs:", jobShopEnv.nr_of_jobs)
                    print("Machines:", jobShopEnv.nr_of_machines)
                    print("Ops per job:", [len(j.operations) for j in jobShopEnv.jobs])

                    params = {
                        'file_path': file_path,
                        'broken_machine_id': broken_machine_id,
                        'disruption_time': disruption_time
                    }

                    log_experiment_start(logger, exp_num, params)

                    log_environment_info(logger, jobShopEnv)

                    logger.info("Starting rescheduling with NSGA-II...")

                    start_rescheduling = datetime.datetime.now()

                    # Run NSGA-II
                    res, decoder, operators = run_nsga2_with_my_operators(
                        jobshop=jobShopEnv,
                        broken_machine=broken_machine_id,
                        current_time=disruption_time,
                        seed=seed,
                        pop_size=n_pop,
                        crossover_rate=0.8,
                        mutation_rate=0.1,
                        termination_method="n_gen"
                    )

                    end_rescheduling = datetime.datetime.now()

                    duration_rescheduling = end_rescheduling - start_rescheduling

                    # Get results
                    pareto_front = res.F
                    pareto_solutions = res.X

                    save_results(res, decoder, results_dir)

                    # Find lb for the first objective
                    dir_milp = os.path.join(results_dir, "MILP", f"EX{exp_num}", "results.json")

                    with open(dir_milp, 'r') as f:
                        dict = json.load(f)
                        obj1_lb = dict['quadr_delay']

                    # Pareto front assessment
                    optimal_front, optimal_filtered_front, optimal_points_found = assess_pareto_front(
                        pareto_solutions,
                        pareto_front,
                        jobShopEnv,
                        broken_machine_id,
                        disruption_time,
                        obj1_lb=obj1_lb
                    )

                    columns_optimal_points_found = [
                        "Time Differences",
                        "Assignment Differences",
                        "Makespan",
                        "Optimal Time Differences",
                        "Optimal Assignment Differences",
                        "Optimal Makespan",
                    ]

                    optimal_points_found_df = pd.DataFrame(optimal_points_found)

                    optimal_points_found_df[columns_optimal_points_found].to_csv(
                        os.path.join(results_dir, "optimal_points_found.csv"),
                        index=False
                    )

                    # Normalization (only filtered front)
                    normalized_pareto_front, normalized_optimal_front, stats_normalized = decoder.normalize_pareto_front(
                        pareto_front,
                        optimal_filtered_front
                    )

                    distances, nearest_indices = compute_distances(
                        normalized_pareto_front,
                        normalized_optimal_front
                    )

                    normalized_distances = compute_distances_full(optimal_points_found_df)

                    columns = [
                        "Time differences",
                        "Assignment differences",
                        "Makespan",
                        "Optimal Time Differences",
                        "Distance Time differences",
                        "Optimal Assignment Differences",
                        "Distance assignment differences",
                        "Optimal Makespan",
                        "Distance makespan"
                    ]

                    normalized_distances[columns].to_csv(
                        os.path.join(results_dir, "optimal_points_found_distances.csv"),
                        index=False
                    )

                    GD = calculate_GD(normalized_pareto_front, normalized_optimal_front)

                    HV, HV_opt = calculate_hypervolumes(normalized_pareto_front, normalized_optimal_front)

                    metrics = {
                        "GD": float(GD),
                        "Hypervolume": float(HV),
                        "Hypervolume_opt": float(HV_opt),
                        "Duration_seconds": duration_rescheduling.total_seconds(),
                        "Duration": str(duration_rescheduling)
                    }

                    metrics_path = os.path.join(results_dir, "metrics.json")

                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=4)

                    print(f"Metrics saved to {metrics_path}")

                    # Logs
                    save_normalized_pareto_distance_log(
                        os.path.join(results_dir, "normalized_pareto_distance.csv"),
                        normalized_pareto_front,
                        normalized_optimal_front,
                        distances,
                        nearest_indices
                    )

                    save_optimal_pareto_front_log(
                        os.path.join(results_dir, "optimal_pareto_front.csv"),
                        optimal_filtered_front
                    )

                    initializer = RandomInitializer(
                        jobShopEnv,
                        disruption_time,
                        broken_machine_id
                    )

                    # Select best solution
                    best_solution, best_objectives = select_best_solution(
                        pareto_front, pareto_solutions
                    )

                    # Update environment
                    updated_jobshop = update_environment_with_solution(
                        jobShopEnv, best_solution, decoder, initializer, disruption_time
                    )

                    save_results(res, decoder, output_dir=str(results_dir))

                    # Log NSGA-II specific results
                    log_nsga2_results(logger, res, best_objectives, "min_time_diff")

                    # Generate and save plots
                    logger.info("Generating and saving plots...")

                    # PLOTS

                    # 1. Pareto front plot
                    # plt.figure(figsize=(10, 6))
                    pareto_path = os.path.join(results_dir, f"pareto_front.pdf")
                    plot_pareto_front_3d_matplotlib(pareto_front, save_path=pareto_path)
                    plt.title(f'Pareto Front - NSGA-II Experiment {exp_num}')
                    plt.close()
                    logger.info(f"Pareto front plot saved: {pareto_path}")

                    pareto_path = os.path.join(results_dir, f"pareto_front_assessment.pdf")
                    plot_pareto_front_3d_matplotlib(pareto_front, assess=True, optimal_pareto_front=optimal_filtered_front,
                                                    save_path=pareto_path)
                    plt.title(f'Pareto Front Assessment - NSGA-II Experiment {exp_num}')
                    plt.close()
                    logger.info(f"Pareto front plot saved: {pareto_path}")

                    pareto_path = os.path.join(results_dir, f"pareto_front_normalized.pdf")
                    plot_pareto_front_3d_matplotlib(
                        normalized_pareto_front,
                        assess=True,
                        optimal_pareto_front=normalized_optimal_front,
                        save_path=pareto_path
                    )
                    plt.title(f'Pareto Front Normalized - NSGA-II Experiment {exp_num}')
                    plt.close()
                    logger.info(f"Pareto front plot saved: {pareto_path}")

                    # 2. Gantt chart
                    # plt.figure(figsize=(14, 8))
                    plot(updated_jobshop)

                    plt.title(f'Rescheduled Gantt Chart - NSGA-II Experiment {exp_num}',
                              fontsize=14, fontweight='bold')
                    gantt_path = os.path.join(results_dir, f"gantt_chart_ex{exp_num}.png")
                    plt.savefig(gantt_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Gantt chart saved: {gantt_path}")

                    # Save updated environment
                    updated_env_path = os.path.join(results_dir, f"rescheduled_environment_ex{exp_num}.pkl")
                    with open(updated_env_path, 'wb') as f:
                        import pickle
                        pickle.dump(updated_jobshop, f)
                    logger.info(f"Updated environment saved: {updated_env_path}")

                    # Log experiment end
                    log_experiment_end(logger, exp_num, success=True)

                    logger.info(f"All results saved for experiment {exp_num}")
                    logger.info(f"  Plots: {pareto_path}")
                    logger.info(f"  Data: {results_dir}")
                    logger.info(f"  Logs: {actual_log_dir}")

                    return {
                        'res': res,
                        'decoder': decoder,
                        'operators': operators,
                        'updated_jobshop': updated_jobshop,
                        'best_solution': best_solution,
                        'best_objectives': best_objectives
                    }

                except Exception as inner_e:
                    traceback.print_exc()
                    raise inner_e  # Rilancia l'eccezione
                    logger.error(f"Error during NSGA-II experiment {exp_num}: {str(e)}")
                    log_experiment_end(logger, exp_num, success=False, message=str(e))
                    raise

    except Exception as e:
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        with open(terminal_output_file, 'a', encoding='utf-8') as term_file:
            term_file.write(f"\n=== EXPERIMENT {exp_num} FAILED ===\n")
            term_file.write(f"Error: {str(e)}\n")
            term_file.write(f"End time: {end_time}\n")
            term_file.write(f"Duration: {duration}\n")
            term_file.write("=" * 80 + "\n")

        logger.error(f"Error during NSGA-II experiment {exp_num}: {str(e)}")
        log_experiment_end(logger, exp_num, success=False, message=str(e))
        raise
    finally:
        close_experiment_logging(logger)
        # Aggiungi info di completamento al file di output
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        with open(terminal_output_file, 'a', encoding='utf-8') as term_file:
            term_file.write(f"\n=== EXPERIMENT {exp_num} COMPLETED ===\n")
            term_file.write(f"End time: {end_time}\n")
            term_file.write(f"Duration: {duration}\n")
            term_file.write("=" * 80 + "\n")


if __name__ == "__main__":
    # Experiment 1
    # run_nsga2_experiment(1, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl"))
    # run_nsga2_experiment(1, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl"))
    # run_nsga2_experiment(1, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl"))
    run_nsga2_experiment(1, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl"))

    # Experiment 2
    run_nsga2_experiment(2, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))
    run_nsga2_experiment(2, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))
    run_nsga2_experiment(2, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))
    run_nsga2_experiment(2, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))

    """

    # Experiment 3
    run_nsga2_experiment(3, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))
    run_nsga2_experiment(3, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))
    run_nsga2_experiment(3, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))
    run_nsga2_experiment(3, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))



    # Experiment 4
    # run_nsga2_experiment(4, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))
    run_nsga2_experiment(4, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))
    run_nsga2_experiment(4, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))
    run_nsga2_experiment(4, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))

    # Experiment 5
    run_nsga2_experiment(5, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    run_nsga2_experiment(5, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    run_nsga2_experiment(5, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    run_nsga2_experiment(5, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    

    # Experiment 6
    run_nsga2_experiment(6, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))
    run_nsga2_experiment(6, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))
    run_nsga2_experiment(6, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))
    run_nsga2_experiment(6, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))

    # Experiment 7
    run_nsga2_experiment(7, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))
    run_nsga2_experiment(7, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))
    run_nsga2_experiment(7, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))
    run_nsga2_experiment(7, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))

    # Experiment 8
    run_nsga2_experiment(8, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_8.pkl"))
    run_nsga2_experiment(8, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_8.pkl"))
    run_nsga2_experiment(8, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_8.pkl"))
    run_nsga2_experiment(8, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_8.pkl"))

    # Experiment 9
    run_nsga2_experiment(9, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_9.pkl"))
    run_nsga2_experiment(9, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_9.pkl"))
    run_nsga2_experiment(9, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_9.pkl"))
    run_nsga2_experiment(9, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_9.pkl"))

    # Experiment 10
    run_nsga2_experiment(10, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_10.pkl"))
    run_nsga2_experiment(10, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_10.pkl"))
    run_nsga2_experiment(10, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_10.pkl"))
    run_nsga2_experiment(10, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_10.pkl"))

    # Experiment 11
    run_nsga2_experiment(11, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_11.pkl"))
    run_nsga2_experiment(11, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_11.pkl"))
    run_nsga2_experiment(11, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_11.pkl"))
    run_nsga2_experiment(11, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_11.pkl"))

    # Experiment 12
    run_nsga2_experiment(12, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_12.pkl"))
    run_nsga2_experiment(12, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_12.pkl"))
    run_nsga2_experiment(12, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_12.pkl"))
    run_nsga2_experiment(12, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_12.pkl"))

    # Experiment 13
    run_nsga2_experiment(13, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_13.pkl"))
    run_nsga2_experiment(13, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_13.pkl"))
    run_nsga2_experiment(13, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_13.pkl"))
    run_nsga2_experiment(13, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_13.pkl"))

    # Experiment 14
    run_nsga2_experiment(14, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_14.pkl"))
    run_nsga2_experiment(14, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_14.pkl"))
    run_nsga2_experiment(14, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_14.pkl"))
    run_nsga2_experiment(14, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_14.pkl"))
    

    """# Experiment 15
    run_nsga2_experiment(15, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_15.pkl"))
    run_nsga2_experiment(15, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_15.pkl"))
    run_nsga2_experiment(15, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_15.pkl"))
    run_nsga2_experiment(15, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_15.pkl"))


    # Experiment 16
    run_nsga2_experiment(16, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_16.pkl"))
    run_nsga2_experiment(16, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_16.pkl"))
    run_nsga2_experiment(16, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_16.pkl"))
    run_nsga2_experiment(16, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_16.pkl"))

    # Experiment 17
    run_nsga2_experiment(17, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_17.pkl"))
    run_nsga2_experiment(17, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_17.pkl"))
    run_nsga2_experiment(17, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_17.pkl"))
    run_nsga2_experiment(17, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_17.pkl"))

    # Experiment 18
    run_nsga2_experiment(18, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_18.pkl"))
    run_nsga2_experiment(18, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_18.pkl"))
    run_nsga2_experiment(18, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_18.pkl"))
    run_nsga2_experiment(18, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_18.pkl"))

    # Experiment 19
    run_nsga2_experiment(19, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_19.pkl"))
    run_nsga2_experiment(19, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_19.pkl"))
    run_nsga2_experiment(19, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_19.pkl"))
    run_nsga2_experiment(19, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_19.pkl"))

    # Experiment 20
    run_nsga2_experiment(20, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_20.pkl"))
    run_nsga2_experiment(20, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_20.pkl"))
    run_nsga2_experiment(20, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_20.pkl"))
    run_nsga2_experiment(20, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_20.pkl"))
"""

    # Experiment 21
    run_nsga2_experiment(21, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_21.pkl"))
    run_nsga2_experiment(21, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_21.pkl"))
    run_nsga2_experiment(21, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_21.pkl"))
    run_nsga2_experiment(21, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_21.pkl"))

    # Experiment 22
    run_nsga2_experiment(22, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_22.pkl"))
    run_nsga2_experiment(22, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_22.pkl"))
    run_nsga2_experiment(22, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_22.pkl"))
    run_nsga2_experiment(22, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_22.pkl"))

    # Experiment 23
    run_nsga2_experiment(23, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_23.pkl"))
    run_nsga2_experiment(23, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_23.pkl"))
    run_nsga2_experiment(23, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_23.pkl"))
    run_nsga2_experiment(23, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_23.pkl"))

    # Experiment 24
    run_nsga2_experiment(24, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_24.pkl"))
    run_nsga2_experiment(24, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_24.pkl"))
    run_nsga2_experiment(24, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_24.pkl"))
    run_nsga2_experiment(24, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_24.pkl"))
    
    # Experiment 25
    run_nsga2_experiment(25, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_25.pkl"))
    run_nsga2_experiment(25, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_25.pkl"))
    run_nsga2_experiment(25, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_25.pkl"))
    run_nsga2_experiment(25, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_25.pkl"))

    # Experiment 26
    run_nsga2_experiment(26, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_26.pkl"))
    run_nsga2_experiment(26, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_26.pkl"))
    run_nsga2_experiment(26, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_26.pkl"))
    run_nsga2_experiment(26, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_26.pkl"))

    # Experiment 28
    run_nsga2_experiment(28, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_28.pkl"))
    run_nsga2_experiment(28, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_28.pkl"))
    run_nsga2_experiment(28, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_28.pkl"))
    run_nsga2_experiment(28, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_28.pkl"))

    # Experiment 30
    run_nsga2_experiment(30, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_30.pkl"))
    run_nsga2_experiment(30, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_30.pkl"))
    run_nsga2_experiment(30, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_30.pkl"))
    run_nsga2_experiment(30, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_30.pkl"))"""