import sys
import os
import datetime
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

SIMULATIONS_DIR = PROJECT_ROOT / "simulations"
NSGA2_DIR = PROJECT_ROOT / "NSGA2"
SRC_DIR = NSGA2_DIR / "src"
RESULTS_DIR = PROJECT_ROOT / "results" / "NSGA2"
SAVE_DIR = PROJECT_ROOT / "results" / "NSGA2_ASS"
RESULTS_SCH_DIR = PROJECT_ROOT / "results_scheduling"
SAVE_LOGS = SIMULATIONS_DIR / "logs" / "NSGA2_ASS"
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
    setup_logging_directory, setup_experiment_logging, close_experiment_logging, log_experiment_start,
    log_experiment_end, load_environment, log_environment_info
)

from NSGA2.nsga_ii import compute_upper_bound

from NSGA2.src.random_initialization import RandomInitializer
from NSGA2.src.decode import Decoder
from NSGA2.src.schedule_manager import ScheduleManager
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


def run_nsga2_assessment(exp_num, n_pop, file_path):
    """
    MILP-based assessment of NSGA-II Pareto front.
    Computes local LOWER BOUNDS (minimization) for each objective.
    """

    actual_log_dir = SAVE_LOGS / f"EX{exp_num}" / f"Pop_{n_pop}"
    actual_log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_experiment_logging(
        log_dir=SAVE_LOGS,
        exp_num=exp_num,
        log_filename=str(actual_log_dir / f"assessment_EX{exp_num}.txt")
    )

    terminal_output_file = actual_log_dir / f"terminal_output_EX{exp_num}.txt"
    start_time = datetime.datetime.now()

    try:
        results_dir = SAVE_DIR / f"EX{exp_num}" / f"Pop_{n_pop}"
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(terminal_output_file, "w", encoding="utf-8") as term_file:
            tee_out = Tee(sys.__stdout__, term_file)
            tee_err = Tee(sys.__stderr__, term_file)

            with redirect_stdout(tee_out), redirect_stderr(tee_err):

                print(f"=== ASSESSMENT EXPERIMENT {exp_num} STARTED ===")

                # -------- Load Pareto CSV --------
                pareto_csv = (
                    RESULTS_DIR
                    / f"EX{exp_num}"
                    / f"Pop_{n_pop}"
                    / "pareto_solutions_full.csv"
                )

                df = pd.read_csv(pareto_csv)
                print(f"Loaded {len(df)} Pareto solutions")

                # -------- Load environment --------
                jobShopEnv = load_environment(file_path, logger)
                data_break = get_data_break()

                broken_machine_id = data_break[f"EX{exp_num}"]["broken_machine_id"]
                disruption_time = data_break[f"EX{exp_num}"]["disruption_time"]

                log_experiment_start(
                    logger,
                    exp_num,
                    {
                        "file_path": file_path,
                        "broken_machine_id": broken_machine_id,
                        "disruption_time": disruption_time,
                    }
                )

                log_environment_info(logger, jobShopEnv)

                initializer = RandomInitializer(
                    jobShopEnv, disruption_time, broken_machine_id
                )
                schedule_manager = ScheduleManager(jobShopEnv)

                decoder = Decoder(
                    jobShopEnv,
                    broken_machine_id,
                    disruption_time,
                    initializer,
                    schedule_manager
                )

                assessment_results = []

                # -------- Assessment loop --------
                for idx, row in df.iterrows():
                    print(f"\nAssessing solution {idx + 1}/{len(df)}")

                    MS_str = row["MS"]
                    OS_str = row["OS"]

                    MS = [int(x) for x in MS_str.split("-")]
                    OS = [int(x) for x in OS_str.split("-")]

                    print(f"MS: {MS}, OS: {OS}")

                    # individual = (MS, OS)
                    individual = MS + OS

                    X, Y, S, C, E, Z, Z_aux = decoder.extract_sets(individual)

                    # --- LOWER BOUNDS (1 objective at a time) ---
                    lb_time = compute_upper_bound(
                        jobShopEnv,
                        broken_machine_id,
                        disruption_time,
                        X,
                        "quadr_delay"
                    )

                    lb_assign = compute_upper_bound(
                        jobShopEnv,
                        broken_machine_id,
                        disruption_time,
                        X,
                        "mach_assign"
                    )

                    lb_makespan = compute_upper_bound(
                        jobShopEnv,
                        broken_machine_id,
                        disruption_time,
                        X,
                        "cmax"
                    )

                    assessment_results.append({
                        "solution_id": idx,

                        "orig_time_diff": row["Time Differences"],
                        "orig_assign_diff": row["Assignment Differences"],
                        "orig_makespan": row["Makespan"],

                        "lb_time_diff": lb_time[0],
                        "lb_assign_diff": lb_assign[1],
                        "lb_makespan": lb_makespan[2],

                        "gap_time_diff": (
                            row["Time Differences"] - lb_time[0]
                        ) / max(row["Time Differences"], 1e-6),

                        "gap_assign_diff": (
                            row["Assignment Differences"] - lb_assign[1]
                        ) / max(row["Assignment Differences"], 1e-6),

                        "gap_makespan": (
                            row["Makespan"] - lb_makespan[2]
                        ) / max(row["Makespan"], 1e-6),
                    })

                # -------- Save CSV --------
                out_csv = results_dir / "assessment_lower_bounds.csv"
                pd.DataFrame(assessment_results).to_csv(out_csv, index=False)

                print(f"\nAssessment completed.")
                print(f"Results saved to: {out_csv}")

                log_experiment_end(logger, exp_num, success=True)

    except Exception as e:
        traceback.print_exc()
        logger.error(str(e))
        log_experiment_end(logger, exp_num, success=False, message=str(e))
        raise

    finally:
        close_experiment_logging(logger)



if __name__ == "__main__":
    # Experiment 1
    run_nsga2_assessment(1, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl"))
    run_nsga2_assessment(1, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl"))
    run_nsga2_assessment(1, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl"))
    run_nsga2_assessment(1, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl"))

    # Experiment 2
    run_nsga2_assessment(2, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))
    run_nsga2_assessment(2, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))
    run_nsga2_assessment(2, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))
    run_nsga2_assessment(2, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"))

    """
    # Experiment 3
    run_nsga2_assessment(3, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))
    run_nsga2_assessment(3, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))
    run_nsga2_assessment(3, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))
    run_nsga2_assessment(3, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"))

    # Experiment 4
    # run_nsga2_assessment(4, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))
    run_nsga2_assessment(4, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))
    run_nsga2_assessment(4, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))
    run_nsga2_assessment(4, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl"))

    # Experiment 5
    run_nsga2_assessment(5, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    run_nsga2_assessment(5, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    run_nsga2_assessment(5, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))
    run_nsga2_assessment(5, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl"))

    # Experiment 6
    run_nsga2_assessment(6, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))
    run_nsga2_assessment(6, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))
    run_nsga2_assessment(6, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))
    run_nsga2_assessment(6, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl"))

    # Experiment 7
    run_nsga2_assessment(7, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))
    run_nsga2_assessment(7, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))
    run_nsga2_assessment(7, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))
    run_nsga2_assessment(7, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl"))

    # Experiment 8
    run_nsga2_assessment(8, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_8.pkl"))
    run_nsga2_assessment(8, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_8.pkl"))
    run_nsga2_assessment(8, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_8.pkl"))
    run_nsga2_assessment(8, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_8.pkl"))

    # Experiment 9
    run_nsga2_assessment(9, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_9.pkl"))
    run_nsga2_assessment(9, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_9.pkl"))
    run_nsga2_assessment(9, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_9.pkl"))
    run_nsga2_assessment(9, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_9.pkl"))

    # Experiment 10
    run_nsga2_assessment(10, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_10.pkl"))
    run_nsga2_assessment(10, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_10.pkl"))
    run_nsga2_assessment(10, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_10.pkl"))
    run_nsga2_assessment(10, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_10.pkl"))

    # Experiment 11
    run_nsga2_assessment(11, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_11.pkl"))
    run_nsga2_assessment(11, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_11.pkl"))
    run_nsga2_assessment(11, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_11.pkl"))
    run_nsga2_experiment(11, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_11.pkl"))

    # Experiment 12
    run_nsga2_experiment(12, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_12.pkl"))
    run_nsga2_experiment(12, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_12.pkl"))
    run_nsga2_experiment(12, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_12.pkl"))
    run_nsga2_assessment(12, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_12.pkl"))

    # Experiment 13
    run_nsga2_assessment(13, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_13.pkl"))
    run_nsga2_assessment(13, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_13.pkl"))
    run_nsga2_assessment(13, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_13.pkl"))
    run_nsga2_assessment(13, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_13.pkl"))

    # Experiment 14
    run_nsga2_assessment(14, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_14.pkl"))
    run_nsga2_assessment(14, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_14.pkl"))
    run_nsga2_assessment(14, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_14.pkl"))
    run_nsga2_assessment(14, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_14.pkl"))
    """

    # Experiment 15
    run_nsga2_assessment(15, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_15.pkl"))
    run_nsga2_assessment(15, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_15.pkl"))
    run_nsga2_assessment(15, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_15.pkl"))
    run_nsga2_assessment(15, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_15.pkl"))

    # Experiment 16
    run_nsga2_assessment(16, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_16.pkl"))
    run_nsga2_assessment(16, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_16.pkl"))
    run_nsga2_assessment(16, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_16.pkl"))
    run_nsga2_assessment(16, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_16.pkl"))

    # Experiment 17
    run_nsga2_assessment(17, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_17.pkl"))
    run_nsga2_assessment(17, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_17.pkl"))
    run_nsga2_assessment(17, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_17.pkl"))
    run_nsga2_assessment(17, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_17.pkl"))

    # Experiment 18
    run_nsga2_assessment(18, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_18.pkl"))
    run_nsga2_assessment(18, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_18.pkl"))
    run_nsga2_assessment(18, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_18.pkl"))
    run_nsga2_assessment(18, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_18.pkl"))

    # Experiment 19
    run_nsga2_assessment(19, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_19.pkl"))
    run_nsga2_assessment(19, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_19.pkl"))
    run_nsga2_assessment(19, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_19.pkl"))
    run_nsga2_assessment(19, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_19.pkl"))

    # Experiment 20
    run_nsga2_assessment(20, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_20.pkl"))
    run_nsga2_assessment(20, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_20.pkl"))
    run_nsga2_assessment(20, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_20.pkl"))
    run_nsga2_assessment(20, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_20.pkl"))

    """
    # Experiment 21
    run_nsga2_assessment(21, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_21.pkl"))
    run_nsga2_assessment(21, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_21.pkl"))
    run_nsga2_assessment(21, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_21.pkl"))
    run_nsga2_assessment(21, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_21.pkl"))

    # Experiment 22
    run_nsga2_assessment(22, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_22.pkl"))
    run_nsga2_assessment(22, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_22.pkl"))
    run_nsga2_assessment(22, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_22.pkl"))
    run_nsga2_assessment(22, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_22.pkl"))

    # Experiment 23
    run_nsga2_assessment(23, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_23.pkl"))
    run_nsga2_assessment(23, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_23.pkl"))
    run_nsga2_assessment(23, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_23.pkl"))
    run_nsga2_assessment(23, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_23.pkl"))

    # Experiment 24
    run_nsga2_assessment(24, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_24.pkl"))
    run_nsga2_assessment(24, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_24.pkl"))
    run_nsga2_assessment(24, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_24.pkl"))
    run_nsga2_assessment(24, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_24.pkl"))

    # Experiment 25
    run_nsga2_assessment(25, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_25.pkl"))
    run_nsga2_assessment(25, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_25.pkl"))
    run_nsga2_assessment(25, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_25.pkl"))
    run_nsga2_assessment(25, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_25.pkl"))

    # Experiment 26
    run_nsga2_assessment(26, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_26.pkl"))
    run_nsga2_assessment(26, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_26.pkl"))
    run_nsga2_assessment(26, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_26.pkl"))
    run_nsga2_assessment(26, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_26.pkl"))

    # Experiment 28
    run_nsga2_assessment(28, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_28.pkl"))
    run_nsga2_assessment(28, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_28.pkl"))
    run_nsga2_assessment(28, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_28.pkl"))
    run_nsga2_assessment(28, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_28.pkl"))

    # Experiment 30
    run_nsga2_assessment(30, 20, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_30.pkl"))
    run_nsga2_assessment(30, 50, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_30.pkl"))
    run_nsga2_assessment(30, 100, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_30.pkl"))
    run_nsga2_assessment(30, 200, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_30.pkl"))"""