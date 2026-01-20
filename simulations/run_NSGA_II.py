import sys
import os
import datetime
import traceback
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Use non-interactive backend
matplotlib.use("Agg")

# ==============================
# Paths
# ==============================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SIMULATIONS_DIR = PROJECT_ROOT / "simulations"
NSGA2_DIR = PROJECT_ROOT / "NSGA2"
SRC_DIR = NSGA2_DIR / "src"
SAVE_DIR = PROJECT_ROOT / "results"
RESULTS_SCH_DIR = PROJECT_ROOT / "results_scheduling"
SAVE_LOGS = SIMULATIONS_DIR / "logs" / "NSGA2"
MILP_DIR = PROJECT_ROOT / "MILP"

sys.path.extend([
    str(PROJECT_ROOT),
    str(SIMULATIONS_DIR),
    str(NSGA2_DIR),
    str(SRC_DIR)
])

# ==============================
# Imports
# ==============================

from simulations.experiment_logging import (
    setup_experiment_logging,
    close_experiment_logging,
    log_experiment_start,
    log_experiment_end,
    load_environment,
    log_environment_info
)

from NSGA2.nsga_ii import run_nsga2_with_my_operators, assess_pareto_front
from NSGA2.utils import (
    plot_pareto_front_3d_matplotlib,
    select_best_solution,
    update_environment_with_solution,
    save_results,
    compute_distances,
    compute_distances_full,
    save_optimal_pareto_front_log,
    save_normalized_pareto_distance_log,
    calculate_GD,
    calculate_hypervolumes
)

from NSGA2.src.random_initialization import RandomInitializer
from NSGA2.src.decode import Decoder
from scheduling_environment.jobShop import JobShop
from visualization.gantt_chart import plot
from simulations.run_MILP_resch import get_data_break


def log_nsga2_results(logger, res, best_objectives, selection_method): 
    """ 
    Log summary statistics of the NSGA-II execution. 
    """ 
    logger.info("NSGA-II Optimization Results:") 
    logger.info(f" Number of generations completed: {res.algorithm.n_gen}") 
    logger.info(f" Number of function evaluations: {res.algorithm.evaluator.n_eval}") 
    logger.info(f" Pareto front size: {len(res.F)}") 
    logger.info(f"Best solution selected ({selection_method}):") 
    logger.info(f" Time Differences: {best_objectives[0]:.2f}") 
    logger.info(f" Assignment Differences: {best_objectives[1]:.2f}") 
    logger.info(f" Makespan: {best_objectives[2]:.2f}") 
    # Log Pareto front statistics 
    if len(res.F) > 0: 
        logger.info("Pareto Front Statistics:") 
        logger.info(f" Time Differences range: {res.F[:, 0].min():.2f} - {res.F[:, 0].max():.2f}") 
        logger.info(f" Assignment Differences range: {res.F[:, 1].min():.2f} - {res.F[:, 1].max():.2f}") 
        logger.info(f" Makespan range: {res.F[:, 2].min():.2f} - {res.F[:, 2].max():.2f}")


# ==============================
# MAIN FUNCTION
# ==============================

def run_nsga2_experiment(exp_num: int, n_pop: int, file_path: str):

    seed = int(hashlib.sha256(f"{exp_num}_{n_pop}".encode()).hexdigest(), 16) % (2**32)

    log_dir = SAVE_LOGS / f"EX{exp_num}" / f"Pop_{n_pop}"
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_experiment_logging(
        log_dir=SAVE_LOGS,
        exp_num=exp_num,
        log_filename=str(log_dir / f"experiment_{exp_num}.txt")
    )

    start_time = datetime.datetime.now()

    try:
        results_dir = SAVE_DIR / "NSGA2" / f"EX{exp_num}" / f"Pop_{n_pop}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # ======================
        # LOAD ENVIRONMENT
        # ======================
        jobShopEnv = load_environment(file_path, logger)
        data_break = get_data_break()

        broken_machine_id = data_break[f"EX{exp_num}"]["broken_machine_id"]
        disruption_time = data_break[f"EX{exp_num}"]["disruption_time"]

        log_environment_info(logger, jobShopEnv)
        log_experiment_start(logger, exp_num, {
            "file": file_path,
            "broken_machine": broken_machine_id,
            "disruption_time": disruption_time
        })

        # ======================
        # RUN NSGA-II
        # ======================
        start = datetime.datetime.now()

        res, decoder, operators = run_nsga2_with_my_operators(
            jobshop=jobShopEnv,
            broken_machine=broken_machine_id,
            current_time=disruption_time,
            seed=seed,
            pop_size=n_pop,
            crossover_rate=0.8,
            mutation_rate=0.3,
            termination_method="improvement"
        )

        duration = datetime.datetime.now() - start

        pareto_front = res.F
        pareto_solutions = res.X

        save_results(res, decoder, results_dir)

        # ======================
        # LOAD MILP LOWER BOUND
        # ======================
        with open(SAVE_DIR / "MILP" / f"EX{exp_num}" / "results.json") as f:
            obj1_lb = json.load(f)["quadratic_delay"]

        # ======================
        # PARETO ASSESSMENT
        # ======================
        optimal_front, optimal_filtered, optimal_points = assess_pareto_front(
            pareto_solutions,
            pareto_front,
            jobShopEnv,
            broken_machine_id,
            disruption_time,
            obj1_lb=obj1_lb
        )

        pd.DataFrame(optimal_points).to_csv(
            results_dir / "optimal_points_found.csv",
            index=False
        )

        # ======================
        # METRICS
        # ======================
        norm_pf, norm_opt, _ = decoder.normalize_pareto_front(
            pareto_front, optimal_filtered
        )

        GD = calculate_GD(norm_pf, norm_opt)
        HV, HV_opt = calculate_hypervolumes(norm_pf, norm_opt)

        with open(results_dir / "metrics.json", "w") as f:
            json.dump({
                "GD": float(GD),
                "Hypervolume": float(HV),
                "Hypervolume_opt": float(HV_opt),
                "Duration_seconds": duration.total_seconds()
            }, f, indent=4)

        # ======================
        # BEST SOLUTION
        # ======================
        best_solution, best_objectives = select_best_solution(
            pareto_front, pareto_solutions
        )

        updated_jobshop = update_environment_with_solution(
            jobShopEnv,
            best_solution,
            decoder,
            RandomInitializer(jobShopEnv, disruption_time, broken_machine_id),
            disruption_time
        )

        # ======================
        # PLOTS
        # ======================
        plot_pareto_front_3d_matplotlib(
            pareto_front,
            save_path=results_dir / "pareto_front.pdf"
        )

        plot(updated_jobshop)
        plt.savefig(results_dir / "gantt_chart.png")
        plt.close()

        log_nsga2_results(logger, res, best_objectives, "min_time_diff")
        log_experiment_end(logger, exp_num, success=True)

        return res

    except Exception as e:
        logger.error(str(e))
        log_experiment_end(logger, exp_num, success=False, message=str(e))
        raise

    finally:
        close_experiment_logging(logger)


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    for exp in range(1, 3):
        for pop in [20, 50, 100, 200]:
            run_nsga2_experiment(
                exp,
                pop,
                str(RESULTS_SCH_DIR / f"schedule_simulation_{exp}.pkl")
            )
