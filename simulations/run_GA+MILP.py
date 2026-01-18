"""
Script that shows the creation of the baseline schedule. 
First, a schedule is created by using the GA presented in the paper: "An effective genetic algorithm for 
the flexible job-shop scheduling problem" Guohui Zhang, Liang Gao, Yang Shi, Expert Systems with 
Applications, Volume 38, Issue 4, 2011, Pages 3563-3573.
The implementation of this algorithm was found in a public repository.
The solution is, then, used as an initial solution for the MILP to reduce the computation time. A 
visualization of the schedule with gantt chart is generated, and a plot of the evolution of the makespan 
over time is generated. 
The Jobshop environment with the final schedule is saved in a pickle file. 
"""
import os.path
import pickle
import matplotlib

matplotlib.use("Agg")  # backend non-GUI
import matplotlib.pyplot as plt
import sys
import time
from pathlib import Path
import logging
logging.getLogger('gurobipy').setLevel(logging.WARNING)

# /home/bazzo/Scrivania/fjsp_rescheduling/simulations
SCRIPT_DIR = Path(__file__).resolve().parent
# /home/bazzo/Scrivania/fjsp_rescheduling/
PROJECT_ROOT = SCRIPT_DIR.parent

# /home/bazzo/Scrivania/fjsp_rescheduling/simulations/logs
LOG_DIR = SCRIPT_DIR / "logs"
RESULTS_DIR = PROJECT_ROOT / "results" / "GA+MILP"
RESULTS_SCH_DIR = PROJECT_ROOT / "results_scheduling"

print(f"SCRIPT_DIR; {SCRIPT_DIR}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"LOG_DIR: {LOG_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"RESULTS_SCH_DIR: {RESULTS_SCH_DIR}")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LOG_DIR))
sys.path.insert(0, str(RESULTS_DIR))
sys.path.insert(0, str(RESULTS_SCH_DIR))


from data.data_parsers.parser_fjsp import parse_fjsp
from MILP.model_MILP import FJS_schedule
from visualization import gantt_chart

from solution_methods.GA.run_GA import run_GA
from solution_methods.GA.utils import plot_fitness_progress_time, plot_fitness_progress_gen 
from solution_methods.GA.src.initialization import initialize_run

parameters = {"instance": {"problem_instance": "custom_problem_instance"},
             "algorithm": {"population_size": 100, "ngen": 100, "seed": 5, "cr": 0.7, "indpb": 0.2, 'multiprocessing': True},
             "output": {"logbook": True, "save_results": True}
             }


def run_ga_milp(path_dataset: str, exp_num: int, time_limit=3600):
    """
    Creates the base schedule using first a Genetic Algorithm and then using the resulting schedule as a MIP start
    on a MILP model.
    """
    jobShopEnv = parse_fjsp(path_dataset)
    print("===============================")
    print(f"Experiment {exp_num}")
    print(f"Number of operations: {jobShopEnv.nr_of_operations}")
    print(f"Number of machines: {jobShopEnv.nr_of_machines}")
    print(f"Number of jobs: {jobShopEnv.nr_of_jobs}")
    print("===============================")
    population, toolbox, stats, hof = initialize_run(jobShopEnv, **parameters)
    start = time.time()
    makespan, jobShopEnv, fitness_progress = run_GA(jobShopEnv, population, toolbox, stats, hof, **parameters)
    end = time.time()
    duration = end - start
    print(f'Computation time: {duration}.')

    EXP_DIR = RESULTS_DIR / f"EX{exp_num}"
    os.makedirs(EXP_DIR, exist_ok=True)

    plot_fitness_progress_time(fitness_progress,
                               save_path=os.path.join(EXP_DIR, f"makespan_evolution_GA_{exp_num}.png"))

    gantt_chart.plot(jobShopEnv)
    plt.savefig(os.path.join(EXP_DIR, f"schedule_gantt_ex{exp_num}_GA.png"), dpi=600)

    # MILP (Starting from GA)
    schedule = FJS_schedule(jobShopEnv)
    schedule.create_model(start_solution=True)
    schedule.run_model(time_limit)

    res = schedule.incumbent

    plt.figure()
    plt.plot(res.times, res.sol_list)
    plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
    plt.xlabel("Computation Time (seconds)", fontsize=11)
    plt.ylabel("Objective Function Value", fontsize=11)
    plt.savefig(os.path.join(EXP_DIR, f"objective_evolution_ex{exp_num}_milp+ga.png"), dpi=300)
    plt.close()

    schedule.update_environment()
    file_path = os.path.join(RESULTS_SCH_DIR, f"schedule_simulation_{exp_num}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(jobShopEnv, f)

    plt.figure()
    gantt_chart.plot(jobShopEnv)
    plt.savefig(os.path.join(EXP_DIR, f"schedule_gantt_ex{exp_num}.png"), dpi=300)
    plt.close()

    print(f'Makespan: {jobShopEnv.makespan}')

    print(f'Unique machines: {jobShopEnv.unique_machine_ids}')



if __name__ == "__main__":
    # Experiment 1 
    run_ga_milp('/fjsp/kacem/Kacem1.fjs', 1)

    # Experiment 2
    run_ga_milp('/fjsp/brandimarte/Mk02.fjs', 2)
    
    # Experiment 3 
    run_ga_milp('/fjsp/brandimarte/Mk10.fjs', 3)