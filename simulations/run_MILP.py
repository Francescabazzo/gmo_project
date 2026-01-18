
import pickle
import matplotlib
matplotlib.use("Agg")  # backend non-GUI
import matplotlib.pyplot as plt
import time
from pathlib import Path
import logging
logging.getLogger('gurobipy').setLevel(logging.WARNING)

from data.data_parsers.parser_fjsp import parse_fjsp
from solution_methods.helper_functions import load_parameters
from MILP.model_MILP import FJS_schedule
from MILP.model_MILP_reschedule import FJS_reschedule
from visualization import gantt_chart

from solution_methods.GA.run_GA import run_GA
from solution_methods.GA.utils import plot_fitness_progress_time, plot_fitness_progress_gen
from solution_methods.GA.src.initialization import initialize_run

parameters = {"instance": {"problem_instance": "custom_problem_instance"},
             "algorithm": {"population_size": 100, "ngen": 100, "seed": 5, "cr": 0.7, "indpb": 0.2, 'multiprocessing': True},
             "output": {"logbook": True, "save_results": True}
             }

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

SAVE_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results_scheduling"
SAVE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print(SAVE_DIR.resolve())



# EX 1
# 4 jobs, 5 machines, 12 operations

jobShopEnv1 = parse_fjsp('/fjsp/kacem/Kacem1.fjs')


schedule1 = FJS_schedule(jobShopEnv1)
schedule1.create_model()
schedule1.run_model(time_limit=1200)

res1=schedule1.incumbent

plt.figure()
plt.plot(res1.times, res1.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(SAVE_DIR / "EX1" / "objective_evolution_ex1_milp.png", dpi=300)
plt.close()

plt.figure()
gantt_chart.plot(jobShopEnv1)
plt.savefig(SAVE_DIR / "EX1" / "schedule_gantt_ex1_milp.png", dpi=300)
plt.close()



# EX 2
# 10 jobs, 6 machines, 58 operations

jobShopEnv2 = parse_fjsp('/fjsp/brandimarte/Mk02.fjs')


schedule2 = FJS_schedule(jobShopEnv2)
schedule2.create_model()
schedule2.run_model(time_limit=1200)

res2=schedule2.incumbent

plt.figure()
plt.plot(res2.times, res2.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(SAVE_DIR / "EX2"/ "objective_evolution_ex2_milp.png", dpi=300)
plt.close()

plt.figure()
gantt_chart.plot(jobShopEnv2)
plt.savefig(SAVE_DIR / "EX2"/ "schedule_gantt_ex2_milp.png", dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv2.makespan}')



# EX 3
# 20 jobs, 10 machines, 240 operations

jobShopEnv3 = parse_fjsp('/fjsp/brandimarte/Mk10.fjs')


schedule3 = FJS_schedule(jobShopEnv3)
schedule3.create_model()
schedule3.run_model(time_limit=1200)

res3=schedule3.incumbent

plt.figure()
plt.plot(res3.times, res3.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(SAVE_DIR / "EX3"/ "objective_evolution_ex3_milp.png", dpi=300)
plt.close()

plt.figure()
gantt_chart.plot(jobShopEnv3)
plt.savefig(SAVE_DIR / "EX3"/ "schedule_gantt_ex3_milp.png", dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv3.makespan}')


#   EX 4
# 12 jobs, 8 machines, 48 operations

jobShopEnv4 = parse_fjsp('/fjsp/fattahi/MFJS10.fjs')


schedule4 = FJS_schedule(jobShopEnv4)
schedule4.create_model()
schedule4.run_model(time_limit=1200)

res4=schedule4.incumbent

plt.figure()
plt.plot(res4.times, res4.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(SAVE_DIR / "EX4"/ "objective_evolution_ex4_milp.png", dpi=300)
plt.close()

plt.figure()
gantt_chart.plot(jobShopEnv4)
plt.savefig(SAVE_DIR / "EX4"/ "schedule_gantt_ex4_milp.png", dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv4.makespan}')



#   EX 5
# 15 jobs, 10 machines, 56 operations

jobShopEnv5 = parse_fjsp('/fjsp/kacem/Kacem4.fjs')


schedule5 = FJS_schedule(jobShopEnv5)
schedule5.create_model()
schedule5.run_model(time_limit=1200)

res5=schedule5.incumbent

plt.figure()
plt.plot(res5.times, res5.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(SAVE_DIR / "EX5"/ "objective_evolution_ex5_milp.png", dpi=300)
plt.close()

plt.figure()
gantt_chart.plot(jobShopEnv5)
plt.savefig(SAVE_DIR / "EX5"/ "schedule_gantt_ex5_milp.png", dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv5.makespan}')