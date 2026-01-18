"""
Script where the creation and run of the MILP model and GA are tested to create a schedule based on benchmark datasets. 
First, a schedule is created using the GA presented in the following paper: "An effective genetic algorithm for the flexible job-shop scheduling problem" 
Guohui Zhang, Liang Gao, Yang Shi, Expert Systems with Applications, Volume 38, Issue 4, 2011, Pages 3563-3573. The algorithm will execute for 100 generations and a population of 100 
individuals. A visualization of the schedule with gantt chart is generated, and a plot of the evolution of the makespan over time is generated. 
The solution found with the GA is, then, used as an initial solution for the MILP to reduce the computation times. A visualization of the schedule with gantt chart is generated, and 
a plot of the evolution of the makespan over time is generated. 
The Jobshop environment with the final schedule is saved in a pickle file. 
Then, the rescheduling MILP is applied to the current schedule, specifying the moment in which the machine broke and the id of the broken machine. 
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

    # EX 8
    # 15 jobs, 8 machines, 293 operations
    # run_ga_milp('/fjsp/dauzere_paulli/09a.fjs', 8, 3600)

    # EX 9
    # 10 jobs, 6 machines, 55 operations
    # run_ga_milp('/fjsp/brandimarte/Mk01.fjs', 9, 3600)

    # EX 10
    # run_ga_milp('/fjsp/brandimarte/Mk04.fjs', 10, 3600)

    # EX 11
    # run_ga_milp('/fjsp/brandimarte/Mk05.fjs', 11, 3600)

    # EX 12
    # run_ga_milp('/fjsp/brandimarte/Mk07.fjs', 12, 3600)

    # EX 13
    # run_ga_milp('/fjsp/brandimarte/Mk08.fjs', 13, 3600)

    # EX 14
    # run_ga_milp('/fjsp/brandimarte/Mk09.fjs', 14, 3600)

    # EX 15
    run_ga_milp('/fjsp/kacem/Kacem2.fjs', 15, 3600)

    # EX 16
    run_ga_milp('/fjsp/kacem/Kacem3.fjs', 16, 3600)

    # EX 17
    run_ga_milp('/fjsp/fattahi/MFJS1.fjs', 17, 3600)

    # EX 18
    run_ga_milp('/fjsp/fattahi/MFJS2.fjs', 18, 3600)

    # EX 19
    run_ga_milp('/fjsp/fattahi/MFJS3.fjs', 19, 3600)

    # EX 20
    run_ga_milp('/fjsp/fattahi/MFJS4.fjs', 20, 3600)

    # EX 21
    run_ga_milp('/fjsp/fattahi/MFJS5.fjs', 21, 3600)

    # EX 22
    run_ga_milp('/fjsp/fattahi/MFJS6.fjs', 22, 3600)

    # EX 23
    run_ga_milp('/fjsp/fattahi/MFJS7.fjs', 23, 3600)

    # EX 24
    run_ga_milp('/fjsp/fattahi/MFJS8.fjs', 24, 3600)

    # EX 25
    run_ga_milp('/fjsp/fattahi/MFJS9.fjs', 25, 3600)

    # EX 26
    run_ga_milp('/fjsp/dauzere_paulli/12a.fjs', 26, 3600)

    # EX 27
    run_ga_milp('/fjsp/dauzere_paulli/14a.fjs', 27, 3600)

    # EX 28
    run_ga_milp('/fjsp/dauzere_paulli/15a.fjs', 28, 3600)

    # EX 29
    run_ga_milp('/fjsp/dauzere_paulli/17a.fjs', 29, 3600)

    # EX 30
    run_ga_milp('/fjsp/dauzere_paulli/18a.fjs', 30, 3600)



#   EX 8
# 15 jobs, 8 machines, 293 operations
"""
jobShopEnv8 = parse_fjsp('/fjsp/dauzere_paulli/09a.fjs')

# GA

population, toolbox, stats, hof = initialize_run(jobShopEnv8, **parameters)
start_8 = time.time()
makespan, jobShopEnv8, fitness_progress = run_GA(jobShopEnv8, population, toolbox, stats, hof, **parameters)
end_8 = time.time()
duration_8 = end_8 - start_8
print(f'Computation time: {duration_8}.')

plot_fitness_progress_time(fitness_progress, save_path=os.path.join(RESULTS_DIR, "EX8", "makespan_evolution_GA_8.png"))

gantt_chart.plot(jobShopEnv8)
plt.savefig(os.path.join(RESULTS_DIR, "EX8", "schedule_gantt_ex8_GA.png"), dpi=600)


# MILP (starting from GA)

schedule8 = FJS_schedule(jobShopEnv8)
schedule8.create_model(start_solution=True)
schedule8.run_model(time_limit=3600)

res8=schedule8.incumbent

plt.figure()
plt.plot(res8.times, res8.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(os.path.join(RESULTS_DIR, "EX8", "objective_evolution_ex8_milp+ga.png"), dpi=300)
plt.close()

schedule8.update_environment()
file_path_8 = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_8.pkl")
with open(file_path_8, "wb") as f:
    pickle.dump(jobShopEnv8, f)

plt.figure()
gantt_chart.plot(jobShopEnv8)
plt.savefig(os.path.join(RESULTS_DIR, "EX8", "schedule_gantt_ex8.png"), dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv8.makespan}')

print(f'Unique machines: {jobShopEnv8.unique_machine_ids}')
"""

"""SAVE_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results_scheduling"
SAVE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print(SAVE_DIR.resolve())"""


"""# EX 1
# 4 jobs, 5 machines, 12 operations

jobShopEnv1 = parse_fjsp('/fjsp/kacem/Kacem1.fjs')


# GA

population, toolbox, stats, hof = initialize_run(jobShopEnv1, **parameters)
start_1 = time.time()
makespan, jobShopEnv1, fitness_progress = run_GA(jobShopEnv1, population, toolbox, stats, hof, **parameters)
end_1 = time.time()
duration_1 = end_1 - start_1
print(f'Computation time: {duration_1}.')

save_path1 = os.path.join(RESULTS_DIR, "EX1", "makespan_evolution_GA_1.png")
plot_fitness_progress_time(fitness_progress, save_path=save_path1, show_plot=False)
plt.close()

save_path1 = os.path.join(RESULTS_DIR, "EX1", "makespan_evolution_GA_gen_1.png")
plot_fitness_progress_gen(fitness_progress, save_path=save_path1, show_plot=False)
plt.close()

gantt_chart.plot(jobShopEnv1)
save_path1 = os.path.join(RESULTS_DIR, "EX1", "schedule_gantt_ex1_GA.png")
plt.savefig(save_path1, dpi=300)


# MILP (starting from GA)

schedule1 = FJS_schedule(jobShopEnv1)
schedule1.create_model(start_solution=True)
schedule1.run_model(time_limit=3600)

res1=schedule1.incumbent

plt.figure()
plt.plot(res1.times, res1.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(os.path.join(RESULTS_DIR, "EX1", "objective_evolution_ex1_milp+ga.png"), dpi=300)
plt.close()

schedule1.update_environment()
file_path_1 = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl")
with open(file_path_1, "wb") as f:
    pickle.dump(jobShopEnv1, f)

plt.figure()
gantt_chart.plot(jobShopEnv1)
plt.savefig(os.path.join(RESULTS_DIR, "EX1", "schedule_gantt_ex1.png"), dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv1.makespan}')

print(f'Unique machines: {jobShopEnv1.unique_machine_ids}')




# EX 2
# 10 jobs, 6 machines, 58 operations

jobShopEnv2 = parse_fjsp('/fjsp/brandimarte/Mk02.fjs')


# GA

population, toolbox, stats, hof = initialize_run(jobShopEnv2, **parameters)
start_2 = time.time()
makespan, jobShopEnv2, fitness_progress = run_GA(jobShopEnv2, population, toolbox, stats, hof, **parameters)
end_2 = time.time()
duration_2 = end_2 - start_2
print(f'Computation time: {duration_2}.')

plot_fitness_progress_time(fitness_progress, save_path=os.path.join(RESULTS_DIR, "EX2", "makespan_evolution_GA_2.png"), show_plot=False)
plt.close()

gantt_chart.plot(jobShopEnv2)
plt.savefig(os.path.join(RESULTS_DIR, "EX2", "schedule_gantt_ex2_GA.png"), dpi=300)


# MILP (starting from GA)

schedule2 = FJS_schedule(jobShopEnv2)
schedule2.create_model(start_solution=True)
schedule2.run_model(time_limit=3600)

res2=schedule2.incumbent

plt.figure()
plt.plot(res2.times, res2.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(os.path.join(RESULTS_DIR, "EX2", "objective_evolution_ex2_milp+ga.png"), dpi=300)
plt.close()

schedule2.update_environment()
file_path_2 = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl")
with open(file_path_2, "wb") as f:
    pickle.dump(jobShopEnv2, f)

plt.figure()
gantt_chart.plot(jobShopEnv2)
plt.savefig(os.path.join(RESULTS_DIR, "EX2", "schedule_gantt_ex2.png"), dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv2.makespan}')

print(f'Unique machines: {jobShopEnv2.unique_machine_ids}')



# EX 3
# 20 jobs, 10 machines, 240 operations

jobShopEnv3 = parse_fjsp('/fjsp/brandimarte/Mk10.fjs')


# GA

population, toolbox, stats, hof = initialize_run(jobShopEnv3, **parameters)
start_3 = time.time()
makespan, jobShopEnv3, fitness_progress = run_GA(jobShopEnv3, population, toolbox, stats, hof, **parameters)
end_3 = time.time()
duration_3 = end_3 - start_3
print(f'Computation time: {duration_3}.')

plot_fitness_progress_time(fitness_progress, save_path=os.path.join(RESULTS_DIR, "EX3", "makespan_evolution_GA_3.png"))

gantt_chart.plot(jobShopEnv3)
plt.savefig(os.path.join(RESULTS_DIR, "EX3", "schedule_gantt_ex3_GA.png"), dpi=300)


# MILP (starting from GA)

schedule3 = FJS_schedule(jobShopEnv3)
schedule3.create_model(start_solution=True)
schedule3.run_model(time_limit=3600)

res3=schedule3.incumbent

plt.figure()
plt.plot(res3.times, res3.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(os.path.join(RESULTS_DIR, "EX3", "objective_evolution_ex3_milp+ga.png"), dpi=300)
plt.close()

schedule3.update_environment()
file_path_3 = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl")
with open(file_path_3, "wb") as f:
    pickle.dump(jobShopEnv3, f)

plt.figure()
gantt_chart.plot(jobShopEnv3)
plt.savefig(os.path.join(RESULTS_DIR, "EX3", "schedule_gantt_ex3.png"), dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv3.makespan}')

print(f'Unique machines: {jobShopEnv3.unique_machine_ids}')



#   EX 4
# 12 jobs, 8 machines, 48 operations

jobShopEnv4 = parse_fjsp('/fjsp/fattahi/MFJS10.fjs')


# GA

population, toolbox, stats, hof = initialize_run(jobShopEnv4, **parameters)
start_4 = time.time()
makespan, jobShopEnv4, fitness_progress = run_GA(jobShopEnv4, population, toolbox, stats, hof, **parameters)
end_4 = time.time()
duration_4 = end_4 - start_4
print(f'Computation time: {duration_4}.')

plot_fitness_progress_time(fitness_progress, save_path=os.path.join(RESULTS_DIR, "EX4","makespan_evolution_GA_4.png"))

gantt_chart.plot(jobShopEnv4)
plt.savefig(os.path.join(RESULTS_DIR, "EX4", "schedule_gantt_ex4_GA.png"), dpi=600)


# MILP (starting from GA)

schedule4 = FJS_schedule(jobShopEnv4)
schedule4.create_model(start_solution=True)
schedule4.run_model(time_limit=3600)

res4=schedule4.incumbent

plt.figure()
plt.plot(res4.times, res4.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(os.path.join(RESULTS_DIR, "EX4", "objective_evolution_ex4_milp+ga.png"), dpi=300)
plt.close()

schedule4.update_environment()
file_path_4 = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_4.pkl")
with open(file_path_4, "wb") as f:
    pickle.dump(jobShopEnv4, f)

plt.figure()
gantt_chart.plot(jobShopEnv4)
plt.savefig(os.path.join(RESULTS_DIR, "EX4", "schedule_gantt_ex4.png"), dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv4.makespan}')

print(f'Unique machines: {jobShopEnv4.unique_machine_ids}')



#   EX 5
# 15 jobs, 10 machines, 56 operations

jobShopEnv5 = parse_fjsp('/fjsp/kacem/Kacem4.fjs')


# GA

population, toolbox, stats, hof = initialize_run(jobShopEnv5, **parameters)
start_5 = time.time()
makespan, jobShopEnv4, fitness_progress = run_GA(jobShopEnv5, population, toolbox, stats, hof, **parameters)
end_5 = time.time()
duration_5 = end_5 - start_5
print(f'Computation time: {duration_5}.')

plot_fitness_progress_time(fitness_progress, save_path=os.path.join(RESULTS_DIR, "EX5", "makespan_evolution_GA_5.png"))

gantt_chart.plot(jobShopEnv5)
plt.savefig(os.path.join(RESULTS_DIR, "EX5", "schedule_gantt_ex5_GA.png"), dpi=600)


# MILP (starting from GA)

schedule5 = FJS_schedule(jobShopEnv5)
schedule5.create_model(start_solution=True)
schedule5.run_model(time_limit=3600)

res5=schedule5.incumbent

plt.figure()
plt.plot(res5.times, res5.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(os.path.join(RESULTS_DIR, "EX5", "objective_evolution_ex5_milp+ga.png"), dpi=300)
plt.close()

schedule5.update_environment()
file_path_5 = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_5.pkl")
with open(file_path_5, "wb") as f:
    pickle.dump(jobShopEnv5, f)

plt.figure()
gantt_chart.plot(jobShopEnv5)
plt.savefig(os.path.join(RESULTS_DIR, "EX5", "schedule_gantt_ex5.png"), dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv5.makespan}')

print(f'Unique machines: {jobShopEnv5.unique_machine_ids}')





#   EX 6
# 15 jobs, 8 machines, 150 operations

jobShopEnv6 = parse_fjsp('/fjsp/brandimarte/Mk03.fjs')


# GA

population, toolbox, stats, hof = initialize_run(jobShopEnv6, **parameters)
start_6 = time.time()
makespan, jobShopEnv6, fitness_progress = run_GA(jobShopEnv6, population, toolbox, stats, hof, **parameters)
end_6 = time.time()
duration_6 = end_6 - start_6
print(f'Computation time: {duration_6}.')

plot_fitness_progress_time(fitness_progress, save_path=os.path.join(RESULTS_DIR, "EX6", "makespan_evolution_GA_6.png"))

gantt_chart.plot(jobShopEnv6)
plt.savefig(os.path.join(RESULTS_DIR, "EX6", "schedule_gantt_ex6_GA.png"), dpi=600)


# MILP (starting from GA)

schedule6 = FJS_schedule(jobShopEnv6)
schedule6.create_model(start_solution=True)
schedule6.run_model(time_limit=3600)

res6=schedule6.incumbent

plt.figure()
plt.plot(res6.times, res6.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(os.path.join(RESULTS_DIR, "EX6", "objective_evolution_ex6_milp+ga.png"), dpi=300)
plt.close()

schedule6.update_environment()
file_path_6 = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_6.pkl")
with open(file_path_6, "wb") as f:
    pickle.dump(jobShopEnv6, f)

plt.figure()
gantt_chart.plot(jobShopEnv6)
plt.savefig(os.path.join(RESULTS_DIR, "EX6", "schedule_gantt_ex6.png"), dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv6.makespan}')

print(f'Unique machines: {jobShopEnv6.unique_machine_ids}')




#   EX 7
# 10 jobs, 10 machines, 150 operations

jobShopEnv7 = parse_fjsp('/fjsp/brandimarte/Mk06.fjs')


# GA

population, toolbox, stats, hof = initialize_run(jobShopEnv7, **parameters)
start_7 = time.time()
makespan, jobShopEnv7, fitness_progress = run_GA(jobShopEnv7, population, toolbox, stats, hof, **parameters)
end_7 = time.time()
duration_7 = end_7 - start_7
print(f'Computation time: {duration_7}.')

plot_fitness_progress_time(fitness_progress, save_path=os.path.join(RESULTS_DIR, "EX7", "makespan_evolution_GA_7.png"))

gantt_chart.plot(jobShopEnv7)
plt.savefig(os.path.join(RESULTS_DIR, "EX7", "schedule_gantt_ex7_GA.png"), dpi=600)


# MILP (starting from GA)

schedule7 = FJS_schedule(jobShopEnv7)
schedule7.create_model(start_solution=True)
schedule7.run_model(time_limit=3600)

res7=schedule7.incumbent

plt.figure()
plt.plot(res7.times, res7.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(os.path.join(RESULTS_DIR, "EX7", "objective_evolution_ex7_milp+ga.png"), dpi=300)
plt.close()

schedule7.update_environment()
file_path_7 = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_7.pkl")
with open(file_path_7, "wb") as f:
    pickle.dump(jobShopEnv7, f)

plt.figure()
gantt_chart.plot(jobShopEnv7)
plt.savefig(os.path.join(RESULTS_DIR, "EX7", "schedule_gantt_ex7.png"), dpi=300)
plt.close()

print(f'Makespan: {jobShopEnv7.makespan}')

print(f'Unique machines: {jobShopEnv7.unique_machine_ids}')

"""


