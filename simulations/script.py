import sys
from pathlib import Path
import time
import pickle
import numpy as np

import matplotlib.pyplot as plt

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

sys.path.append(str(BASE_DIR))

from solution_methods.GA.run_GA import run_GA
from solution_methods.GA.utils import plot_fitness_progress_time
from solution_methods.GA.src.initialization import initialize_run

from MILP.model_MILP import FJS_schedule

parameters = {"instance": {"problem_instance": "custom_problem_instance"},
             "algorithm": {"population_size": 100, "ngen": 100, "seed": 5, "cr": 0.7, "indpb": 0.2, 'multiprocessing': True},
             "output": {"logbook": True, "save_results": True}
             }

from NSGA2.nsga_ii import run_nsga2_with_my_operators, test_initial_population
from NSGA2.utils import (plot_pareto_front_3d_matplotlib, select_best_solution,
    update_environment_with_solution, save_results, plot_pareto_front_2d, debug_pareto_front,
    print_pareto_statistics, calculate_metrics, analyze_possible_combinations
    )
from NSGA2.src.random_initialization import RandomInitializer
from NSGA2.src.schedule_manager import ScheduleManager
from NSGA2.src.decode import Decoder
from NSGA2.src.operators import GeneticOperators

from visualization.gantt_chart import plot

from experiment_logging import load_environment

from data.dataset_generator import generate_fjsp_dataset, save_fjsp_file
from data.data_parsers.parser_fjsp import parse_fjsp

RESULTS_DIR = BASE_DIR / "results_scheduling"
SAVE_DIR = BASE_DIR / "images" / "NSGA2"
SAVE_LOGS = BASE_DIR / "simulations" / "logs" / "NSGA2"
SAVE_DATA = BASE_DIR / "data" / "random_datasets"


"""dataset = generate_fjsp_dataset(8, 6, 3, 7, 4, 10)
print(dataset)

save_fjsp_file(dataset, SAVE_DATA / "d3.fjs")

jobShop = parse_fjsp(str("/random_datasets/d3.fjs"))

population, toolbox, stats, hof = initialize_run(jobShop, **parameters)
start = time.time()
makespan, jobShop, fitness_progress = run_GA(jobShop, population, toolbox, stats, hof, **parameters)
end = time.time()
duration = end - start
print(f'Computation time: {duration}.')

plot_fitness_progress_time(fitness_progress, save_path=SAVE_DIR/ "EX8" /"makespan_evolution_GA_8.png")

plot(jobShop)
plt.savefig(SAVE_DIR / "EX8"/ "schedule_gantt_ex8_GA.png", dpi=600)


# MILP (starting from GA)

schedule = FJS_schedule(jobShop)
schedule.create_model(start_solution=True)
schedule.run_model(time_limit=2000)

res=schedule.incumbent

plt.figure()
plt.plot(res.times, res.sol_list)
plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
plt.xlabel("Computation Time (seconds)", fontsize=11)
plt.ylabel("Objective Function Value", fontsize=11)
plt.savefig(SAVE_DIR / "EX8" / "objective_evolution_ex8_milp+ga.png", dpi=300)
plt.close()

schedule.update_environment()
file_path = RESULTS_DIR / "schedule_simulation_8.pkl"
with open(file_path, "wb") as f:
    pickle.dump(jobShop, f)

plt.figure()
plot(jobShop)
plt.savefig(SAVE_DIR / "EX8" / "schedule_gantt_ex8.png", dpi=300)
plt.close()

print(f'Makespan: {jobShop.makespan}')"""

jobShop = load_environment(RESULTS_DIR / "schedule_simulation_7.pkl")

broken_machine = 2
current_time = 5

X, F, fronts = test_initial_population(jobShop, 2, 5, 200)

plot_pareto_front_3d_matplotlib(F)

schedule_manager = ScheduleManager(jobShop)
initializer = RandomInitializer(jobShop, current_time, broken_machine)
operations_toprocess, operation_times = initializer.extract_sets()
decoder = Decoder(jobShop, broken_machine, current_time,
                  initializer, schedule_manager)

test_operators = GeneticOperators(
    jobshop=jobShop,
    mutation_rate=0.1,
    crossover_rate=0.8,
    operations_toprocess=operations_toprocess,
    operation_times=operation_times,
    broken_machine=broken_machine
)

# Analyze possible unique combinations
n_unique = analyze_possible_combinations(
    decoder=decoder,
    operators=test_operators,
    initializer=initializer,
    n_samples=500  # Test 500 random solutions
)


# Executes with customized operators
res, decoder, operators = run_nsga2_with_my_operators(
    jobshop=jobShop,
    broken_machine=broken_machine,
    current_time=current_time,
    pop_size=200,    # Small population size for test
    crossover_rate=0.8,
    mutation_rate=0.1,
    termination_method="improvement"
)



print("\n=== POPULATION ANALYSIS ===")
print(f"Total population size: {len(res.X)}")
print(f"Pareto front size: {len(res.F)}")

print(f"Type of res: {type(res)}")
print(f"Attributes of res: {dir(res)}")
print(f"res.X shape: {res.X.shape if hasattr(res.X, 'shape') else 'No X'}")
print(f"res.F shape: {res.F.shape if hasattr(res.F, 'shape') else 'No F'}")

# Verifica se ci sono soluzioni con Assignment Differences
all_assign_diffs = res.F[:, 1] if len(res.F) > 0 else np.array([])

if len(all_assign_diffs) > 0:
    print(f"Assignment Differences in final population: {all_assign_diffs.min()} - {all_assign_diffs.max()}")
    unique_vals = np.unique(all_assign_diffs)
    print(f"Unique values: {unique_vals}")

    # Conta le soluzioni per ogni valore
    for val in unique_vals:
        count = np.sum(all_assign_diffs == val)
        print(f"  Value {val}: {count} solutions")

# Se il Pareto front è uguale alla popolazione, allora il problema è diverso
if len(res.F) == len(res.X):
    print("⚠️  WARNING: Pareto front size = Population size")
    print("   Questo significa che TUTTE le soluzioni sono considerate non-dominated!")

# DEBUG

print("\n" + "=" * 60)
print("🔍 SCHEDULE MANAGER DEBUG")
print("=" * 60)


pareto_front = res.F
pareto_solutions = res.X

print(f"Pareto front size: {len(pareto_front)}")

fronts = debug_pareto_front(res, pareto_front)

# Scatter().add(res.F).show()

# Show best solutions
for i, (solution, objectives) in enumerate(zip(pareto_solutions[:5], pareto_front[:5])):
    print(f"Solution {i + 1}:")
    print(f"  Time Differences: {objectives[0]:.1f}")
    print(f"  Assignment Differences: {objectives[1]:.1f}")
    print(f"  Makespan: {objectives[2]:.1f}")
    print()
# Statistics
print_pareto_statistics(pareto_front)

# Visualization (3D)
plot_pareto_front_3d_matplotlib(pareto_front)

# Visualization (2D)
plot_pareto_front_2d(pareto_front)

initializer = RandomInitializer(jobShop, current_time, broken_machine)

# Select best solution
best_solution, best_objectives = select_best_solution(
    pareto_front, pareto_solutions, method="min_time_diff"
)

# Update environment
updated_jobshop = update_environment_with_solution(
    jobShop, best_solution, decoder, initializer, current_time
)

metrics = calculate_metrics(pareto_front)

# Save everything
save_results(res, decoder)

plot(updated_jobshop)
plt.show()

print("NSGA-II completed successfully!")
