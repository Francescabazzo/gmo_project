import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Directory conntaining this file 
SCRIPT_DIR = Path(__file__).resolve().parent

# Root directory of the project 
PROJECT_ROOT = SCRIPT_DIR.parent

# Relevant project directories
SRC_DIR = SCRIPT_DIR / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_SCH_DIR = PROJECT_ROOT / "results_scheduling"
MILP_DIR = PROJECT_ROOT / "MILP"

# Debug information 
print(f"SCRIPT_DIR: {SCRIPT_DIR}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"SRC_DIR: {SRC_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"MILP_DIR: {MILP_DIR}")

# Extend the PYTHONPATH (cross-platform)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MILP_DIR))


# ============================================================
# PROJECT IMPORTS 
# ============================================================

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# MILP import (for assessment)
from MILP.model_MILP_assessment import FJS_reschedule

# NSGA2 imports
from NSGA2.src.operators import GeneticOperators
from NSGA2.src.random_initialization import RandomInitializer
from NSGA2.src.decode import Decoder
from NSGA2.src.schedule_manager import ScheduleManager
from NSGA2.utils import (
    select_best_solution,
    update_environment_with_solution,
    print_pareto_statistics,
    save_results, find_max_ms_needed
)
from NSGA2.plot import (
    plot_pareto_front_2d,
    plot_pareto_front_3d_matplotlib
)

# Simulations and visualization import
from simulations.experiment_logging import load_environment
from scheduling_environment.jobShop import JobShop
from visualization.gantt_chart import plot


class MySampling(Sampling):
    """
    Generates the initial population by adapting the RandomInitializer
    to pymoo's sampling interface.

    Args:
        initializer (RandomInitializer): Component responsible for creating feasible individuals.
    """

    def __init__(self, initializer):
        super().__init__()
        self.initializer = initializer

    def _do(self, n_samples):
        """
        Generates a set of random individuals.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Initial population with shape (n_samples, n_variables).
        """
        return self.initializer.create_population(n_samples)


class MyCrossover(Crossover):
    """
    Custom crossover operator adapted for pymoo. Applies crossover with a
    given probability using the GeneticOperators module.

    Args:
        operators (GeneticOperators): Object containing crossover logic.
        crossover_rate (float): Probability of applying crossover.
    """

    def __init__(self, operators, crossover_rate=0.8):
        super().__init__(n_parents=2, n_offsprings=2)
        self.operators = operators
        self.crossover_rate = crossover_rate

    def _do(self, x ):
        """
        Applies customized crossover to each mating pair.

        Args:
            x (np.ndarray): Parents array of shape (2, n_matings, n_variables).

        Returns:
            np.ndarray: Offspring array of shape (2, n_matings, n_variables).
        """
        n_matings = x.shape[1]
        n_var = x.shape[2]      # Length of the chromosome

        offspring = np.zeros((self.n_offsprings, n_matings, n_var), dtype=x.dtype)

        # Process each mating
        for k in range(n_matings):
            parent1 = x[0, k]  # shape: (92,) -> ms + os
            parent2 = x[1, k]  # shape: (92,) -> ms + os

            child1, child2 = self.operators.crossover(parent1, parent2)

            # Assigns children in the correct position
            offspring[0, k] = child1
            offspring[1, k] = child2

        return offspring


class MyMutation(Mutation):
    """
    Custom mutation operator adapted for pymoo. Mutates each individual
    using the GeneticOperators module.

    Args:
        operators (GeneticOperators): Object containing mutation logic.
        mutation_rate (float): Probability of applying mutation.
    """

    def __init__(self, operators, mutation_rate=0.1):
        super().__init__()
        self.operators = operators
        self.mutation_rate = mutation_rate

    def _do(self, x):
        """
        Applies mutation to each individual.

        Args:
            x (np.ndarray): population to mutate.

        Returns:
            np.ndarray: mutated population.
        """""
        for i in range(len(x)):
            x[i] = self.operators.mutation(x[i])
        return x


# =============================================================================
# Pymoo PROBLEM
# =============================================================================

class ReschedulingProblem(ElementwiseProblem):
    """
    Pymoo problem formulation for job shop rescheduling. Each solution
    represents machine-selection and operation-sequence variables.
    """

    def __init__(self, decoder, nr_op, nr_os, max_job_id, max_ms_value):
        """
        Initializes the problem.

        Args:
            decoder (Decoder): decoder that maps arrays into "schedules".
            nr_op (int): number of machine-selection variables.
            nr_os (int): number of operation-sequence variables.
            max_job_id (int): Maximum job ID for OS part bounds.
            max_ms_value (int): Maximum value for MS part (0-based index).
        """
        self.decoder = decoder
        self.nr_op = nr_op
        self.nr_os = nr_os
        n_var = nr_op + nr_os

        # Bounds definition
        xl = np.zeros(n_var, dtype=int)
        xu = np.concatenate([
            np.array([max_ms_value] * nr_op),  # MS: 0-max_ms_value
            np.array([max_job_id] * nr_os)      # OS: 0-max_job_id
        ])

        super().__init__(
            n_var=n_var,
            n_obj=3,  # 3 objectives
            n_constr=1,
            xl=xl,
            xu=xu,
            vtype=int
        )


    def _evaluate(self, x, out):
        """
        Evaluates a candidate solution using the decoder.

        Args:
            X (np.ndarray): population of dimension (N, n_var)
            out (dict): output dictionary where 'F' will be stored.

        Returns:
            None: results stored in `out["F"]`.
        """
        try:
            objectives, g_no_early = self.decoder.decode_with_no_early_constraint(x)
            out["F"] = objectives
            out["G"] = np.array([g_no_early])  # ✅ shape (1,)
        except Exception as e:
            print(f"Decoding error: {e}")
            out["F"] = np.array([1e6, 1e6, 1e6])  # Penalty
            out["G"] = np.array([1e6])


# =============================================================================
# Main function
# =============================================================================

def run_nsga2_with_my_operators(
        jobshop: JobShop,
        broken_machine: int,
        current_time: int,
        pop_size=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        termination_method = "improvement"
):
    """
    Runs NSGA-II with customized sampling, crossover, and mutation for
    job shop rescheduling.

    Args:
        jobshop (JobShop): Current job shop environment.
        broken_machine (int): ID of the machine that has failed.
        current_time (int): Rescheduling start time.
        pop_size (int, optional): Size of the population.
        crossover_rate (float, optional): Probability of crossover.
        mutation_rate (float, optional): Probability of mutation.
        termination_method (str, optional): Termination criterion method.

    Returns:
        tuple: (result, decoder, operators)
        - result: Pymoo optimization output.
        - decoder (Decoder): Decoder used to evaluate solutions.
        - operators (GeneticOperators): Custom genetic operators.
    """
    # Initialize components
    schedule_manager = ScheduleManager(jobshop)
    initializer = RandomInitializer(jobshop, current_time, broken_machine)
    operations_toprocess, operation_times = initializer.extract_sets()
    decoder = Decoder(jobshop, broken_machine, current_time,
                      initializer, schedule_manager)
    max_job_id = jobshop.nr_of_jobs

    # Prepare custom operators
    my_operators = GeneticOperators(
        jobshop=jobshop,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        operations_toprocess=operations_toprocess,
        operation_times=operation_times,
        broken_machine=broken_machine
    )

    # Create pymoo problem
    nr_op = len(operations_toprocess)
    nr_os = nr_op
    max_ms_value = find_max_ms_needed(my_operators)
    problem = ReschedulingProblem(
        decoder, 
        nr_op, 
        nr_os, 
        max_job_id, 
        max_ms_value
    )

    # Configures NSGA-II with custom operators
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MySampling(initializer),  # Initializer
        crossover=MyCrossover(my_operators, crossover_rate),  # Crossover
        mutation=MyMutation(my_operators, mutation_rate),  # Mutation
        eliminate_duplicates=True
    )

    # Initialize termination based on the parameter of the function
    if termination_method == "improvement":
        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8,              # Design space tolerance - stops when decision variables change less than this value (absolute)
            cvtol=1e-6,             # Constraint violation tolerance - stops when constraint violations are below this value (absolute)
            ftol=0.0005,            # Objective space tolerance - stops when objective values change less than this value
            period=30,              # Number of generations to consider in sliding window for tolerance calculations
            n_max_gen=1000,         # Maximum number of generations before forced termination
            n_max_evals=100000      # Maximum number of function evaluations before forced termination
        )
    elif termination_method == "n_gen":
        termination = termination = get_termination("n_gen", 500)
    else:
        termination = get_termination("n_gen", 100)  # Default

    print("Starting NSGA-II.")
    print(f"Population: {pop_size}")
    print(f"Crossover rate: {crossover_rate}, Mutation rate: {mutation_rate}")

    # Optimization
    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=True,
        save_history=False,
        return_least_infeasible=False,
        eliminate_duplicates=True
    )

    # Extracts only the Pareto Front from the optima solutions
    print("\n" + "=" * 60)
    print("PARETO FRONT DEFINITION")
    print("=" * 60)

    if hasattr(res, 'opt'):
        if hasattr(res.opt, 'get'):
            # Extracts chromosomes and objectives from the Pareto Front
            pareto_solutions = res.opt.get("X").copy()
            pareto_front = res.opt.get("F").copy()

            print(f"Pareto front extracted from res.opt:")
            print(f"  Solutions number: {len(pareto_solutions)}")
            print(f"  Chromosome dimension: {pareto_solutions.shape}")
            print(f"  Objectives dimension: {pareto_front.shape}")

            # Verify they are non-dominated
            nds = NonDominatedSorting()
            test_fronts = nds.do(pareto_front)
            if len(test_fronts) == 1 and len(test_fronts[0]) == len(pareto_front):
                print("  All the solutions are non dominated (valid front)")
            else:
                print(f"  Attention: {len(test_fronts[0])} of {len(pareto_front)} are not non dominated (not valid)")

            res.X = pareto_solutions
            res.F = pareto_front
        else:
            print("res.opt does not have a get() method, using res.X and res.F")
    else:
        print("res does not the attribute 'opt'")

    print("=" * 60 + "\n")

    return res, decoder, my_operators


# =============================================================================
# Utilities
# =============================================================================

def test_initial_population(jobshop: JobShop, machine_id: int, broken_time: int, pop_size: int):
    """
    Tests the initial population by analyzing Pareto front relationships.

    Args:
        jobshop (JobShop): Job shop environment.
        machine_id (int): ID of the broken machine.
        broken_time (int): Time when the machine broke.
        pop_size (int): Size of the population to test.

    Returns:
        tuple: (X, F, fronts)
            - X: Population decision vectors
            - F: Objective values
            - fronts: Non-dominated sorting frontS
    """
    print("=" * 70)
    print("INITIAL POPULATION ANALYSIS")
    print("=" * 70)

    schedule = ScheduleManager(jobshop)
    initializer = RandomInitializer(jobshop, broken_time, machine_id)
    decoder = Decoder(jobshop, machine_id, broken_time, initializer, schedule)

    my_sampling = MySampling(initializer)
    X = my_sampling._do(None, pop_size)

    F = []
    for i in range(pop_size):
        individual = X[i]
        objectives = decoder.decode(individual)
        F.append(objectives)
        if i < 5:
            print(f"  Sol {i}: {objectives}")

    F = np.array(F)

    print(f"\nObjective ranges:")
    print(f"Time Differences:     {F[:, 0].min():.1f} - {F[:, 0].max():.1f}")
    print(f"Assignment Differences: {F[:, 1].min():.1f} - {F[:, 1].max():.1f}")
    print(f"Makespan:             {F[:, 2].min():.1f} - {F[:, 2].max():.1f}")

    nds = NonDominatedSorting()
    fronts = nds.do(F)

    print(f"Results:")
    print(f"  Time: {F[:, 0].min():.1f}-{F[:, 0].max():.1f}")
    print(f"  Assignment Differences: {F[:, 1].min():.1f}-{F[:, 1].max():.1f}")
    print(f"  Makespan: {F[:, 2].min():.1f}-{F[:, 2].max():.1f}")
    print(f"  Fronts: {len(fronts)} (Pareto front: {len(fronts[0])} solutions)")

    # Check if all in first front
    if len(fronts[0]) == pop_size:
        print(f"  ⚠️  ALL solutions non-dominated!")

    return X, F, fronts


def assess_pareto_front(X: np.ndarray, F: np.ndarray, jobshop: JobShop, broken_machine_id: int, disruption_time: int, obj1_lb = None):
    """
    Evaluates Pareto solutions by solving MILP models with fixed objectives.

    Args:
        X (np.ndarray): Decision variables of Pareto solutions.
        F (np.ndarray): Objective values of Pareto solutions.
        jobshop (JobShop): Job shop environment.
        broken_machine_id (int): Failed machine ID.
        disruption_time (int): Time of disruption.
        obj1_lb: Lower bound for the first objective 

    Returns:
        tuple:
            - optimal_pareto_front (np.ndarray): MILP-refined Pareto front.
            - optimal_filtered_front (np.ndarray): Non-dominated subset.
            - optimal_points_found (list): Detailed solution data.
    """
    print("=============== PARETO FRONT ASSESSMENT ===============")
    # Debug
    print(f"Size of X: {X.shape}, Size of F: {F.shape}")

    initializer = RandomInitializer(jobshop, disruption_time, broken_machine_id)
    schedule_manager = ScheduleManager(jobshop)

    unique_points_dict = {}
    unique_indices = []

    for i in range(F.shape[0]):
        point_key = tuple(F[i])

        if point_key not in unique_points_dict:
            unique_points_dict[point_key] = i
            unique_indices.append(i)

    print(f"Found {F.shape[0]} total points, {len(unique_indices)} unique points")

    unique_F = F[unique_indices]
    unique_X = X[unique_indices]

    # Array with three values per row, where each row represents a point of the optimal Pareto Front
    optimal_pareto_front = []
    distances = []

    # Structure to contain the Pareto front and the optimal values
    optimal_points_found = []

    decoder = Decoder(jobshop, broken_machine_id, disruption_time, initializer, schedule_manager)

    for idx, i in enumerate(unique_indices):
        obj_quadr_delay = unique_F[idx][0]   # 1st objective
        obj_mach_assignm = unique_F[idx][1]  # 2nd objective
        obj_cmax = unique_F[idx][2]          # 3rd objective
        point = np.array([obj_quadr_delay, obj_mach_assignm, obj_cmax])
        # Debug objective values
        print("==========================================================")
        print(f"UNIQUE INDIVIDUAL {idx + 1}/{len(unique_indices)} (Original index: {i}): {unique_X[idx]}")
        print(f"Value of the first objective: {obj_quadr_delay}.")
        print(f"Value of the second objective: {obj_mach_assignm}.")
        print(f"Value of the third objective: {obj_cmax}.")

        individual = unique_X[idx]

        reschedule1 = FJS_reschedule(jobshop, broken_machine_id, disruption_time)
        reschedule1.extract_info()
        if obj1_lb is not None:
            reschedule1.create_model(obj2_value=obj_mach_assignm,
                obj3_value=obj_cmax, obj1_lb=obj1_lb)
        else:
            reschedule1.create_model(obj2_value=obj_mach_assignm,
                                     obj3_value=obj_cmax)
        new_td, _, _ = reschedule1.run_model(1200)
        reschedule1.print_results()
        new_point1 = np.array([new_td, obj_mach_assignm, obj_cmax])
        optimal_pareto_front.append(new_point1)
        distance1 = np.linalg.norm(point - new_point1)

        reschedule2 = FJS_reschedule(jobshop, broken_machine_id, disruption_time)
        reschedule2.extract_info()
        reschedule2.create_model(obj1_value=obj_quadr_delay, obj3_value=obj_cmax)
        _, new_ad, _ = reschedule2.run_model(1200)
        reschedule2.print_results()
        new_point2 = np.array([obj_quadr_delay, new_ad, obj_cmax])
        optimal_pareto_front.append(new_point2)
        distance2 = np.linalg.norm(point - new_point2)

        reschedule3 = FJS_reschedule(jobshop, broken_machine_id, disruption_time)
        reschedule3.extract_info()
        reschedule3.create_model(obj1_value=obj_quadr_delay, obj2_value=obj_mach_assignm)
        _, _, new_cmax = reschedule3.run_model(1800)
        reschedule3.print_results()
        new_point3 = np.array([obj_quadr_delay, obj_mach_assignm, new_cmax])
        optimal_pareto_front.append(new_point3)
        distance3 = np.linalg.norm(point - new_point3)

        optimal_points_found.append({
            "Time differences": obj_quadr_delay,
            "Assignment differences": obj_mach_assignm,
            "Makespan": obj_cmax,
            "Optimal Time Differences": new_td,
            "Optimal Assignment Differences": new_ad,
            "Optimal Makespan": new_cmax
        })

        distance = np.array([distance1, distance2, distance3])

        distances.append(distance)

    optimal_pareto_front = np.array(optimal_pareto_front)

    nds = NonDominatedSorting()
    fronts= nds.do(optimal_pareto_front)

    # Extract only the Pareto optimal solutions (first front)
    if len(fronts) > 0:
        pareto_indices = fronts[0]  # Indexes of the non-dominated solutions
        optimal_front_filtered = optimal_pareto_front[pareto_indices]
    else:
        optimal_front_filtered = optimal_pareto_front

    return optimal_pareto_front, optimal_front_filtered, optimal_points_found



# =============================================================================
# TEST SCRIPT
# =============================================================================

if __name__ == "__main__":
    """
    Main execution script for NSGA-II rescheduling optimization.
    """
    file_path = os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl")
    jobshop = load_environment(file_path)
    broken_machine = 0
    current_time = 1

    # Execute NSGA-II with customized operators
    res, decoder, operators = run_nsga2_with_my_operators(
        jobshop=jobshop,
        broken_machine=broken_machine,
        current_time=current_time,
        pop_size=200,
        crossover_rate=0.8,
        mutation_rate=0.1,
        termination_method="improvement"
    )

    # Population analysis
    print("\n=== POPULATION ANALYSIS ===")
    print(f"Total population size: {len(res.X)}")
    print(f"Pareto front size: {len(res.F)}")

    pareto_front = res.F
    pareto_solutions = res.X

    optimal_front, optimal_filtered_front= assess_pareto_front(
        pareto_solutions,
        pareto_front,
        jobshop,
        broken_machine,
        current_time
    )

    initializer = RandomInitializer(jobshop, current_time, broken_machine)

    # Select best solution
    best_solution, best_objectives = select_best_solution(
        pareto_front, pareto_solutions, method="min_time_diff"
    )

    # Update environment
    updated_jobshop = update_environment_with_solution(
        jobshop, best_solution, decoder, initializer, current_time
    )

    # Save everything
    save_results(res, decoder, os.path.join(RESULTS_DIR, "NSGA2", "EX_main"))

    plot(updated_jobshop)
    save_path = os.path.join(RESULTS_DIR, "NSGA2", "EX_main", "schedule_gantt_1")
    plt.savefig(save_path, dpi=300)
    plt.show()

    # Statistics
    print_pareto_statistics(pareto_front)

    # Visualization (3D)
    plot_pareto_front_3d_matplotlib(pareto_front)
    plot_pareto_front_3d_matplotlib(pareto_front, assess=True, optimal_pareto_front=optimal_filtered_front)

    print("NSGA-II completed successfully!")