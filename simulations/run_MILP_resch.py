"""
Original script for rescheduling experiments - uses only generic logging functions.
Uses new Incumbent class for multi-objective tracking
"""
import os.path
import matplotlib
from absl.logging import LOG_DIR

#matplotlib.use("Agg")  # backend non-GUI
import matplotlib.pyplot as plt
import sys
import logging
import numpy as np
from pathlib import Path
logging.getLogger('gurobipy').setLevel(logging.WARNING)

# /home/bazzo/Scrivania/fjsp_rescheduling/simulations
SCRIPT_DIR = Path(__file__).resolve().parent
# /home/bazzo/Scrivania/fjsp_rescheduling/
PROJECT_ROOT = SCRIPT_DIR.parent

LOG_DIR = SCRIPT_DIR / "logs"
RESULTS_DIR = PROJECT_ROOT / "results" / "MILP"
RESULTS_SCH_DIR = PROJECT_ROOT / "results_scheduling"

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"LOG_DIR: {LOG_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"RESULTS_SCH_DIR: {RESULTS_SCH_DIR}")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LOG_DIR))
sys.path.insert(0, str(RESULTS_SCH_DIR))
sys.path.insert(0, str(RESULTS_DIR))


from simulations.experiment_logging import (
    setup_logging_directory, setup_experiment_logging, setup_gurobi_logging,
    close_experiment_logging, log_experiment_start, log_experiment_end,
    load_environment, log_environment_info
)

from simulations.break_machine import break_machine

# Import the model for this script
from MILP.model_MILP_reschedule import FJS_reschedule

data_break = {
    'EX1': {
        'broken_machine_id': 1,
        'disruption_time': 4
    },
    'EX2': {
        'broken_machine_id': 0,
        'disruption_time': 1
    },
    'EX3': {
        'broken_machine_id': 5,
        'disruption_time': 75
    }
}



def run_milp_experiment(exp_num, file_path, time_limit=1200, n_sim=1):
    """
    Run an experiment using the specific MILP model.
    """
    
    # Setup log directory
    log_dir = setup_logging_directory(LOG_DIR, log_dir_name="")

    log_dir_ex = LOG_DIR / f"EX{exp_num}"

    os.makedirs(log_dir_ex, exist_ok=True)
    
    # Setup logging
    logger = setup_experiment_logging(log_dir, exp_num)

    # Set up Gurobi Log File 
    gurobi_log_file = setup_gurobi_logging(log_dir, exp_num, log_to_console=True)

    # Load environment
    jobShopEnv = load_environment(file_path, logger)

    broken_machine_id, disruption_time = break_machine(jobShopEnv, avoid_unique=True, seed=exp_num)

    data_break.update({
        f"EX{exp_num}":
            {'broken_machine_id': broken_machine_id,
             'disruption_time': disruption_time}
    })

    try:
        # Log experiment start
        params = {
            'file_path': file_path,
            'broken_machine_id': broken_machine_id,
            'disruption_time': disruption_time,
            'time_limit': time_limit
        }

        log_experiment_start(logger, exp_num, params)

        log_environment_info(logger, jobShopEnv)
        
        # Configure the MILP model
        logger.info("Starting rescheduling with MILP model...")

        if n_sim == 1:
            reschedule = FJS_reschedule(jobShopEnv, broken_machine_id, disruption_time)

            # Reschedule with the specific model
            reschedule.extract_info()

            reschedule.create_model()

            reschedule.model.setParam('LogFile', str(gurobi_log_file))
            reschedule.model.setParam('OutputFlag', 1)  # 1 = print to console, 0 = silent

            # reschedule.debug_relaxation()

            reschedule.run_model(time_limit=time_limit)

            reschedule.update_environment()

            EXP_DIR = RESULTS_DIR / f"EX{exp_num}"
            os.makedirs(EXP_DIR, exist_ok=True)

            reschedule.plot_incumbent(EXP_DIR)

            incumbent = reschedule.incumbent

            reschedule.plot_schedule()
            gantt_path = os.path.join(RESULTS_DIR, f"EX{exp_num}", f"schedule_gantt_ex{exp_num}_resch_nolin.png")
            plt.savefig(gantt_path, dpi=300)
            logger.info(f"Gantt chart saved in: {gantt_path}")

            reschedule.print_results()

            # Log experiment end
            log_experiment_end(logger, exp_num, success=True)

            return reschedule, incumbent, jobShopEnv

        elif n_sim > 1:

            results = {
                'quadratic_delay_sum': [],
                'machine_assignment_sum': [],
                'cmax': []
            }

            for current_phase in range(n_sim):
                reschedule = FJS_reschedule(jobShopEnv, broken_machine_id, disruption_time)

                # Reschedule with the specific model
                reschedule.extract_info()

                reschedule.create_model()

                # reschedule.debug_relaxation()

                reschedule.run_model(time_limit=time_limit)

                quadratic_delay=reschedule.results['quadratic_delay']
                assignment_changes = reschedule.results['assignment_changes']
                cmax = reschedule.results['cmax']

                results['quadratic_delay_sum'].append(quadratic_delay)
                results['machine_assignment_sum'].append(assignment_changes)
                results['cmax'].append(cmax)

                print(f"Simulation {current_phase}: ")
                print(f"- Quadratic delay = {quadratic_delay}")
                print(f"- Assignment changes = {assignment_changes}")
                print(f"- Cmax = {cmax}")

            reschedule.update_environment()

            incumbent_path = os.path.join(RESULTS_DIR, f"EX{exp_num}")

            incumbent = reschedule.incumbent

            #reschedule.plot_incumbent(incumbent_path)

            """plt.figure()
            reschedule.plot_schedule()
            gantt_path = os.path.join(RESULTS_DIR, f"EX{exp_num}", f"schedule_gantt_ex{exp_num}_resch_nolin.png")
            plt.savefig(gantt_path, dpi=300)
            plt.close()
            logger.info(f"Gantt chart saved in: {gantt_path}")"""

            mean_quadratic_delay = np.mean(results['quadratic_delay_sum'])
            sd_quadratic_delay = np.std(results['quadratic_delay_sum'])

            mean_machine_assignments = np.mean(results['machine_assignment_sum'])
            sd_machine_assignments = np.std(results['machine_assignment_sum'])

            mean_cmax = np.mean(results['cmax'])
            sd_cmax = np.std(results['cmax'])

            print(f"EX{exp_num}, {n_sim} simulations: ")
            print(f"- Mean Quadratic Delay Sum: {mean_quadratic_delay}")
            print(f"- Standard Deviation of the Quadratic Delay Sum: {sd_quadratic_delay}")
            print(f"- Mean Machine Assignments Sum: {mean_machine_assignments}")
            print(f"- Standard Deviation of the Machine Assignments Sum: {sd_machine_assignments}")
            print(f"- Mean Cmax: {mean_cmax}")
            print(f"- Standard Deviation of the Cmax: {sd_cmax}")

            # reschedule.print_results()

            # Log experiment end
            log_experiment_end(logger, exp_num, success=True)

            return reschedule, incumbent, jobShopEnv


        """reschedule = FJS_reschedule(jobShopEnv, broken_machine_id, disruption_time)

        # Reschedule with the specific model 
        reschedule.extract_info()

        reschedule.create_model()

        reschedule.model.setParam('LogFile', str(gurobi_log_file))
        reschedule.model.setParam('OutputFlag', 1)  # 1 = print to console, 0 = silent

        reschedule.debug_relaxation()

        reschedule.run_model(time_limit=time_limit)

        reschedule.update_environment()

        incumbent_path = os.path.join(RESULTS_DIR, f"EX{exp_num}")

        reschedule.plot_incumbent(incumbent_path)
        
        incumbent = reschedule.incumbent"""
        
        # Log results
        """if incumbent and incumbent.sol_list:
            logger.info(f"Rescheduling completed")
            logger.info(f"Best objective value: {incumbent.sol_list[-1]}")
            logger.info(f"Computation time: {incumbent.times[-1]} seconds")
        else:
            logger.warning("No solution found")"""

        """if incumbent and incumbent.times and incumbent.sol_list:
            plt.figure()
            plt.plot(incumbent.times, incumbent.sol_list)
            plt.title('Evolution of the Objective Function value during optimization', fontsize=12)
            plt.xlabel("Computation Time (seconds)", fontsize=11)
            plt.ylabel("Objective Function Value", fontsize=11)
            objective_path = os.path.join(RESULTS_DIR, f"EX{exp_num}", f"objective_evolution_ex{exp_num}_resch_nolin.png")
            plt.savefig(objective_path, dpi=300)
            plt.close()
            logger.info(f"Objective evolution plot saved in: {objective_path}")"""

        """plt.figure()
        reschedule.plot_schedule()
        gantt_path = os.path.join(RESULTS_DIR, f"EX{exp_num}", f"schedule_gantt_ex{exp_num}_resch_nolin.png")
        plt.savefig(gantt_path, dpi=300)
        plt.close()
        logger.info(f"Gantt chart saved in: {gantt_path}")

        reschedule.print_results()

        # Log experiment end
        log_experiment_end(logger, exp_num, success=True)
        
        return reschedule, incumbent, jobShopEnv"""
        
    except Exception as e:
        logger.error(f"Error during experiment {exp_num}: {str(e)}")
        log_experiment_end(logger, exp_num, success=False, message=str(e))
        raise
    finally:
        close_experiment_logging(logger)


def get_data_break():
    return data_break


if __name__ == "__main__":
    print(f"Working directory: {RESULTS_DIR}")
    
    # Experiment 1
    run_milp_experiment(1, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_1.pkl"), time_limit=3600)
    
    # Experiment 2
    run_milp_experiment(2, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_2.pkl"), time_limit=3600)

    # Experiment 3
    run_milp_experiment(3, os.path.join(RESULTS_SCH_DIR, "schedule_simulation_3.pkl"), time_limit=3600)