import os 
os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""

import sys
sys.modules["tkinter"] = None

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.ioff()

import matplotlib.pyplot as plt

from pathlib import Path



# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_SCH_DIR = PROJECT_ROOT / "results_scheduling"
SIMULATIONS_DIR = PROJECT_ROOT / "simulations"

# Add project paths
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SIMULATIONS_DIR))
sys.path.insert(0, str(RESULTS_SCH_DIR))

# ============================================================
# IMPORTS FROM PROJECT MODULES
# ============================================================

from simulations.run_GA_MILP import run_ga_milp 
from simulations.run_MILP_resch import run_milp_experiment
from simulations.run_NSGA_II import run_nsga2_experiment


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENTS = [
    {
        "exp_num": 1,
        "dataset": "/fjsp/kacem/Kacem1.fjs"
    },
    {
        "exp_num": 2,
        "dataset": "/fjsp/brandimarte/Mk02.fjs"
    },
    {
        "exp_num": 3,
        "dataset": "/fjsp/brandimarte/Mk10.fjs"
    }
]

# Population sizes for NSGA-II
NSGA_POPULATIONS = [20, 50, 100, 200]


# ============================================================
# FULL SIMULATION PIPELINE
# ============================================================

def run_full_pipeline():
    """
    Run the complete simulation pipeline:
    1. Generate baseline schedule using GA + MILP
    2. Perform rescheduling using MILP
    3. Optimize rescheduling using NSGA-II
    """

    print("=" * 80)
    print("STARTING FULL SIMULATION PIPELINE")
    print("=" * 80)

    for exp in EXPERIMENTS:
        exp_num = exp["exp_num"]
        dataset = exp["dataset"]

        print(f"\nRunning experiment {exp_num}")
        print("-" * 60)

        # --------------------------------------------------
        # STEP 1: GA + MILP (baseline schedule generation)
        # --------------------------------------------------
        print("Step 1 - GA + MILP baseline scheduling")

        run_ga_milp(dataset, exp_num)

        schedule_file = RESULTS_SCH_DIR / f"schedule_simulation_{exp_num}.pkl"

        if not schedule_file.exists():
            raise FileNotFoundError(f"Schedule file not found: {schedule_file}")

        # --------------------------------------------------
        # STEP 2: MILP-based rescheduling
        # --------------------------------------------------
        print("Step 2 - MILP rescheduling")

        run_milp_experiment(
            exp_num=exp_num,
            file_path=str(schedule_file),
            time_limit=3600
        )

        # --------------------------------------------------
        # STEP 3: NSGA-II optimization
        # --------------------------------------------------
        print("Step 3 - NSGA-II optimization")

        for pop in NSGA_POPULATIONS:
            print(f"  Running NSGA-II with population size {pop}")

            run_nsga2_experiment(
                exp_num=exp_num,
                n_pop=pop,
                file_path=str(schedule_file)
            )
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    print("\nFULL SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_full_pipeline()
