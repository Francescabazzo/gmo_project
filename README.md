# Flexible Job Shop Scheduling – Rescheduling Framework

## Overview

This project implements a complete optimization framework for the **Flexible Job Shop Scheduling Problem (FJSP)** with random machine breakdown with:

- Genetic Algorithm (GA) and MILP optimization (Gurobi) to generate a baseline schedule 
- Rescheduling after machine breakdown with MILP model (Gurobi)
- Rescheduling after machine breakdown using NSGA-II
- Pareto front evaluation
- Gantt chart visualization

This project was developed for the final exam of the course of Global and Multi-objective Optimization.

---

## Project Structure

```
gmo_project/
│
└── data /
│ ├── data_parsers /
│ ├── fjsp / 
│ ├── dataset_generator.py 
│
├── NSGA2/
│ ├── nsga_ii.py

│ ├── src/
│ │ ├── decode.py
│ │ ├── operators.py
│ │ ├── random_initialization.py
│ │ ├── schedule_manager.py
│ └── nsga_ii.py
│ └── plot.py 
│ └── utils.py
│
├── MILP/
│ ├── incumbent.py 
│ ├── incumbent_mo.py 
│ ├── model_MILP.py
│ ├── model_MILP_reschedule.py
│ └── model_MILP_assessment.py
│
├── results/

├── results_scheduling/
|
├── simulations/
│ ├── logs /
│ ├── break_machine.py
│ ├── experiment_logging.py
│ ├── run_GA_MILP.py
│ ├── run_MILP_resch.py
│ ├── run_NSGA_II.py
│ └── run_full_simulation.py
│
├── scheduling_environment/
│ └── job.py
│ └── jobShop.py
│ └── machine.py
│ └── operation.py
│ └── simulationEnv
│
├── visualization/
│ └── gantt_chart.py
|
└── README.md
└── requirements.txt 
```


## Requirements 

### Python 
```
Python >= 3.9
```

### Libraries 

Install everythig with: 

```
pip install -r requirements.txt
```

## Running the full experiment 

```
python simulations/run_full_simulation.py
```

This will automatically compute the following actions on three benchmark datasets:
1. Generate an initial schedule (GA + MILP)
2. Apply machine breakdown
3. Run MILP rescheduling
4. Run NSGA-II optimization
5. Save results, plots, and logs


## Bibliography

The folders solution_methods, visualization, and scheduling environment were taken from

Reijnen, R., van Straaten, K., Bukhsh, Z., & Zhang, Y. (2023). **Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods**. arXiv preprint arXiv:2308.12794. https://doi.org/10.48550/arXiv.2308.12794