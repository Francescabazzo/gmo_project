# Import libraries
import random
from typing import Tuple
from scheduling_environment.jobShop import JobShop


def break_machine(jobshop: JobShop, avoid_unique: object = True, seed: int = 42) -> Tuple[int, int]:
    """
    Breaks randomly a machine from the job shop.

    Args:
        jobshop: job shop environment
        avoid_unique: if True, does not break "unique machines"
        seed: base seed for reproducibility

    Returns:
        Tuple of the id of the broken machine and the moment in which it breaks
    """
    rnd = random.Random(seed)

    unique_machines = jobshop.unique_machine_ids
    all_machines = list(range(jobshop.nr_of_machines))

    if avoid_unique and unique_machines:
        available_machines = [m for m in all_machines if m not in unique_machines]
    else:
        available_machines = all_machines

    if not available_machines:
        raise ValueError("No machines available to break (all might be unique machines)")

    broken_machine = rnd.choice(available_machines)

    current_makespan = jobshop.makespan
    max_time = int(current_makespan // 2)

    breakdown_time = rnd.randint(0, max_time)

    return broken_machine, breakdown_time