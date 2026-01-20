import numpy as np
import random

from scheduling_environment.jobShop import JobShop
from typing import Dict, List, Tuple


class GeneticOperators:
    """
    Class implementing genetic operators (crossover and mutation) for the genetic algorithm.
    Handles both Machine Selection (MS) and Operation Sequence (OS) parts of the chromosome.
    """


    def __init__(self, jobshop: JobShop, mutation_rate: float, crossover_rate: float, operations_toprocess: list,
                 operation_times: Dict, broken_machine: int):
        """
        Initialize genetic operators with problem instance and parameters.

        Args:
            jobshop: JobShop environment instance
            mutation_rate: Probability of mutation for each individual
            crossover_rate: Probability of crossover for each pair of parents
            operations_toprocess: List of operation IDs that need to be rescheduled
            operation_times: Dictionary mapping (operation_id, machine_id) to processing time
            broken_machine: ID of the broken machine that cannot be used
        """
        self.env = jobshop
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.operations_toprocess = operations_toprocess
        self.broken_machine = broken_machine
        self.operation_times = operation_times


    def crossover(
            self, 
            parent1: np.ndarray, 
            parent2: np.ndarray, 
            apply_prob=True
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs crossover on two parents to generate two children.
        Applies different crossover methods for MS and OS parts.

        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome

        Returns:
            Tuple of two child chromosomes
        """
        p1_copy = parent1.copy()
        p2_copy = parent2.copy()

        # Skip crossover if random probability exceeds crossover rate
        if apply_prob and random.random() >= self.crossover_rate:
            return p1_copy, p2_copy

        n = len(self.operations_toprocess)
        # Split chromosomes into MS (Machine Selection) and OS (Operation Sequence) parts
        ms1 = p1_copy[ :n]
        ms2 = p2_copy[ :n]
        os1 = p1_copy[n: ]
        os2 = p2_copy[n: ]

        # Randomly choose between two MS crossover methods
        if random.random() > 0.5:
            # Two-points crossover
            child_ms1, child_ms2 = self._crossover_ms_two_p(ms1, ms2)
        else:
            # Uniform crossover
            child_ms1, child_ms2 = self._crossover_ms_unif(ms1, ms2)

        # Apply POX crossover for OS part
        child_os1, child_os2 = self._crossover_os_pox(os1, os2)

        # Combine MS and OS parts to form complete children
        return np.concatenate([child_ms1, child_os1]), np.concatenate([child_ms2, child_os2])


    def mutation(self, individual, apply_prob=True):
        """
        Applies mutation to an individual chromosome.
        Mutates both MS and OS parts separately.

        Args:
            individual: Chromosome to mutate

        Returns:
            Mutated chromosome
        """
        individual_copy = individual.copy()

        # Skip mutation if random probability exceeds mutation rate
        if apply_prob and random.random() >= self.mutation_rate:
            return individual_copy

        n = len(self.operations_toprocess)
        # Split chromosome into MS and OS parts
        ms = individual_copy[:n]
        os = individual_copy[n:]

        # Apply mutation to both parts
        child_ms = self._mutate_ms(ms)
        child_os = self._mutate_os(os)

        # Combine mutated parts
        return np.concatenate([child_ms, child_os])


    def debug_real_assignments(self, decoder, solutions):
        print("=== REAL MACHINE ASSIGNMENTS ===")

        for i, sol in enumerate(solutions[:5]):
            n = len(decoder.operations_toprocess)
            ms = sol[:n]

            print(f"Solution {i}:")
            op_idx = 0
            assignment_diff = 0

            for job in self.env.jobs:
                for operation in job.operations:
                    if operation.operation_id in decoder.operations_toprocess:
                        assigned_idx = ms[op_idx]
                        valid_machines = [m for m in operation.optional_machines_id
                                          if m != self.broken_machine]

                        if valid_machines and assigned_idx < len(valid_machines):
                            actual_machine = valid_machines[assigned_idx]
                            original_machine = operation.scheduled_machine
                            print(f"  Op {operation.operation_id}: machine {actual_machine} (idx={assigned_idx})")

                        op_idx += 1


    def debug_broken_machine_impact(self):
        print("=== BROKEN MACHINE IMPACT ===")
        total_operations = 0
        constrained_operations = 0

        for job in self.env.jobs:
            for operation in job.operations:
                if operation.operation_id in self.operations_toprocess:
                    total_operations += 1
                    valid_machines = [m for m in operation.optional_machines_id
                                      if m != self.broken_machine]

                    original_options = len(operation.optional_machines_id)
                    current_options = len(valid_machines)

                    if current_options < original_options:
                        constrained_operations += 1
                        print(f"Op {operation.operation_id}: {original_options} -> {current_options} options")

        print(f"Total operations: {total_operations}")
        print(f"Operations constrained by broken machine: {constrained_operations}")


    def debug_machine_options(self):
        option_counts = {}

        for job in self.env.jobs:
            for operation in job.operations:
                if operation.operation_id in self.operations_toprocess:
                    valid_machines = [m for m in operation.optional_machines_id
                                      if m != self.broken_machine]
                    count = len(valid_machines)
                    option_counts[count] = option_counts.get(count, 0) + 1

        print("=== MACHINE OPTIONS DISTRIBUTION ===")
        for count, freq in sorted(option_counts.items()):
            print(f"Operations with {count} machine options: {freq}")


    def _crossover_ms_two_p(self, ms1, ms2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs two-point crossover on Machine Selection parts.

        Args:
            ms1: MS part of first parent
            ms2: MS part of second parent

        Returns:
            Tuple of two children MS parts
        """
        n = len(ms1)
        # Select two random crossover points
        l1, l2 = random.sample(range(0, n), 2)
        cx1, cx2 = min(l1, l2), max(l1, l2)

        # Create children by swapping segments between crossover points
        new_ms1 = np.concatenate([
            ms1[: cx1],
            ms2[cx1:cx2],
            ms1[cx2:]
        ])

        new_ms2 = np.concatenate([
            ms2[: cx1],
            ms1[cx1:cx2],
            ms2[cx2:]
        ])

        return new_ms1, new_ms2


    def _crossover_ms_unif(self, ms1, ms2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs uniform crossover on Machine Selection parts.

        Args:
            ms1: MS part of first parent
            ms2: MS part of second parent

        Returns:
            Tuple of two children MS parts
        """
        ms1 = np.asarray(ms1)
        ms2 = np.asarray(ms2)

        # Create random mask to select genes from each parent
        mask = np.random.randint(0, 2, size=len(ms1))

        # Apply uniform crossover using the mask
        new_ms1 = np.where(mask, ms2, ms1)
        new_ms2 = np.where(mask, ms1, ms2)

        return new_ms1, new_ms2


    def _crossover_os_pox(self, os1, os2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs Precedence Preservative Crossover (POX) on Operation Sequence parts.
        Preserves relative order of operations from parents.

        Args:
            os1: OS part of first parent
            os2: OS part of second parent

        Returns:
            Tuple of two children OS parts
        """
        # Step 1: Get all unique jobs
        all_jobs = list(set(os1))

        # Step 2: Create two random job subsets
        k = random.randint(1, len(all_jobs) - 1)
        js1 = set(random.sample(all_jobs, k=k))
        js2 = set(all_jobs) - js1

        # Initialize children with placeholder values
        child1 = np.full_like(os1, -1)
        child2 = np.full_like(os2, -1)

        # Step 3: Copy jobs from parents based on job sets
        for i in range(len(os1)):
            if os1[i] in js1:
                child1[i] = os1[i]
            if os2[i] in js2:
                child2[i] = os2[i]

        # Step 4: Fill empty positions with remaining jobs in original order
        # For child1: fill with jobs from os2 not in js1
        remaining_os2 = [job for job in os2 if job not in js1]
        idx2 = 0
        for i in range(len(child1)):
            if child1[i] == -1 and idx2 < len(remaining_os2):
                child1[i] = remaining_os2[idx2]
                idx2 += 1
        # For child2: fill with jobs from os1 not in js2
        remaining_os1 = [job for job in os1 if job not in js2]
        idx1 = 0
        for i in range(len(child2)):
            if child2[i] == -1 and idx1 < len(remaining_os1):
                child2[i] = remaining_os1[idx1]
                idx1 += 1

        return child1, child2


    def _mutate_ms(self, ms: np.ndarray) -> np.ndarray:
        """
        Mutates Machine Selection part using shortest processing time strategy.
        For selected operations, changes machine assignment to the one with minimum processing time.

        Args:
            ms: Machine Selection part to mutate

        Returns:
            Mutated MS part
        """
        mutated_ms = ms.copy()
        op_idx = 0

        # Iterate through all operations in all jobs
        for job in self.env.jobs:
            for operation in job.operations:
                if operation.operation_id in self.operations_toprocess:
                    # Get valid machines (excluding broken machine)
                    valid_machines = [m for m in operation.optional_machines_id
                                        if m != self.broken_machine]
                    if valid_machines:
                        # Find the machine with minimum processing time
                        best_machine_idx = self._find_shortest_processing_machine(
                            operation, valid_machines
                        )
                        random_machine_index = random.randint(0, len(valid_machines) - 1)
                        if random.random() > 0.5: 
                            mutated_ms[op_idx] = random_machine_index
                        else: 
                            mutated_ms[op_idx] = best_machine_idx

                    op_idx += 1

        return mutated_ms


    def _find_shortest_processing_machine(self, operation, valid_machines: List[int]) -> int:
        """
        Finds the index of the machine with minimum processing time for a given operation.

        Args:
            operation: Operation to process
            valid_machines: List of available machine IDs

        Returns:
            Index of the machine with shortest processing time in the valid_machines list
        """
        min_time = float('inf')
        best_machine_idx = 0

        # Evaluate all valid machines to find the fastest one
        for idx, machine_id in enumerate(valid_machines):
            processing_time = operation.processing_times[machine_id]
            if processing_time < min_time:
                min_time = processing_time
                best_machine_idx = idx

        return best_machine_idx


    def _mutate_os(self, os: np.ndarray) -> np.ndarray:
        """
        Mutates Operation Sequence part using swap mutation.
        Randomly swaps operations in the sequence.

        Args:
            os: Operation Sequence part to mutate

        Returns:
            Mutated OS part
        """
        mutated_os = os.copy()

        # Iterate through all positions in the sequence
        for i in range(len(mutated_os)):
            # Select a random position to swap with
            j = random.randint(0, len(mutated_os) - 1)
            # Perform swap
            mutated_os[i], mutated_os[j] = mutated_os[j], mutated_os[i]

        return mutated_os