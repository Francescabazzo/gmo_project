import numpy as np
from typing import Dict, List, Tuple
from scheduling_environment.jobShop import JobShop


class RandomInitializer:
    """
    Class that creates a random initial population for the GAs NSGA-II and NSGA-III using the MSOS representation.
    """


    def __init__(self, jobshop: JobShop, current_time: int, broken_machine: int):
        self.env = jobshop
        self.time = current_time
        self.broken_machine = broken_machine

        self.operations_toprocess = []
        self.operations_ongoing = []
        self.operations_completed = []
        self.operations_final = []
        self.operation_times = {}

        self._categorize_operations()
        self.nr_op = len(self.operations_toprocess)


    def _categorize_operations(self):
        """
        Categorizes operations based on their status and the current time (moment in which the machine breaks).
        """

        for job in self.env.jobs:
            operations = [operation.operation_id for operation in job.operations]
            self.operations_final.append(operations[-1])

        for operation in self.env.operations:
            # Operation is scheduled to start in the future
            if operation.scheduled_start_time >= self.time:
                self.operations_toprocess.append(operation.operation_id)
            # Operation is currently ongoing
            elif operation.scheduled_end_time > self.time > operation.scheduled_start_time:
                if operation.scheduled_machine == self.broken_machine:
                    # Operation on ongoing machine needs to be rescheduled
                    self.operations_toprocess.append(operation.operation_id)
                else:
                    # Operation on working machines continues normally
                    self.operations_ongoing.append(operation.operation_id)
            else:
                self.operations_completed.append(operation.operation_id)
            # Process operation times for all optional machines
            for k in operation.optional_machines_id:
                if (
                        operation.scheduled_end_time > self.time > operation.scheduled_start_time
                        and k == self.broken_machine and k == operation.scheduled_machine
                ):
                    # Handle operation ongoing on broken machine: calculate remaining processing time
                    operation.set_broken(self.time)
                    broken_info = operation.is_broken_info
                    percent_to_do = broken_info['percentage_to_do']
                    self.operation_times[(operation.operation_id, k)] = percent_to_do * operation.processing_times[k]
                else:
                    # "Normal" processing time
                    self.operation_times[(operation.operation_id, k)] = operation.processing_times[k]


    def create_ms(self) -> list:
        """
        Creates a random MS part: for each operation, it contains the index of the machine where the operation will be
        processed.

        Returns:
            list: For each operation, contains the index of the machine where the operation will be processed.
        """
        ms = []
        for operation_id in self.operations_toprocess:
            operation = self.env.get_operation(operation_id)
            # Exclude broken machine from available machines
            valid_machines = [m for m in operation.optional_machines_id if m != self.broken_machine]
            if not valid_machines:
                raise ValueError(f"Operation number {operation.operation_id} has no valid machines.")
            # Randomly select a machine from available options
            machine_id = np.random.randint(0, len(valid_machines))
            ms.append(machine_id)
        return ms


    def create_os(self) -> list:
        """
        Creates a random OS part: it contains the job IDs repeated for the number of operations the job contains.

        Returns:
            list: Contains job IDs repeated for the number of operations each job has, then shuffled to create a
            random sequence.
        """
        os = []
        for job in self.env.jobs:
            # Count how many operations of this specific job needs to be rescheduled
            num_operations_to_reschedule = sum(
                1 for op in job.operations
                if op.operation_id in self.operations_toprocess
            )
            # Add job ID for each operation that needs rescheduling
            os.extend([job.job_id]*num_operations_to_reschedule)
        # Shuffle to create random sequence
        np.random.shuffle(os)
        return os


    def create_individual(self) -> np.ndarray:
        """
        Creates an individual by combining MS and OS.

        Returns:
            np.ndarray: Complete chromosome representation [MS part | OS part]
        """
        ms = self.create_ms()
        os = self.create_os()

        return np.array(ms + os)


    def create_population(self, size: int) -> np.ndarray:
        """
        Creates an initial random population of a specified size.

        Args:
            size (int): Number of individuals in the population

        Returns:
            np.ndarray: Array of individual chromosomes
        """
        population = []
        for i in range(size):
            individual = self.create_individual()
            population.append(individual)
        return np.array(population)


    def extract_sets(self) -> Tuple[List[int], Dict]:
        """
        Extracts the sets of operations to process and their processing times.

        Returns:
            Tuple[List[int], Dict]:
                - List of operation IDs to be processed
                - Dictionary mapping (operation_id, machine_id) to processing time
        """
        return self.operations_toprocess, self.operation_times


    def get_operations_ongoing(self) -> List[int]:
        """
        Extracts the set of operations that are currently ongoing in the machines.

        Returns:
            List[int]: list of operation IDs of ongoing operations
        """
        return self.operations_ongoing


    def get_operations_toprocess(self) -> List[int]:
        """
        Extracts the set of operations to process.

        Returns:
            List[int]: list of operation IDs of operations to be processed
        """
        return self.operations_toprocess


    def get_operations_completed(self) -> List[int]:
        """
        Extracts the set of operations that were completed.

        Returns:
            List[int]: list of operation IDs of operations that were completed
        """
        return self.operations_completed


    def get_operations_final(self):
        return self.operations_final
