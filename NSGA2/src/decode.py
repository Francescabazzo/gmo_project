import numpy as np
from typing import Dict, List, Tuple
from pymoo.indicators.gd import GD

from NSGA2.src.random_initialization import RandomInitializer
from scheduling_environment.jobShop import JobShop
from NSGA2.src.schedule_manager import ScheduleManager


class Decoder:
    """
    Decoder class that converts chromosome representation (MS-OS) into a feasible schedule
    and evaluates the solution by calculating objective functions.
    """


    def __init__(self, jobshop: JobShop, broken_machine: int, current_time: int,
                 initializer: RandomInitializer, schedule_manager: ScheduleManager):
        """
        Initialize the decoder with problem instance and parameters.

        Args:
            jobshop: JobShop environment instance
            broken_machine: ID of the broken machine that cannot be used
            current_time: Current simulation time when rescheduling is triggered
            operations_toprocess: List of operation IDs that need to be rescheduled
            operation_times: Dictionary mapping (operation_id, machine_id) to processing time
            schedule_manager: Manager handling the original schedule for comparison

        """
        self.env = jobshop
        self.broken_machine = broken_machine
        self.current_time = current_time
        self.operations_toprocess, self.operation_times = initializer.extract_sets()
        self.nr_op = len(self.operations_toprocess)
        self.operations_ongoing = initializer.get_operations_ongoing()
        self.operations_completed = initializer.get_operations_completed()
        self.operations_final = initializer.get_operations_final()
        self.schedule_manager = schedule_manager
        self.start_times = {}
        self.end_times = {}
        self.job_times = {}
        self.last_real_objectives = []

        self.ideal_point = None
        self.nadir_point = None
        self.normalization_history = []


    def start_new_generation(self):
        self.last_real_objectives = []


    def update_reference_points(self, population_objectives: np.ndarray):
        """
        Updates ideal point and nadir point.

        Args:
            population_objectives: Array of shape (n_individuals, n_objectives) with all the objectives of the generation.
        """
        if len(population_objectives) == 0:
            return

        # Compute current ideal and nadir point of the population.
        current_ideal = np.min(population_objectives, axis=0)
        current_nadir = np.max(population_objectives, axis=0)

        # For the ideal point, take the minimum between the current ideal point and the minimum in the history
        if self.ideal_point is None:
            self.ideal_point = current_ideal.copy()
        else:
            self.ideal_point = np.minimum(self.ideal_point, current_ideal)

        # Take the current nadir point
        self.nadir_point = current_nadir.copy()


    def decode(self, individual: np.ndarray) -> np.ndarray:
        """
        Decodes an individual chromosome into a schedule and evaluates its quality.

        Args:
            individual: Chromosome representing a solution [MS part | OS part]

        Returns:
            np.ndarray: Array of three objective function values
        """
        # Split chromosome into Machine Selection (MS) and Operation Sequence (OS) parts
        ms = individual[ :self.nr_op]
        os = individual[self.nr_op: ]

        # Decode MS part to get machine assignments
        machine_assignments = self._decode_ms(ms)
        if machine_assignments is None:
            # Return worst-case objectives for invalid individual
            return ([1e6, 1e6, 1e6])

        # Decode OS part to get operation sequence
        operation_sequence = self._decode_os(os)

        self.simulate_schedule(operation_sequence, machine_assignments)

        # Simulate the schedule to get job completion times
        job_completion_times = self._get_job_completion_times()

        objectives = self._calculate_objectives(job_completion_times, machine_assignments)

        # normalized_objectives = self._normalize_objectives(objectives)

        # Calculate the three objective functions
        return objectives


    def decode_with_no_early_constraint(self, individual: np.ndarray):
        """
        Come decode(), ma in più calcola un vincolo:
        nessun job può finire prima del completion time del piano originale.

        Ritorna:
          - objectives: np.ndarray shape (3,)
          - g: float, vincolo (<=0 ok, >0 violazione)
        """
        # Split chromosome
        ms = individual[:self.nr_op]
        os = individual[self.nr_op:]

        # Decode MS

        machine_assignments = self._decode_ms(ms)
        if machine_assignments is None:
            return np.array([1e6, 1e6, 1e6]), 1e6

        # Decode OS + simulate
        operation_sequence = self._decode_os(os)
        self.simulate_schedule(operation_sequence, machine_assignments)
        job_completion_times = self._get_job_completion_times()

        # Vincolo: new_end >= old_end per ogni job
        original_schedule = self.schedule_manager.get_original_schedule()
        old_job_ct = original_schedule["job_completion_times"]

        # g = max(old_end - new_end)  -> se >0 c'è almeno un anticipo
        g = -np.inf
        for job_id, new_end in job_completion_times.items():
            old_end = old_job_ct.get(job_id)
            if old_end is not None:
                g = max(g, old_end - new_end)

        if g == -np.inf:
            g = 0.0

        # Obiettivi originali
        objectives = self._calculate_objectives(job_completion_times, machine_assignments)
        return objectives, g


    def extract_sets(self, individual, debug=False):
        """
        Given an individual, it extracts the schedule, defined by the sets:
        - X_ik: assignment of operation i to machine k
        - S: start time of operation i
        - C: end time of operation i

        Args:
             individual: Chromosome representing a solution [MS part | OS part]

            Returns:
                X: dictionary representing the set of assignments
                S: dictionary representing the set of start times
                C: dictionary representing the set of end times
        """
        ms = individual[:self.nr_op]    # Machine selection part
        os = individual[self.nr_op:]    # Operation sequence part

        machine_assignments = self._decode_ms(ms)
        operation_sequence = self._decode_os(os)

        self.simulate_schedule(operation_sequence, machine_assignments)

        original_schedule = self.schedule_manager.get_original_schedule()
        operation_ids = [op.operation_id for op in self.env.operations]
        machine_ids = [m.machine_id for m in self.env.machines]

        # Set of assignments operation - machine
        X = {}
        # Set of assignments of operation - machine in the "old schedule"
        X_old = {}

        # Populate X_old from the original schedule
        for i in operation_ids:
            for k in machine_ids:
                if original_schedule['machine_assignments'][i] == k:
                    X_old[i,k]=1
                else:
                    X_old[i,k]=0

        # Populate X from current individual (solution)
        for i in operation_ids:
            assigned_machine = None

            # For operations to process
            if i in self.operations_toprocess:
                assigned_machine = machine_assignments.get(i)
            # For completed/ongoing, get from original schedule
            elif i in self.operations_ongoing + self.operations_completed:
                assigned_machine = original_schedule['machine_assignments'].get(i)

            # Set X[i,k] for all machines
            for k in machine_ids:
                if k == assigned_machine:
                    X[i, k] = 1
                else:
                    X[i, k] = 0

        # Set of the start times of the operations
        S = {}
        # Set of the end times of the operations
        C = {}

        # Derive the start and end times of the ongoing and completed operations from the "old schedule"
        for operation in self.env.operations:
            i = operation.operation_id
            if i in self.operations_ongoing + self.operations_completed:
                S[i] = operation.scheduled_start_time
                C[i] = operation.scheduled_end_time
            else:
                S[i] = self.start_times.get(i, 0)
                C[i] = self.end_times.get(i, 0)

        Y = {}

        for operation1 in self.env.operations:
            i = operation1.operation_id
            for operation2 in self.env.operations:
                h = operation2.operation_id
                if i != h:
                    if S[i] < S[h]:
                        Y[i, h] = 1
                    else:
                        Y[i, h] = 0

        E = {}

        for i,k in X:
            if X[i,k] == 1 and X_old[i,k] == 0:
                E[i,k] = 1
            elif X[i,k] == 1 and X_old[i,k] == 1:
                E[i,k] = 0
            elif X[i,k] == 0 and X_old[i,k] == 1:
                E[i,k] = 1
            elif X[i,k] == 0 and X_old[i,k] == 0:
                E[i,k] = 0

        Z = {}
        Z_aux = {}

        for operation in self.env.operations:
            i = operation.operation_id
            if i in self.operations_toprocess:
                j = operation.job_id
                Z[i] = C[i] - original_schedule['job_completion_times'][j]
                Z_aux[i] = Z[i]*Z[i]

        if debug:
            self._print_extract_sets_debug(X, S, C, operation_ids, machine_ids)

        return X, Y, S, C, E, Z, Z_aux


    def _print_extract_sets_debug(self, X, S, C, operation_ids, machine_ids):
        print("\n" + "=" * 60)
        print("SCHEDULE SUMMARY")
        print("=" * 60)

        # Get job information for each operation
        job_info = {}
        for op in self.env.operations:
            job_info[op.operation_id] = op.job_id

        # Print table header
        print(f"{'Op':<4} {'Job':<4} {'Machine':<8} {'Start':<8} {'End':<8} {'Dur':<6} {'Status':<12}")
        print("-" * 60)

        # Sort operations by start time
        ops_sorted = sorted(operation_ids, key=lambda i: S.get(i, 0))

        for op_id in ops_sorted:
            # Get machine assignment
            assigned_machine = None
            for machine_id in machine_ids:
                if X.get((op_id, machine_id), 0) == 1:
                    assigned_machine = machine_id
                    break

            # Get times
            start_time = S.get(op_id, 0)
            end_time = C.get(op_id, 0)
            duration = end_time - start_time if end_time > start_time else 0

            # Determine status
            if op_id in self.operations_toprocess:
                status = "TO_PROCESS"
            elif op_id in self.operations_ongoing:
                status = "ONGOING"
            elif op_id in self.operations_completed:
                status = "COMPLETED"
            else:
                status = "UNKNOWN"

            print(f"{op_id:<4} {job_info.get(op_id, '?'):<4} {assigned_machine or 'N/A':<8} "
                  f"{start_time:<8.1f} {end_time:<8.1f} {duration:<6.1f} {status:<12}")

        # Print per-machine summary
        print("\n" + "-" * 60)
        print("PER-MACHINE SCHEDULE:")
        print("-" * 60)

        # Group operations by machine
        machine_ops = {}
        for machine_id in machine_ids:
            machine_ops[machine_id] = []

        for (op_id, machine_id), val in X.items():
            if val == 1:
                machine_ops[machine_id].append(op_id)

        for machine_id in machine_ids:
            ops = machine_ops[machine_id]
            if ops:
                # Sort by start time
                ops_sorted = sorted(ops, key=lambda i: S.get(i, 0))
                timeline = []
                for op_id in ops_sorted:
                    timeline.append(f"{op_id}({S.get(op_id, 0):.0f}-{C.get(op_id, 0):.0f})")

                print(f"Machine {machine_id}: {' → '.join(timeline)}")

        # Calculate makespan
        if C:
            makespan = max(C.values())
            print(f"\nMakespan: {makespan:.1f}")

        print("=" * 60 + "\n")


    def _decode_ms(self, ms: np.ndarray) -> Dict:
        """
        Decodes the Machine Selection part of the chromosome.
        Converts machine indices to actual machine IDs for each operation.

        Args:
            ms: Machine Selection part of chromosome

        Returns:
            Dict: Mapping from operation_id to assigned machine_id, or None if invalid
        """
        machine_assignments = {}
        idx=0

        # Iterate through all operations in all jobs
        for job in self.env.jobs:
            for operation in job.operations:
                if operation.operation_id in self.operations_toprocess:
                    # Get valid machines (excluding broken machine)
                    valid_machines = [m for m in operation.optional_machines_id if m!= self.broken_machine]
                    ms_idx = ms[idx]
                    # Check if machine index is valid
                    if ms_idx >= len(valid_machines):
                        return None
                    # Assign machine to operation
                    machine_id = valid_machines[ms_idx]
                    machine_assignments[operation.operation_id] = machine_id
                    idx += 1

        return machine_assignments


    def _decode_os(self, os: np.ndarray) -> List[int]:
        """
        Decodes the Operation Sequence part of the chromosome.
        Converts job sequence to operation sequence respecting job precedence constraints.

        Args:
            os: Operation Sequence part of chromosome

        Returns:
            List[int]: Sequence of operation IDs in execution order
        """
        operation_sequence = []
        job_counter = {job.job_id: 0 for job in self.env.jobs}

        # Convert sequence of job ids to sequence of operation ids
        for job_id in os:
            job = self.env.get_job(job_id)
            operations_to_reschedule = [
                op.operation_id for op in job.operations
                if op.operation_id in self.operations_toprocess
            ]
            # Add next operation of this job to sequence
            counter = job_counter[job_id]
            if counter < len(operations_to_reschedule):
                next_operation = operations_to_reschedule[counter]
                operation_sequence.append(next_operation)
                job_counter[job_id] += 1

        return operation_sequence


    def simulate_schedule(self, operation_sequence, machine_assignments):
        """
        Simulates the execution of operations to calculate completion times.

        Args:
            operation_sequence: Ordered list of operation IDs to be processed
            machine_assignments: Mapping from operation_id to machine_id

        Returns:
            Dict: Mapping from job_id to completion time
        """
        # Initialize machine availability times
        machine_times = {}

        for machine in self.env.machines:
            ongoing_op = None
            for op_schedule in machine.scheduled_operations:
                if op_schedule.operation_id in self.operations_ongoing:
                    ongoing_op = op_schedule
                    break

            if ongoing_op:
                # Machine has ongoing operation - use its completion time
                machine_times[machine.machine_id] = ongoing_op.scheduled_end_time
            else:
                machine_times[machine.machine_id] = self.current_time

        # Initialize job completion times
        self.job_times = {j.job_id: self.current_time for j in self.env.jobs}

        for job in self.env.jobs:
            job_ongoing_end_time = self.current_time
            for machine in self.env.machines:
                for op_schedule in machine.scheduled_operations:
                    if (op_schedule.job_id == job.job_id and
                            op_schedule.operation_id in self.operations_ongoing):
                        job_ongoing_end_time = max(job_ongoing_end_time, op_schedule.scheduled_end_time)

            self.job_times[job.job_id] = job_ongoing_end_time

        # Process operations in sequence
        for operation_id in operation_sequence:
            operation = self.env.get_operation(operation_id)
            machine_id = machine_assignments[operation_id]
            job_id = operation.job_id

            # Get processing time (considering partial processing for broken operations)
            processing_time = self.operation_times.get(
                (operation.operation_id, machine_id),
                operation.processing_times[machine_id]
            )

            # Calculate start and end times (respecting machine and job constraints)
            start_time = max(self.job_times[job_id], machine_times[machine_id], operation.scheduled_start_time, self.current_time)
            end_time = start_time + processing_time

            self.start_times[operation_id] = start_time
            self.end_times[operation_id] = end_time

            # Update job and machine times
            self.job_times[job_id] = end_time
            machine_times[machine_id] = end_time


    def _calculate_objectives(self, job_completion_times: Dict, machine_assignments: Dict) -> np.ndarray:
        """
        Calculates the three objective functions for the schedule.

        Args:
            job_completion_times: Mapping from job_id to completion time in new schedule
            machine_assignments: Mapping from operation_id to machine_id in new schedule

        Returns:
            np.ndarray: Array containing [time_differences, assignment_differences, makespan]
        """
        original_schedule = self.schedule_manager.get_original_schedule()

        # 1st objective: Sum of squared delays in job completion times
        time_differences = 0
        for job_id, new_end in job_completion_times.items():
            old_end = original_schedule['job_completion_times'].get(job_id)
            time_differences += (new_end-old_end)**2

        # 2nd objective: Number of changed machine assignments
        assignment_differences = 0
        changes = []
        for op_id, new_machine in machine_assignments.items():
            old_machine = original_schedule['machine_assignments'].get(op_id)
            if old_machine != new_machine:
                assignment_differences += 1
                changes.append(f"Op {op_id}: {old_machine}→{new_machine}")

        # 3rd objective: Makespan
        makespan = max(job_completion_times.values())

        return np.array([time_differences, assignment_differences, makespan])


    def _get_start_times(self):
        return self.start_times


    def _get_end_times(self):
        return self.end_times


    def _get_job_completion_times(self):
        return self.job_times


    def normalize_pareto_front(self, original_front, optimal_front=None):
        if optimal_front is not None and len(optimal_front) > 0:
            all_fronts = np.vstack([original_front, optimal_front])
        else:
            all_fronts = original_front

        # Compute min and max
        min_vals = np.min(all_fronts, axis=0)
        max_vals = np.max(all_fronts, axis=0)

        # Compute range
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0

        # Normalize the front
        normalized_original = (original_front - min_vals) / ranges

        normalized_optimal = None
        if optimal_front is not None and len(optimal_front) > 0:
            normalized_optimal = (optimal_front - min_vals) / ranges

        # Save stats
        stats = {
            'min': min_vals,
            'max': max_vals,
            'ranges': ranges
        }

        return normalized_original, normalized_optimal, stats


    def _normalize_objectives(self, real_objectives: np.ndarray) -> np.ndarray:
        """
        Normalize objective values using the reference points.

        Args:
            real_objectives: values of the non-normalized objectives

        Returns:
            np.ndarray: normalized objectives
        """
        if self.ideal_point is None or self.nadir_point is None:
            return real_objectives

        # Normalization: (value - ideal point) / (nadir - ideal point)
        normalized = (real_objectives - self.ideal_point) / (self.nadir_point - self.ideal_point)

        # For debug
        self.normalization_history.append({
            'real': real_objectives.copy(),
            'normalized': normalized.copy(),
            'ideal': self.ideal_point.copy(),
            'nadir': self.nadir_point.copy()
        })

        return normalized


