# Importing libraries
import time
import os
import gurobipy as gb
from scheduling_environment.jobShop import JobShop
from visualization import gantt_chart
from MILP.incumbent import Incumbent, add_current_sol
from functools import partial


class FJS_reschedule(JobShop):
    """
    Class for rescheduling a Flexible Job Shop (FJS) environment after a machine breakdown.
    Inherits from JobShop.

    Attributes:
        id_broken: ID of the broken machine.
        time_broken: Time at which the machine breaks down.
        _original_env: Original JobShop environment.
        old_schedule: Dictionary storing previous schedule information.
        operations_completed: List of operations already completed.
        operations_ongoing: List of operations in progress (not on the broken machine).
        operations_toprocess: List of operations not yet started and eligible for rescheduling.
        operations_final: List of final operations for each job.
        machine_allocations: Dictionary of valid machines for each operation.
        operation_times: Dictionary of processing times, considering partial completion if broken.
        model: Gurobi MILP model for rescheduling.
        results: Dictionary storing optimization results.
    """

    def __init__(self, jobShopEnv, id_broken, time_broken):
        super().__init__()
        self.id_broken = id_broken
        self.time_broken = time_broken
        self._original_env = jobShopEnv
        self.old_schedule = {}
        self.operations_completed = []
        self.operations_ongoing = []
        self.operations_toprocess = []
        self.operations_final = []
        self.machine_allocations = {}
        self.operation_times = {}
        self.model = None
        self.results = {}

        self.incumbent = Incumbent()

    @property
    def operations(self):
        """
        Return the operationd from the original environment.
        """
        return self._original_env.operations

    @property
    def machines(self):
        """
        Return the machines from the original environment.
        """
        return self._original_env.machines

    @property
    def jobs(self):
        """
        Return the jobs from the original environment.
        """
        return self._original_env.jobs

    def extract_info(self):
        """
        Extracts information from the current schedule from the JobShop environment.
        This function populates:
        - old_schedule:
            - X_old: previous machine assignments
            - Y_old: sequencing
            - S_old: start times
            - C_old: end times
        - operations_completed: operations already completed.
        - operations_ongoing: operations in progress (not on the broken machine).
        - operations_toprocess: operations not yet started.
        - operations_final: final operations of each job.
        - machine_allocations: machines where each operation can be processed.
        - operation_times: processing times for operations, adjusting for partially completed operations on broken machine.
        """
        # ------ Dictionaries for old schedule ------

        # Binary assignment variables X_old[i,k]
        X_old = {}
        for operation in self.operations:
            operation_id = operation.operation_id
            for machine in self.machines:
                machine_id = machine.machine_id
                if operation.scheduled_machine == machine_id:
                    X_old[operation_id, machine_id] = 1
                else:
                    X_old[operation_id, machine_id] = 0

        # Sequencing variables Y_old[i,h]
        Y_old = {}
        for operation1 in self.operations:
            i = operation1.operation_id
            for operation2 in self.operations:
                h = operation2.operation_id
                if operation1.scheduled_start_time < operation2.scheduled_start_time:
                    Y_old[i, h] = 1
                else:
                    Y_old[i, h] = 0

        # Start and end times
        S_old = {operation.operation_id: operation.scheduled_start_time for operation in self.operations}
        C_old = {operation.operation_id: operation.scheduled_end_time for operation in self.operations}

        self.old_schedule = {
            "X_old": X_old,
            "Y_old": Y_old,
            "S_old": S_old,
            "C_old": C_old
        }

        # ------ Categorize operations based on their status at time broken ------
        for operation in self.operations:
            machine = operation.scheduled_machine

            if operation.scheduled_end_time <= self.time_broken:
                self.operations_completed.append(
                    operation.operation_id)  # The operation has been completely processed, it must not be rescheduled.
            elif (
                    operation.scheduled_start_time < self.time_broken < operation.scheduled_end_time
            # The operation has started before a machine has broken, but has not finished yet.
            ):
                if machine != self.id_broken:
                    self.operations_ongoing.append(
                        operation.operation_id)  # The operation is ongoing in another machine, it must not be rescheduled.
                else:
                    self.operations_toprocess.append(
                        operation.operation_id)  # The operation is ongoing in the broken machine, so it must be rescheduled in another one.
            else:
                self.operations_toprocess.append(
                    operation.operation_id)  # The operation has not started yet, it can be rescheduled.

        # Identify final operations of each job -> necessary for our main objective
        for job in self.jobs:
            operations = [operation.operation_id for operation in job.operations]
            self.operations_final.append(operations[-1])

        # ------ Machines available for each operation ------
        self.machine_allocations = {
            operation.operation_id: [
                m for m in operation.optional_machines_id
            ]
            for operation in self.operations
        }

        # ------ Compute processing times (adjusted if broken) ------
        self.operation_times = {}
        for operation in self.operations:
            op_id = operation.operation_id
            for k in self.machine_allocations[op_id]:
                if (
                        operation.scheduled_start_time < self.time_broken < operation.scheduled_end_time
                        and k == self.id_broken and k == operation.scheduled_machine
                ):
                    operation.set_broken(self.time_broken)
                    broken_info = operation.is_broken_info
                    percent_to_do = broken_info['percentage_to_do']
                    self.operation_times[(op_id, k)] = percent_to_do * operation.processing_times[k]
                else:
                    self.operation_times[(op_id, k)] = operation.processing_times[k]


    def print_info(self):
        """
        Prints the current schedule of the environment by machine.
        """
        for machine in self.machines:
            print("*******************************************************************")
            print(f"Machine ID {machine.machine_id} has the following operations:")
            for operation in machine.scheduled_operations:
                print(f"Operation ID {operation.operation_id} with start time {operation.scheduled_start_time}")


    def plot_schedule(self):
        """
        Plots the current schedule using the gantt_chart visualization.
        """
        gantt_chart.plot(self._original_env)


    def create_model(self, obj1_value=None, obj2_value=None, obj3_value=None, obj1_lb=None):
        """
        Creates the MILP model to reschedule the production plan after a machine breakdown.
        """

        # Retrieve old schedule for the environment
        X_old = self.old_schedule["X_old"]
        S_old = self.old_schedule["S_old"]
        C_old = self.old_schedule["C_old"]

        operation_ids = [operation.operation_id for operation in self.operations]
        machine_ids = [machine.machine_id for machine in self.machines]

        # Big M constant
        Cmax_old = max(self.old_schedule["C_old"].values())

        remaining_time = 0
        for i in self.operations_toprocess:
            op = self.get_operation(i)
            max_pt = max(op.processing_times[m]
                         for m in op.optional_machines_id
                         if m != self.id_broken)
            remaining_time += max_pt

        # Big M constant
        M = Cmax_old + remaining_time

        model = gb.Model("FJSP_Rescheduling")

        # ------ Decision variables ------

        X = {}      # set of assignments, x_ik: binary variable equal to 1 if operation i is assigned to machine k
        Y = {}      # set of sequences, y_ii': binary variable equal to 1 if operation i is assigned before operation i'
        S = {}      # set of starting times of operations, s_i: int variable that defines the starting time of operation i
        E = {}      # set of auxiliary variables for X
        U = {}
        Z = {}      # set of auxiliary variables for complete times of job
        Z_aux = {}  # set of auxiliary variable for the squared of the delay in the complete times of jobs
        C = {}  # c_i: end time of the operations

        for i in operation_ids:
            S[i] = model.addVar(lb=0, vtype=gb.GRB.INTEGER, name=f"s_{i}")
            C[i] = model.addVar(lb=0, vtype=gb.GRB.INTEGER, name=f"c_{i}")
            for k in machine_ids:
                X[i, k] = model.addVar(lb=0, vtype=gb.GRB.BINARY, name=f"x_{i}_{k}")
                E[i, k] = model.addVar(lb=0, vtype=gb.GRB.BINARY, name=f"e_{i}_{k}")

        for operation1 in self.operations:
            i = operation1.operation_id
            for operation2 in self.operations:
                h = operation2.operation_id
                if i != h:
                    Y[i, h] = model.addVar(lb=0, vtype=gb.GRB.BINARY, name=f"y_{i}_{h}")

        for i in self.operations_final:
            U[i] = model.addVar(vtype=gb.GRB.BINARY, name=f"U_{i}")
            Z[i] = model.addVar(lb=0, vtype=gb.GRB.INTEGER, name=f"z_{i}")
            Z_aux[i] = model.addVar(lb=0, vtype=gb.GRB.CONTINUOUS, name=f"z_aux_{i}")

        cmax = model.addVar(
            lb=0, vtype=gb.GRB.INTEGER, name="cmax"
        )

        if obj1_lb is not None:
            obj_quadr = model.addVar(
                lb=obj1_lb,
                vtype=gb.GRB.INTEGER,
                name="obj_quadr_delay"
            )
        else:
            obj_quadr = model.addVar(
                lb=0,
                vtype=gb.GRB.INTEGER,
                name="obj_quadr_delay"
            )

        obj_mach_assignments = model.addVar(
            lb=0, vtype=gb.GRB.INTEGER, name="obj_mach_assignments"
        )

        C_old = self.old_schedule["C_old"]
        Cmax_old = max(C_old.values())

        remaining_work = 0
        for op in self.operations:
            if hasattr(op, "processing_times") and op.processing_times:
                max_pt = max(op.processing_times.values())
            else:
                max_pt = 0
            remaining_work += max_pt

        UB_makespan = Cmax_old + remaining_work

        # ------ Objective functions ------

        """
        Our model is multi-objective, we need to minimise three values at the same time. 
        I start with two objectives, a third one will be added later. 
        Since the first objective is to minimise the  total delay of the final operations of the jobs, 
        """

        tolerance = 0.001

        #   Optimize for the first objective
        if obj1_value is None:

            model.setObjective(
                obj_quadr, gb.GRB.MINIMIZE
            )

            model.addConstr(
                obj_mach_assignments <= obj2_value * (1 + tolerance)
            )
            model.addConstr(
                obj_mach_assignments >= obj2_value * (1 - tolerance)
            )

            model.addConstr(
                cmax <= obj3_value * (1 + tolerance)
            )

            model.addConstr(
                cmax >= obj3_value * (1 - tolerance)
            )

        # Optimize for the second objective
        elif obj2_value is None:

            model.setObjective(
                obj_mach_assignments, gb.GRB.MINIMIZE
            )

            model.addConstr(
                obj_quadr <= obj1_value * (1 + tolerance)
            )

            model.addConstr(
                obj_quadr >= obj1_value * (1 - tolerance)
            )

            model.addConstr(
                cmax <= obj3_value * (1 + tolerance)
            )

            model.addConstr(
                cmax >= obj3_value * (1 - tolerance)
            )

        # Optimize for the third objective
        elif obj3_value is None:

            model.setObjective(
                cmax, gb.GRB.MINIMIZE
            )

            model.addConstr(
                obj_quadr <= obj1_value * (1 + tolerance)
            )

            model.addConstr(
                obj_quadr >= obj1_value * (1 - tolerance)
            )

            model.addConstr(
                obj_mach_assignments <= obj2_value * (1 + tolerance)
            )
            model.addConstr(
                obj_mach_assignments >= obj2_value * (1 - tolerance)
            )

        else:
            raise ValueError("Not valid values.")


        # ------ Constraints ------

        # Preserve assignments of completed and ongoing operations
        for i in self.operations_ongoing + self.operations_completed:
            for machine in self.machines:
                k = machine.machine_id
                model.addConstr(X[i, k] == X_old[i, k], name="assignm_done_" + str(i) + "_" + str(k))

        # Preserve start and end times
        for i in self.operations_ongoing + self.operations_completed:
            operation = self.get_operation(i)
            model.addConstr(S[i] == S_old[i], name="start_done_" + str(i))
            model.addConstr(C[i] == C_old[i], name="end_done_" + str(i))

        # Operations to process cannot start before the breakdown
        for i in self.operations_toprocess:
            model.addConstr(
                S[i] >= S_old[i],
                name="start_toproc_" + str(i)
            )
            model.addConstr(
                S[i] >= self.time_broken,
                name="start_toproc" + str(i)
            )

        model.addConstr(
            obj_quadr == gb.quicksum(Z_aux[i] for i in self.operations_final)
        )

        model.addConstr(
            obj_mach_assignments == (gb.quicksum(
                E[i, k] for i in operation_ids for k in self.machine_allocations[i])) / 2
        )

        # Completion time = start time + processing time
        for i in self.operations_toprocess:
            operation = self.get_operation(i)
            valid_machines = [k for k in operation.processing_times.keys()]
            if not valid_machines:
                raise ValueError(f"No valid machine for operation {i}")
            model.addConstr(
                C[i] == S[i] + gb.quicksum(
                    X[i, k] * self.operation_times[i,k]
                    for k in valid_machines
                ),
                name="compl_toproc_" + str(i)
            )

        # Assign operations to exactly one valid machine
        for i in self.operations_toprocess:
            valid_machines = [k for k in self.machine_allocations[i] if k != self.id_broken]
            model.addConstr(
                gb.quicksum(X[i, k] for k in valid_machines) == 1,
                name="assignm_" + str(i)
            )
            # Force assignment to 0 on invalid machines (including the broken one)
            for k in machine_ids:
                if k not in valid_machines:
                    model.addConstr(X[i, k] == 0, name=f"no_invalid_machine_{i}")

        # Precedence constraint for operations within the same job
        for i in self.operations_toprocess:
            operation = self.get_operation(i)
            job = operation.job
            operations_in_job = job.operations
            idx = operations_in_job.index(operation)
            if idx > 0:  # the operation has a predecessor
                pred = operations_in_job[idx - 1]
                if pred.operation_id in self.operations_completed + self.operations_ongoing:
                    # If predecessor is already done/ongoing, start after its original completion time
                    model.addConstr(
                        S[i] >= C_old[pred.operation_id],
                        name=f"prec_done_{i}_{pred.operation_id}"
                    )
                else:
                    # Otherwise standard precedence: S_i >= S_pred + processing_time(pred)
                    model.addConstr(
                        S[operation.operation_id] >= S[pred.operation_id] +
                        gb.quicksum(
                            self.operation_times[pred.operation_id,k] * X[pred.operation_id, k] for k in pred.optional_machines_id),
                        name="prec_job_" + str(i)
                    )

        # Machine non-overlap constraints
        for i in operation_ids:
            for h in operation_ids:
                if i != h:
                    common_machines = set(self.machine_allocations[i]).intersection(self.machine_allocations[h])
                    for k in common_machines:
                        if (i, k) in self.operation_times and (h, k) in self.operation_times:
                            model.addConstr(
                                S[i] >= S[h] + self.operation_times[h, k] - M * (2 - X[i, k] - X[h, k] + Y[i, h]),
                                name="op_machine_1_" + str(i) + "_" + str(h)
                            )
                            model.addConstr(
                                S[h] >= S[i] + self.operation_times[i, k] - M * (3 - X[i, k] - X[h, k] - Y[i, h]),
                                name="op_machine_2_" + str(i) + "_" + str(h)
                            )

        # Constraint for cmax
        for i in self.operations_toprocess:
            model.addConstr(
                cmax >= S[i] + gb.quicksum((self.operation_times[i, k] * X[i, k])
                                           for k in self.machine_allocations[i]),
                name="cmax_" + str(i)
            )

        for i in self.operations_final:
            model.addConstr(
                cmax <= C[i] + M * (1 - U[i]),
                name=f"cmax_upper_{i}"
            )
            model.addConstr(
                cmax >= C[i],
                name=f"cmax_lower_{i}"
            )

        model.addConstr(
            gb.quicksum(U[i] for i in self.operations_final) == 1,
            name="one_last_operation"
        )

        # Auxiliary Z constraints for final operations (delay contribution)
        epsilon= 0.001
        for i in self.operations_final:
            model.addConstr(
                Z[i] <= C[i] - C_old[i] + epsilon,
                name=f"tight_upper_{i}"
            )

            model.addConstr(
                Z[i] >= C[i] - C_old[i] - epsilon,
                name=f"tight_lower_{i}"
            )

            model.addConstr(
                C[i] >= C_old[i],
                name=f"tight_lower_{i}"
            )

            """max_possible_completion = UB_makespan
            Z_ub = max(0, max_possible_completion - C_old[i])
            model.addConstr(
                Z[i] >= C[i] - C_old[i],
                name="compl_op_" + str(i)
            )
            model.addConstr(
                Z[i] >= 0,
                name="compl_op_nonneg" + str(i)
            )
            model.addConstr(
                Z[i] <= Z_ub,
                name=f"delay_ub_{i}"
            )"""
            model.addConstr(
                Z[i] >= 0
            )
            model.addQConstr(
                Z_aux[i] == Z[i] * Z[i],
                name="compl_op_aux_" + str(i)
            )

        # Constraint for auxiliary variable E (measures assignment changes)
        for operation in self.operations:
            i = operation.operation_id
            for machine in self.machines:
                k = machine.machine_id
                model.addConstr(X_old[i, k] - X[i, k] >= -E[i, k], name="const_x_1_" + str(i) + " " + str(h))
                model.addConstr(X_old[i, k] - X[i, k] <= E[i, k], name="const_x_2_" + str(i) + " " + str(h))

        self.model = model

        self.model.update()


    def run_model(self, time_limit=None, X_start=None, S_start=None,
                  C_start=None, Y_start=None, E_start=None, Z_start=None, Z_aux_start=None):
        """
        Solve the Gurobi model by managing the multiple objectives using a lexicographic approach.
        Since one of the objectives is quadratic, so it is not linear, the leicographic approach is applied "manually", solving
        the model first using the "most important" objective, and then solving it on the second objective, adding a costraint
        that keeps the value correlated to the first objective as close as possible to the value found before within a certain
        tolerance.

        Args:
            time_limit (float or int, optional): time limit in seconds for the solver.
        Returns:
            dict or None: result dictionary with X, S, C, cmax if a solution is found; otherwise, None
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")

        if X_start is not None:
            for (i, k), val in X_start.items():
                var = self.model.getVarByName(f"x_{i}_{k}")
                if var:
                    var.Start = val

        if S_start is not None:
            for i, val in S_start.items():
                var = self.model.getVarByName(f"s_{i}")
                if var:
                    var.Start = val

        if C_start is not None:
            for i, val in C_start.items():
                var = self.model.getVarByName(f"c_{i}")
                if var:
                    var.Start = val

        if E_start is not None:
            for (i,k), val in E_start.items():
                var = self.model.getVarByName(f"e_{i}_{k}")
                if var:
                    var.Start = val

        if Y_start is not None:
            for (i,h), val in Y_start.items():
                var = self.model.getVarByName(f"y_{i}_{h}")
                if var:
                    var.Start = val

        if Z_start is not None:
            for i, val in Z_start.items():
                var = self.model.getVarByName(f"z_{i}")
                if var:
                    var.Start = val

        if Z_aux_start is not None:
            for i, val in Z_aux_start.items():
                var = self.model.getVarByName(f"z_aux_{i}")
                if var:
                    var.Start = val

        callback = partial(add_current_sol, incumbent_obj=self.incumbent)
        self.model._start_time = time.time()

        self.model.Params.TimeLimit = time_limit
        
        self.model.update()

        self.model.Params.Presolve = 2
        self.model.Params.MIPFocus = 3
        self.model.Params.Heuristics = 0.1
        self.model.Params.ImproveStartGap = 0.5
        self.model.Params.Presolve = 2

        self.model.optimize(callback=callback)

        # If infeasible, compute IIS and print relevant info
        if self.model.Status == gb.GRB.INFEASIBLE:
            self.model.computeIIS()
            # Print out the IIS constraints and variables
            print('\nThe following constraints and variables are in the IIS:')
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')

            for v in self.model.getVars():
                if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')

        if self.model.Status == gb.GRB.OPTIMAL or self.model.Status == gb.GRB.TIME_LIMIT:

            # Extract the results
            operation_ids = [operation.operation_id for operation in self.operations]
            machine_ids = [machine.machine_id for machine in self.machines]
            X_values = {(i, k): self.model.getVarByName(f"x_{i}_{k}").X for i in operation_ids for k in
                        machine_ids}
            S_values = {i: self.model.getVarByName(f"s_{i}").X for i in operation_ids}
            C_values = {i: self.model.getVarByName(f"c_{i}").X for i in operation_ids}
            cmax_value = self.model.getVarByName("cmax").X

            X_old = self.old_schedule["X_old"]
            C_old = self.old_schedule["C_old"]

            # Objective 1: Quadratic delay
            quadratic_delay = 0
            for i in self.operations_final:
                delay = max(0, C_values[i] - C_old[i])
                quadratic_delay += delay * delay

            # Objective 2: Assignment changes
            assignment_changes = 0
            for i in self.operations_toprocess:
                old_machine = None
                new_machine = None

                # Find old machine
                for k in self.machine_allocations[i]:
                    if X_old.get((i, k), 0) > 0.5:
                        old_machine = k
                        break

                # Find new machine
                for k in self.machine_allocations[i]:
                    if X_values.get((i, k), 0) > 0.5:
                        new_machine = k
                        break

                if old_machine is not None and new_machine is not None:
                    if old_machine != new_machine:
                        assignment_changes += 1

            # Save results in the class
            self.results = {
                "X": X_values,
                "S": S_values,
                "C": C_values,
                "cmax": cmax_value,
                "quadratic_delay": quadratic_delay,
                "assignment_changes": assignment_changes
            }
            return self.results["quadratic_delay"], self.results["assignment_changes"], self.results["cmax"]
        else:
            print("No feasible solution found.")
            self.results = None
            return None


    def print_results(self):
        """
        Prints the results stored in self.results.
        """
        if not self.results:
            print("No solution found.")
            return

        print("\n=== SOLUTION ===")

        # cmax
        print(f"Cmax: {self.results['cmax']}")

        # Start and Completion times
        print("\nOperations:")
        for i in sorted(self.results["S"].keys()):
            s = self.results["S"][i]
            c = self.results["C"][i]
            # assigned machine
            assigned_machines = [k for (ii, k), val in self.results["X"].items() if ii == i and val > 0.5]
            machine_str = f"Machine {assigned_machines[0]}" if assigned_machines else "N/A"
            print(f" - Operation {i}: start={s:.1f}, end={c:.1f}, {machine_str}")

        true_makespan = max(self.results["C"].values())
        print("\n[DEBUG]")
        print("TRUE makespan from C[i]:", true_makespan)
        print("Model cmax:", self.results["cmax"])

        print(f"\n Objective values: cmax {self.results['cmax']}, quadratric delay {self.results['quadratic_delay']}, "
              f"machine assignments {self.results['assignment_changes']}")

        print("\n Machine Assignments (X[i,k]=1):")
        for (i, k), val in self.results["X"].items():
            if val > 0.5:
                print(f"   Operation {i} → Machine {k}")

        print("\n Machine Assignments (X_old[i,k]=1):")
        for (i, k), val in self.old_schedule["X_old"].items():
            if val == 1:
                print(f"   Operation {i} → Machine {k}")


    def plot_incumbent(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.incumbent.plot_all_phases(save_dir)


    def debug_relaxation(self):
        print("\n=== DETAILED LINEAR RELAXING DEBUG ===")

        # Solve linear relaxing
        model_relax = self.model.copy()
        for v in model_relax.getVars():
            if v.vtype in [gb.GRB.BINARY, gb.GRB.INTEGER]:
                v.vtype = gb.GRB.CONTINUOUS

        model_relax.optimize()

        if model_relax.status != gb.GRB.OPTIMAL:
            print("Not optimal relaxing!")
            return

        print(f"Objective value relaxing: {model_relax.objVal}")

        # For the first 5 operations to process
        for i in list(self.operations_toprocess)[:5]:
            S_val = model_relax.getVarByName(f"s_{i}").X
            C_val = model_relax.getVarByName(f"c_{i}").X
            C_old_val = self.old_schedule["C_old"][i]

            op = self.get_operation(i)
            valid_machines = [k for k in op.processing_times.keys() if k != self.id_broken]

            # Compute sum(X*p) manually
            total_weighted = 0
            for k in valid_machines:
                X_val = model_relax.getVarByName(f"x_{i}_{k}").X
                p = op.processing_times[k]
                total_weighted += X_val * p
                if X_val > 0.01:
                    print(f"  X[{i},{k}]={X_val:.3f} * p={p} = {X_val * p:.2f}")

            print(f"Op {i}: S={S_val:.1f}, C={C_val:.1f}, C_old={C_old_val}")
            print(f"  sum(X*p) = {total_weighted:.1f}, C-S = {C_val - S_val:.1f}")
            print(
                f"  Verify C == S + sum(X*p): {C_val:.1f} == {S_val:.1f} + {total_weighted:.1f} = {S_val + total_weighted:.1f}")

            if i in self.operations_final:
                Z_val = model_relax.getVarByName(f"z_{i}").X
                print(
                    f"  Z[{i}] = {Z_val:.1f} (should be max(0, {C_val:.1f} - {C_old_val:.1f}) = {max(0, C_val - C_old_val):.1f})")


    def update_environment(self):
        """
        Apply the optimization results to the original environment (update operations and machines).
        """
        if not self.results:
            print("⚠️ No solution available, run run_model().")
            return

        X_values = self.results["X"]
        S_values = self.results["S"]
        C_values = self.results["C"]

        # 1. Update each operation's scheduling information based on the solution
        for op in self._original_env.operations:
            i = op.operation_id

            # Identify assigned machine (X[i,k] = 1)
            assigned_machine = None
            for (ii, k), val in X_values.items():
                if ii == i and val > 0.5:
                    assigned_machine = k
                    break

            # Skip operations that are not assigned in the solution (already completed)
            if assigned_machine is None:
                continue

            start_time = S_values.get(i, None)
            end_time = C_values.get(i, None)
            if start_time is None or end_time is None:
                continue

            duration = end_time - start_time
            setup_time = 0

            # Add/update scheduling info on the operation
            op.add_operation_scheduling_information(
                assigned_machine, start_time, setup_time, duration
            )

        # 2. Update machine processed operations lists
        for machine in self._original_env.machines:
            machine._processed_operations = [
                op for op in self._original_env.operations if op.scheduled_machine == machine.machine_id
            ]

        # 3. Update the environment's scheduled/unscheduled operation lists
        self._original_env._scheduled_operations = [
            op for op in self._original_env.operations if op.scheduling_information
        ]
        self._original_env._operations_to_be_scheduled = [
            op for op in self._original_env.operations if not op.scheduling_information
        ]

        self._original_env.update_operations_available_for_scheduling()

        print("Environment updated with the new schedule!")