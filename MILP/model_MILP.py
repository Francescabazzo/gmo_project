import time

import gurobipy as gb

from scheduling_environment.jobShop import JobShop 
from MILP.incumbent import Incumbent, add_current_sol
from functools import partial


class FJS_schedule(JobShop):


    def __init__(self, jobShopEnv): 
        super().__init__()

        self._original_env = jobShopEnv
        self.operations_final = []
        self.model = None 
        self.results = {}

        self.incumbent = Incumbent()

    @property
    def operations(self):
        return self._original_env.operations
    
    @property
    def machines(self):
        return self._original_env.machines
    
    @property
    def jobs(self):
        return self._original_env.jobs


    def create_model(self, start_solution=False):
        """
        Function that creates the MILP Model. 
        """

        operation_ids = [operation.operation_id for operation in self.operations]
        machine_ids = [machine.machine_id for machine in self.machines]

        for job in self.jobs: 
            operations = [operation.operation_id for operation in job.operations]
            self.operations_final.append(operations[-1])

        model = gb.Model("FJSP_Scheduling")

        #   Constant (big M)
        M = sum(
            self.get_operation(op_id).processing_times[m]
            for op_id in operation_ids
            for m in self.get_operation(op_id).optional_machines_id
        )

        #   Decision variables 

        X = {}  # set of assignments, x_ik: binary variable equal to 1 if operation i is assigned to machine k 
        Y = {}  # set of sequences, y_ii': binary variable equal to 1 if operation i is assigned before operation i' 
        S = {}  # set of starting times of operations, s_i: int variable that defines the starting time of operation i 
        C = {}  # set of ending times of operations, c_i: int variable that defines the ending time of operation i

        for i in operation_ids: 
            S[i] = model.addVar(lb=0, vtype=gb.GRB.INTEGER, name=f"s_{i}")
            C[i] = model.addVar(lb=0,vtype=gb.GRB.INTEGER, name=f"c_{i}" )
            for k in machine_ids:
                X[i, k] = model.addVar(lb=0, vtype=gb.GRB.BINARY, name=f"x_{i}_{k}")

                
        for operation1 in self.operations:
            i = operation1.operation_id
            for operation2 in self.operations:
                h = operation2.operation_id
                if i!= h: 
                    Y[i, h] = model.addVar(lb=0, vtype=gb.GRB.BINARY, name=f"y_{i}_{h}")

        cmax = model.addVar(
            lb=0, vtype=gb.GRB.INTEGER, name="cmax"
        )

        #   Objective function
        model.setObjective(
            cmax,
            gb.GRB.MINIMIZE
        )

        #   Constraints 

        #   Each operation must be assigned to one and only machine 
        for operation in self.operations:  
            i = operation.operation_id 
            model.addConstr(
                gb.quicksum(X[i,k] for k in operation.optional_machines_id) == 1,
                name="assignm_"+str(i)+" "+str(h)
            )

        for operation in self.operations: 
            i = operation.operation_id
            job = operation.job 
            operations_in_job = job.operations  #   List of Operation 
            #   idx of the operation specified in the "order" of the operations of the specific job 
            idx = operations_in_job.index(operation)
        
            if idx > 0:  # the operation has a predecessor 
                pred = operations_in_job[idx - 1]
                h = pred.operation_id 
                model.addConstr(
                    S[i] >= S[h] + gb.quicksum(pred.processing_times[k]*X[h, k] for k in pred.optional_machines_id),
                    name="prec_job_"+str(i)+" "+str(h)
                )

        for operation1 in self.operations: 
            i = operation1.operation_id
            for operation2 in self.operations: 
                h = operation2.operation_id
                if i != h:
                    set1 = set(operation1.optional_machines_id)
                    set2 = set(operation2.optional_machines_id)
                    common_machines = set1.intersection(set2)
                    for k in common_machines: 
                        model.addConstr(
                            S[i] >= S[h] + operation2.processing_times[k] - (2-X[i,k]-X[h,k]+Y[i,h])*M,
                            name="op_machine_1_"+str(i)+str(h)
                        )
                        model.addConstr(
                            S[h] >= S[i]+operation1.processing_times[k]-(3-X[i,k]-X[h,k]-Y[i,h])*M,
                            name="op_machine_2_"+str(i)+str(h)
                        )

        for operation in self.operations: 
            i = operation.operation_id 
            model.addConstr(
                cmax >= S[i]+gb.quicksum(X[i,k]*operation.processing_times[k] for k in operation.optional_machines_id),
                name="cmax_op_"+str(operation.operation_id)
            )

        for operation in self.operations:
            i = operation.operation_id 
            model.addConstr(
                C[i] == S[i] + gb.quicksum(
                        X[i,k] * operation.processing_times[k] for k in operation.optional_machines_id
                    )
            )
            
        self.model = model
        self.model.update()

        # =============================
        #  MIP START from GA solution
        # =============================
        if start_solution:
            try:
                scheduled_ops = [op for op in self._original_env.operations if op.scheduled_machine is not None]
                if not scheduled_ops:
                    print("No GA solution found.")
                else:
                    print(f"Adding {len(scheduled_ops)} operations from GA solution as MIP start...")

                    for op in scheduled_ops:
                        i = op.operation_id
                        k = op.scheduled_machine
                        s = op.scheduled_start_time
                        c = op.scheduled_end_time

                        # X[i,k] = 1 for assigned machine
                        var_x = model.getVarByName(f"x_{i}_{k}")
                        if var_x is not None:
                            var_x.Start = 1.0

                        # all other X[i,·] = 0
                        for k_alt in op.optional_machines_id:
                            if k_alt != k:
                                var_x_alt = model.getVarByName(f"x_{i}_{k_alt}")
                                if var_x_alt is not None:
                                    var_x_alt.Start = 0.0

                        # starting and completion times
                        if s is not None:
                            var_s = model.getVarByName(f"s_{i}")
                            if var_s is not None:
                                var_s.Start = s
                        if c is not None:
                            var_c = model.getVarByName(f"c_{i}")
                            if var_c is not None:
                                var_c.Start = c

                    # initialize Y[i,h] from start times (precedence)
                    operation_ids = [op.operation_id for op in self._original_env.operations]
                    for op1 in scheduled_ops:
                        for op2 in scheduled_ops:
                            if op1.operation_id == op2.operation_id:
                                continue
                            y_name = f"y_{op1.operation_id}_{op2.operation_id}"
                            var_y = model.getVarByName(y_name)
                            if var_y is not None:
                                if (op1.scheduled_start_time is not None and
                                    op2.scheduled_start_time is not None and
                                    op1.scheduled_start_time < op2.scheduled_start_time):
                                    var_y.Start = 1.0
                                else:
                                    var_y.Start = 0.0

                    # initialize cmax
                    cmax_var = model.getVarByName("cmax")
                    if cmax_var is not None:
                        cmax_val = max(op.scheduled_end_time for op in scheduled_ops if op.scheduled_end_time is not None)
                        cmax_var.Start = cmax_val

                    print("MIP start successfully added from GA solution.")
            except Exception as e:
                print(f"Warning: unable to set MIP start from GA solution: {e}")



    def run_model(self, time_limit=None): 
        """
        Runs the model, if one has been created 
        """

        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
    
        # Set time limit 
        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit

        callback = partial(add_current_sol, incumbent_obj=self.incumbent)
        self.model._start_time = time.time()
        # Optimize 
        self.model.optimize(callback=callback)

        # do IIS if the model is infeasible
        if self.model.Status == gb.GRB.INFEASIBLE:
            self.model.computeIIS()
            # Print out the IIS constraints and variables
            print('\nThe following constraints and variables are in the IIS:')
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')

            for v in self.model.getVars():
                if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')


        # Check if the solution exists 
        if self.model.Status == gb.GRB.OPTIMAL or self.model.Status == gb.GRB.TIME_LIMIT:
            # Extract the results 
            operation_ids = [operation.operation_id for operation in self.operations]
            machine_ids = [machine.machine_id for machine in self.machines]
            X_values = {(i,k): self.model.getVarByName(f"x_{i}_{k}").X for i in operation_ids for k in machine_ids}
            S_values = {i: self.model.getVarByName(f"s_{i}").X for i in operation_ids}
            C_values = {i: self.model.getVarByName(f"c_{i}").X for i in operation_ids}
            cmax_value = self.model.getVarByName("cmax").X

            # Save results in the class 
            self.results = {
                "X": X_values,
                "S": S_values,
                "C": C_values,
                "cmax": cmax_value
            }
            return self.results
        else:
            print("No feasible solution found.")
            self.results = None
            return None

    
    def print_results(self): 
        """
        Print the results
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

        print("\n Machine Assignments (X[i,k]=1):")
        for (i, k), val in self.results["X"].items():
            if val > 0.5:
                print(f"   Operation {i} → Machine {k}")

    
    def update_environment(self): 
        if not self.results: 
            print("No solution available, run run_model().")

        X_values = self.results["X"]
        S_values = self.results["S"]
        C_values = self.results["C"]

        # 1. Update the operations
        for op in self._original_env.operations:
            i = op.operation_id

            # Find the assigned machine (X[i,k] = 1)
            assigned_machine = None
            for (ii, k), val in X_values.items():
                if ii == i and val > 0.5:
                    assigned_machine = k
                    break

            if assigned_machine is None:
                continue  # not assigned (this can happen for completed activities)

            start_time = S_values.get(i, None)
            end_time = C_values.get(i, None)
            if start_time is None or end_time is None:
                continue

            duration = end_time - start_time
            setup_time = 0  

            op.add_operation_scheduling_information(
                assigned_machine, start_time, setup_time, duration
            )

        # 2. Update machines 
        for machine in self._original_env.machines:
            machine._processed_operations = [
                op for op in self._original_env.operations if op.scheduled_machine == machine.machine_id
            ]

        # 3. Update the environment
        self._original_env._scheduled_operations = [
            op for op in self._original_env.operations if op.scheduling_information
        ]
        self._original_env._operations_to_be_scheduled = [
            op for op in self._original_env.operations if not op.scheduling_information
        ]

        self._original_env.update_operations_available_for_scheduling()

        print("Environment updated with the new schedule!")