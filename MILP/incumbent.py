import gurobipy as gb 
import time

class Incumbent:
    def __init__(self):
        self.times = []     
        self.sol_list = []   

def add_current_sol(model: gb.Model, where, incumbent_obj):
    if where == gb.GRB.Callback.MIPSOL:
        if len(incumbent_obj.sol_list) == 0 or incumbent_obj.sol_list[-1] > model.cbGet(gb.GRB.Callback.MIPSOL_OBJ):
            incumbent_obj.sol_list.append(model.cbGet(gb.GRB.Callback.MIPSOL_OBJ))
            incumbent_obj.times.append(time.time() - model._start_time)