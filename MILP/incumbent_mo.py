import gurobipy as gb
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

from sympy.printing.pretty.pretty_symbology import line_width


class Incumbent:

    def __init__(self):
        self.phase1 = {
            'times': [],
            'values': []
        }
        self.phase2 = {
            'times': [],
            'values': []
        }
        self.phase3 = {
            'times': [],
            'values': []
        }
        self.current_phase = None
        self.phase_start_time = None

    def reset(self):
        self.phase1.clear()
        self.phase2.clear()
        self.phase3.clear()
        self.current_phase = None
        self.phase_start_time = None

    def start_phase(self, phase_num: int):
        if phase_num in [1, 2, 3]:
            self.current_phase = phase_num
            self.phase_start_time = time.time()
        else:
            raise ValueError("phase_num must be in [1,2,3].")

    def add_solution(self, obj_value: float):
        if self.current_phase is None or self.phase_start_time is None:
            raise ValueError("Attention: no active phase, so not possible to add solution.")

        phase_elapsed = time.time() - self.phase_start_time

        if self.current_phase == 1:
            times_list = self.phase1['times']
            values_list = self.phase1['values']
        elif self.current_phase == 2:
            times_list = self.phase2['times']
            values_list = self.phase2['values']
        elif self.current_phase == 3:
            times_list = self.phase3['times']
            values_list = self.phase3['values']
        else:
            return

        # Check if there is an improvement from the previous solution
        is_improvement = False

        # First solution of the phase
        if not values_list:
            is_improvement = True
            # Improvement from the last value
        elif obj_value < values_list[-1]:
            is_improvement = True

        # Add only if there is an improvement
        if is_improvement:
            times_list.append(phase_elapsed)
            values_list.append(obj_value)

    def get_phase_data(self, phase_num: int):
        if phase_num == 1:
            return self.phase1
        elif phase_num == 2:
            return self.phase2
        elif phase_num == 3:
            return self.phase3
        else:
            raise ValueError("Phase number must be 1, 2, or 3.")

    def plot_phase1(self, save_path: Optional[str] = None):
        self._plot_single_phase(
            phase_number=1,
            data=self.phase1,
            title="Phase 1: Optimization of the quadratic delay",
            ylabel="Quadratic delay",
            color="blue",
            save_path=save_path
        )

    def plot_phase2(self, save_path: Optional[str] = None):
        self._plot_single_phase(
            phase_number=2,
            data=self.phase2,
            title="Phase 2: Optimization of the machine assignment changes",
            ylabel="Sum of the machine assignment changes",
            color="green",
            save_path=save_path
        )

    def plot_phase3(self, save_path: Optional[str] = None):
        self._plot_single_phase(
            phase_number=3,
            data=self.phase3,
            title="Phase 3: Optimization of the cmax",
            ylabel="Cmax",
            color="violet",
            save_path=save_path
        )

    def _plot_single_phase(self, phase_number: int, data: Dict, title: str, ylabel: str, color: str,
                           save_path: Optional[str]=None):
        """
        Helper method to plot a single plot.

        Args:
            phase_num: phase number
            data: phase data
            title: graph title
            ylabel: axis y label
            color: line color
            save_path: path where to save
        """
        if not data['times'] or not data['values']:
            print(f"No data for phase {phase_number}")
            return

        plt.figure(figsize=(12,6))

        # Main plot
        plt.plot(data['times'], data['values'], color=color, linewidth=2.5,
                 marker='o', markersize=6, markerfacecolor=color, markeredgecolor='dark'+color, markeredgewidth=1)

        # Line to indicate the final value
        final_value=data['values'][-1]
        plt.axhline(y=final_value, color='red', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Final value: {final_value: .2f}')

        # Graph configuration
        plt.xlabel('Time of the phase (seconds)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, loc='upper right')

        plt.tight_layout()

        if save_path:
            if save_path.endswith('/') or save_path.endswith('\\'):
                save_path = f"{save_path}_phase{phase_number}_incumbent.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')


    def plot_all_phases(self, save_dir: Optional[int]):
        save_path1=f"{save_dir}_phase1_incumbent.png"
        self.plot_phase1(save_path1)

        save_path2 = f"{save_dir}_phase2_incumbent.png"
        self.plot_phase2(save_path2)

        save_path3 = f"{save_dir}_phase3_incumbent.png"
        self.plot_phase3(save_path3)


def add_current_sol(model: gb.Model, where, incumbent_obj: Incumbent):
    if where == gb.GRB.Callback.MIPSOL:
        try:
            obj_value = model.cbGet(gb.GRB.Callback.MIPSOL_OBJ)
            incumbent_obj.add_solution(obj_value)
        except:
            pass