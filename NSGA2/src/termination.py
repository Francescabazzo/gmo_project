from pymoo.core.termination import Termination
from pymoo.termination import get_termination


class CombinedTermination(Termination):
    """
    Custom termination criterion that combines maximum generations
    and no-improvement conditions with OR logic.
    """

    def __init__(self, n_max_gen=500, ftol=0.01):
        super().__init__()
        self.n_max_gen = n_max_gen
        self.ftol = ftol

        # Crea le termination criteria usando la factory function
        self.max_gen_termination = get_termination("n_gen", n_max_gen)
        self.ftol_termination = get_termination("ftol", tol=ftol)

    def _update(self, algorithm):
        current_gen = algorithm.n_gen

        # Criterion 1: Maximum number of generations
        if self.max_gen_termination.has_terminated(algorithm):
            print(f"Stopped: Reached {current_gen} generations (max: {self.n_max_gen})")
            return True

        # Criterion 2: ftol improvement
        if self.ftol_termination.has_terminated(algorithm):
            print(f"Stopped: Improvement < {self.ftol * 100}%")
            return True

        # Log progress
        if current_gen % 20 == 0:
            print(f"Gen {current_gen}: Continuing...")

        return False