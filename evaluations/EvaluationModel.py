import evaluation_models as em
from optimizers.CoordinateGradientDescent import CoordinateGradientDescent as CGD
import plot_evaluation
from dataclasses import dataclass
import numpy as np


@dataclass
class EvaluationModel:
    name: str
    path_data: str
    parameters: tuple
    hyperparameters: dict
    Algo: CGD
    n_runs: int = 100
    reset: bool = True
    precision_deltas: int = 1000
    dimension: int = 10

    @property
    def has_run(self):
        try:
            l_data: list = em.get_l_data()
            return self.name in l_data
        except NotImplementedError:
            return False

    @property
    def data(self):
        print(self.name)
        return em.get_data(self.name)

    def create_data(self):
        em.create_data(self.Algo, self.parameters, self.hyperparameters,
                       n_runs=self.n_runs, name=self.name, reset=True)

    def plot_eaf(self, mode="", save=True):
        costs, objectives, probabilities = em.create_eaf(self.data, self.precision_deltas)
        if mode == "y_log":
            y_label = "log(val - sol)"
        else:
            y_label = "val - sol"
        plot_evaluation.plot_eaf(costs, objectives, probabilities, mode=mode,
                                 title=f"Empirical Attainment Function", y_label=y_label, show=True, save=save)

    def show_trajectory(self):
        plot_evaluation.plot_trajectories(self.data,
                                          x_label="costs", y_label="val - solution",
                                          title=f"{self.name}",
                                          show=True)

    def plot_ert(self, level=0.95, mode=""):
        costs, objectives, probabilities = em.create_eaf(self.data, self.precision_deltas)
        plot_evaluation.plot_ert(costs, objectives, probabilities, level, mode=mode)

    def plot_erts(self, levels=[0.95], mode=""):
        costs, objectives, probabilities = em.create_eaf(self.data, self.precision_deltas)
        plot_evaluation.plot_erts(costs, objectives, probabilities, levels, mode=mode)

    def plot_budget_slices(self, precision_budget):

        self.dimension = len(self.parameters[3])

        max_budget = self.hyperparameters["epochs_max"]
        budgets = np.linspace(1, max_budget, precision_budget)
        costs, objectives, probabilities = em.create_eaf(self.data, self.precision_deltas)
        plot_evaluation.plot_quantiles_budget(costs, objectives, probabilities, budgets=budgets,
                                              dimension=self.dimension)
