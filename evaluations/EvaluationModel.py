import evaluation_models as em
from optimizers.CoordinateGradientDescent import CoordinateGradientDescent as CGD
import plot_evaluation
from dataclasses import dataclass


@dataclass
class EvaluationModel:
    name: str
    path_data: str
    parameters: tuple
    hyperparameters: dict
    Algo: CGD
    n_runs: int = 100
    reset: bool = True

    @property
    def has_run(self):
        try:
            l_data: list = em.get_l_data()
            return self.name in l_data
        except NotImplementedError:
            return False

    @property
    def data(self):
        return em.get_data(self.name)

    def create_data(self):
        em.create_data(self.Algo, self.parameters, self.hyperparameters,
                       n_runs=self.n_runs, name=self.name, reset=self.reset)

    def plot_eaf(self, nb_steps_probability=50, save=True):
        costs, objectives, probabilities = em.create_eaf(self.data, nb_steps_probability)
        plot_evaluation.plot_eaf(costs, objectives, probabilities,
                                 title=f"Empirical Attainment Function", show=True, save=save)

    # title=f"Empirical Attainment Function\n{self.name}
    def show_trajectory(self):
        plot_evaluation.plot_trajectories(self.data,
                                          x_label="costs", y_label="val - solution",
                                          title=f"{self.name}",
                                          show=True)
