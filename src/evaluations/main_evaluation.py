import optimizers.CoordinateGradientDescent as CGD

import evaluation_models
import plot_evaluation
from EvaluationModel import EvaluationModel


# evaluation_models = evaluations.evaluation_models
# plot_evaluation = evaluations.plot_evaluation
# EvaluationModel = evaluations.EvaluationModel.EvaluationModel


def main():
    FUNC, D_FUNC, F_SOLUTION, OPTIMIZER_COORDINATE, X_0, LEARNING_RATE = evaluation_models.define_problem()
    MODEL = CGD.CoordinateGradientDescent
    N_RUNS = 100
    PRECISION_DELTAS = 1000
    MODEL_PARAMETERS = FUNC, D_FUNC, F_SOLUTION, X_0

    MODEL_HYPERPARAMETERS = {
         "learning_rate": LEARNING_RATE,
         "epochs_max": 10000,
         "mode": 'random',
         "lr_def": 'maxi'}

    evaluator = EvaluationModel("Coordinate_Gradient_Descent", "", MODEL_PARAMETERS, MODEL_HYPERPARAMETERS, MODEL)
    evaluator.n_runs = N_RUNS
    evaluator.precision_deltas = PRECISION_DELTAS

    TO_BE_RUN = {
        "create_data": True,
        "eaf": True
         }

    if TO_BE_RUN["create_data"]:
        evaluator.create_data()

    if TO_BE_RUN["eaf"]:
        evaluator.plot_eaf(mode="y_log")
    LEVELS = [0.1, 0.5,  0.95, .99]
    evaluator.plot_erts(levels=LEVELS, mode="y_log")
    PRECISION_BUDGET = 5
    evaluator.plot_budget_slices(PRECISION_BUDGET)


if __name__ == "__main__":
    main()