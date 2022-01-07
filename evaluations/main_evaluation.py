import evaluation_models
import optimizers.CoordinateGradientDescent as CGD
import plot_evaluation
from EvaluationModel import EvaluationModel


def main():
    FUNC, D_FUNC, F_SOLUTION, OPTIMIZER_COORDINATE, X_0, LEARNING_RATE = evaluation_models.define_problem()
    MODEL = CGD.CoordinateGradientDescent

    MODEL_PARAMETERS = FUNC, D_FUNC, F_SOLUTION, X_0

    MODEL_HYPERPARAMETERS = {
         "learning_rate": LEARNING_RATE,
         "epochs_max": 100,
         "mode": 'random',
         "lr_def": 'maxi'}

    evaluator = EvaluationModel("Coordinate_Gradient_Descent", "", MODEL_PARAMETERS, MODEL_HYPERPARAMETERS, MODEL)

    TO_BE_RUN = {
        "create_data": True,
        "eaf": True
         }

    if TO_BE_RUN["create_data"]:
        evaluator.create_data()

    if TO_BE_RUN["eaf"]:
        evaluator.plot_eaf()


if __name__ == "__main__":
    main()