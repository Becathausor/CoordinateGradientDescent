from optimizers import CoordinateGradientDescent
from problems import linear_regression


def main():
    FUNC = linear_regression.linear_application
    D_FUNC = linear_regression.d_linear_application
    F_SOLUTION = linear_regression.solution
    OPTIMIZER_COORDINATE = linear_regression.linear_optimize_coordinate

    L_LR_DEF = ['adaptative_index', 'maxi']
    MODEL_HYPERPARAMETERS = {"learning_rate": lambda index=-1: 1 / linear_regression.constante_lipschitz(index),
                             "epochs_max": 1000,
                             "mode": 'random',
                             "lr_def": 'maxi'}

    X_0 = linear_regression.X_0

    model = CoordinateGradientDescent.CoordinateGradientDescent(FUNC, D_FUNC, F_SOLUTION, X_0, **MODEL_HYPERPARAMETERS)
    model.optimize()
    model.plot_history()

    # Que faut-il run/afficher ?
    TO_BE_RUN = {"Conditionnement": True,
                 "Problème étudié": True,
                 "Descente de Gradient": True,
                 "Solution théorique": True,
                 "Visualisation": False}

    if TO_BE_RUN["Conditionnement"]:
        print("CONDITIONNEMENT")
        print(linear_regression.conditionnement())

    if TO_BE_RUN["Problème étudié"]:
        print("PROBLEME")
        print(f"Application: {linear_regression.N}")
        print(f"Objective: {linear_regression.Y}")

    if TO_BE_RUN["Descente de Gradient"]:
        print("DESCENTE DE GRADIENT")
        print(f"Descente: {model.L_value}")
        print(f"Solution: {model.parameters}")
        print(f"minimum: {model.L_value[-1]}")

    if TO_BE_RUN["Solution théorique"]:
        print("THÉORIE")
        sol, mini = linear_regression.solution()
        print(f"Solution: {sol}")
        print(f"minimum: {mini}")

        sol, mini = linear_regression.solution()
        print("CONDITIONNEMENT")
        if model.L_value[-1] - mini > 0.1:
            print("Erreurs importantes")
        else:
            print("Apparemment une bonne descente")
        print(linear_regression.conditionnement())

    if TO_BE_RUN["Visualisation"]:
        print("VISUALISATION")
        model.visualize_descent()

    # TODO: To be debugged
    # print("DESCENTE OPTIMISÉE")
    # model_opti = optimizers.CoordinateDescent(FUNC, OPTIMIZER_COORDINATE, F_SOLUTION, X_0,
    #                                           epochs_max=1000, mode='random')
    # model_opti.optimize()
    # model_opti.plot_history()


if __name__ == '__main__':
    main()