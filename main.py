from optimizers import CoordinateGradientDescent, GradientDescent
import numpy as np


def main():
    # problem_choice = input("Choose between the Linear Reression (1) or the SVM (2): \n")
    problem_choice = 1
    if problem_choice == "2":
        from problems import linear_svm as problem
        FUNC = problem.primal_function
        D_FUNC = problem.d_lagrangian
        F_SOLUTION = problem.solution
        # OPTIMIZER_COORDINATE = problem.linear_optimize_coordinate

        L_LR_DEF = ['adaptative_index', 'maxi']
        MODEL_HYPERPARAMETERS = {"learning_rate": lambda index=-1: 1 / problem.constante_lipschitz(index),
                                 "epochs_max": 10000,
                                 "mode": 'random',
                                 "lr_def": 'adaptative_index'}

        X_0 = problem.X_0
    else: # Linear Regression
        from problems import linear_regression as problem
        FUNC = problem.linear_application
        D_FUNC = problem.d_linear_application
        F_SOLUTION = problem.solution
        # OPTIMIZER_COORDINATE = problem.linear_optimize_coordinate

        L_LR_DEF = ['adaptative_index', 'maxi']
        MODEL_HYPERPARAMETERS = {"learning_rate": lambda index=-1: 1 / problem.constante_lipschitz(index),
                                 "epochs_max": 10000,
                                 "mode": 'random',
                                 "lr_def": 'adaptative_index'}

        X_0 = problem.X_0

    coordinate = True

    if coordinate:

        model = CoordinateGradientDescent.CoordinateGradientDescent(FUNC, D_FUNC, F_SOLUTION, X_0, **MODEL_HYPERPARAMETERS)

    else:
        model = GradientDescent.GradientDescent(FUNC, D_FUNC, F_SOLUTION, X_0, **MODEL_HYPERPARAMETERS)
    model.optimize()
    model.plot_history()

    # Que faut-il run/afficher ?
    TO_BE_RUN = {"Conditionnement": False,
                 "Problème étudié": True,
                 "Descente de Gradient": True,
                 "Solution théorique": True,
                 "Visualisation": False}

    if TO_BE_RUN["Conditionnement"] and problem_choice != "2":
        print("CONDITIONNEMENT")
        print(problem.conditionnement())

    if TO_BE_RUN["Problème étudié"] and problem_choice != "2":
        print("PROBLEME")
        print(f"Application: {problem.N}")
        print(f"Objective: {problem.Y}")

    if TO_BE_RUN["Descente de Gradient"]:
        print("DESCENTE DE GRADIENT")
        print(f"Descente: {model.L_value}")
        print(f"Solution: {model.parameters}")
        print(f"minimum: {model.L_value[-1]}")

    if TO_BE_RUN["Solution théorique"]:
        print("THÉORIE")
        sol, mini = problem.solution()
        print(f"Solution: {sol}")
        print(f"minimum: {mini}")

        sol, mini = problem.solution()
        if problem_choice != "2":
            print("CONDITIONNEMENT")
            if model.L_value[-1] - mini > 0.1:
                print("Erreurs importantes")
            else:
                print("Apparemment une bonne descente")
            print(problem.conditionnement())

    if TO_BE_RUN["Visualisation"]:
        print("VISUALISATION")
        model.visualize_descent()

    if problem_choice == "2":
        param = model.parameters[:problem.DIM+1]
        w = param[:-1]
        w0 = param[-1]

        pred = problem.product_hadamard_(problem.LABELS, np.dot(problem.FEATURES, w) + w0)
        accuracy = np.sum(pred > 0.1) / len(pred)
        print(f"Accuracy: {accuracy}")


if __name__ == '__main__':
    main()