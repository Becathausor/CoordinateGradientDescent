import numpy as np
import sys

sys.path.append('C:\\Users\\thier\\Documents\\Telecom\\3A\\Projet\\CoordinateGradientDescent\\problems')
from problems import linear_regression
import os


def register_data(l_data, title=".txt"):
    """Enregistre les données d'un run en une ligne dans un fichiel texte du nom de title
    """
    line = " ".join(map(str, l_data))
    with open(title, "a") as file:
        file.write(line)
        file.write("\n")


def get_l_data():
    """Récupère la liste des fichiers de données"""
    raise NotImplementedError


def get_data(filename):
    """ Récupère les données d'un fichier texte en une liste de listes de runs"""
    with open(f"data\\{filename}.txt", "r") as file:
        runs_str = file.readlines()
    for k, line in enumerate(runs_str):
        if not k < len(runs_str):
            runs_str[k] = line[:-1]

    runs_moche = [list(map(float, run_str.split(" "))) for run_str in runs_str]
    nb_runs = len(runs_str)
    cost_max = len(runs_moche[0])
    runs = np.zeros((nb_runs, cost_max))
    for ind, run in enumerate(runs_moche):
        runs[ind] = np.array(run)
    print(runs.shape)

    return runs


def create_data(Algo, Parameters, Hyperparameters, name="", n_runs=100, reset=True):
    """Crée le fichier de données en appliquant l'algorithme Algo dans 'data\\name.txt'

    Parameters:
        Algo: Model

        Parameters: tuple
            Paramètres pour définir le problème étudié l'algo

        Hyperparameters: dictionary
            Dictionnaire des hyperparamètres pour l'entrainement

        name: str
            Nom du programme à tester

        n_runs: int
            Nombre de runs à effectuer

        reset: bool
            Réinitialise les données alors effectuées

        """
    if reset:
        erase_file(f"data\\{name}.txt")
        print("Has erased")

    for k in range(n_runs):
        model = Algo(*Parameters, **Hyperparameters)
        model.optimize()
        register_data(np.array(model.L_value) - model.f_solution()[1], title=f"data\\{name}.txt")


def erase_file(filename):
    """Efface le fichier de nom filename"""
    os.remove(filename)


def create_model(Model, args, kwargs):
    """ Génère un modèle"""
    return Model(*args, **kwargs)


def define_problem():
    """ Récupère les données du problème, par défaut le problème linéaire"""
    return linear_regression.linear_application, \
           linear_regression.d_linear_application, \
           linear_regression.solution, \
           linear_regression.linear_optimize_coordinate, \
           linear_regression.X_0, \
           lambda k=-1: 1 / linear_regression.constante_lipschitz(k)


def fill_rectangle(tab, coordinates):
    """ Remplit les rectangle dominé par une solution de coordonnées"""
    x, y = coordinates
    tab[x:, y:] += 1


def create_ert(runs, delta):
    """ Crée une ert"""

    n, cost_max = runs.shape
    ert_tranche = np.zeros(cost_max)

    costs = np.arange(1, cost_max + 1)
    ert_tranche = np.mean(runs < delta, axis=0)
    return ert_tranche


def create_eaf(runs, nb_steps_probability=50):
    """Crée une eaf"""
    check = True
    if check:
        print("Create EAF")
    quality_min = np.min(runs)
    quality_max = np.max(runs)
    nb_runs, cost_max = runs.shape

    deltas = np.exp(np.linspace(np.log(quality_min), np.log(quality_max), nb_steps_probability))
    eaf = np.zeros((nb_steps_probability, cost_max))

    if check:
        print(f"cost_max: {cost_max}")
        print(f"deltas: {deltas.shape}")
        print(f"eaf: {eaf.shape}")

    for ind, delta in enumerate(deltas):
        eaf[ind, :] = create_ert(runs, delta)

    costs = np.arange(1, cost_max + 1)

    X, Y = np.meshgrid(costs, deltas)
    if check:
        print(f"x: {X.shape}")
        print(f"y: {Y.shape}")
        print("End create eaf")
    return X, Y, eaf

