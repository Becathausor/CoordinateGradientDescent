import numpy as np
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

    runs = [list(map(float, run_str.split(" "))) for run_str in runs_str]

    return runs


def create_data(Algo, Parameters, Hyperparameters, name="", n_runs=100, reset=True):
    """Crée le fichier de données en appliquant l'algorithme Algo dans 'data/name.txt'

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
        try:
            erase_file(f"data\\{name}.txt")
        except FileNotFoundError:
            pass
    for k in range(n_runs):
        model = Algo(*Parameters, **Hyperparameters)
        model.optimize()
        register_data(model.L_value, title=f"data\\{name}.txt")


def erase_file(filename):
    """Efface le fichier de nom filename"""
    os.remove("data\\{filename}.txt")
    raise NotImplementedError


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

    cost_max = max([len(run) for run in runs])
    ert_tranche = np.zeros(int(cost_max))

    n = len(runs)

    for run in runs:
        qualities = run
        costs = list(range(1, len(run) + 1))
        ert_run = np.array(list(map(lambda x: x < delta, qualities)))
        if len(ert_run) < cost_max:
            ert__run = np.zeros(int(cost_max))

            # Rallongement de la liste
            ert__run[:len(ert_run)] = ert_run
            ert__run[len(ert_run):] = ert_run[-1]
            ert_run = ert__run

        # Ajout des échantillons bien trouvés
        ert_tranche += ert_run
    ert_tranche /= n
    return ert_tranche


def create_eaf(runs, nb_steps_probability=50):
    """Crée une eaf"""
    quality_min = min([min(run) for run in runs])
    quality_max = max([max(run) for run in runs])

    deltas = np.linspace(quality_min, quality_max, nb_steps_probability)
    eaf = np.array(list(map(lambda d: create_ert(runs, d), deltas)))
    costs = np.arange(len(eaf[0]))
    qualities = deltas

    x, y = np.meshgrid(costs, qualities)

    return x, y, eaf
