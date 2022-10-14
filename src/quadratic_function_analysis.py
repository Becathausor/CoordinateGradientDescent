import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import cm


DIM = 10
np.random.seed(1)
A = np.random.uniform(-10, 10, (DIM))
X_0 = np.zeros(DIM)
ALPHA = 1


def func(x):
    return (x ** 2).T @ A / 2


def d_func(x):
    return A * x


def loi_descente(d, n):
    law = np.zeros((d + 1, n))
    law[0, 0] = 1
    for j in range(1, n):
        for i in range(1, d + 1):
            law[i, j] = (i / d) * law[i, j - 1] + (1 - (i - 1) / d) * law[i - 1, j - 1]
    plt.imshow(law, cmap="Reds")

    plt.xlabel("Iterations")
    plt.ylabel("Nombre de coordonnées annulées")
    plt.title("Law of the coordinate descent of a separable function")
    plt.savefig("Separable function")
    plt.show()
    return law


def simulate_optimization(d_range, n_simulation, raw=False):
    D = len(d_range)
    simulations = np.zeros((D, n_simulation))
    for index_d, d in enumerate(d_range):
        print(d)

        for k in range(d):
            simulations[index_d, :] += np.random.geometric(1 - k / d, size=n_simulation)


    return simulations


def estimate_moments(simulations):
    times = np.mean(simulations, axis=1)
    simu_car = simulations ** 2
    variances = np.mean(simu_car, axis=1) - times ** 2

    return times, variances


def quantiles(runs, nb_steps_probability):
    """Compute the quantiles of a simulation through an eaf"""

    deltas = np.linspace(0, 1 - 1 / nb_steps_probability, nb_steps_probability)
    print(deltas)
    sorted_runs = np.sort(runs, axis=1)
    quant = np.zeros((runs.shape[0], nb_steps_probability))

    for ind, delta in enumerate(deltas):
        seuil_ind = int(runs.shape[1] * delta)

        quant[:, ind] = sorted_runs[:, seuil_ind]

    return quant, deltas


def plot_simulation_attainment(d_range, n_simulation, p_confidence=0.95, simulations=np.array([]), save=True):
    
    if simulations.any():
        times, variances = estimate_moments(simulations)
    else:
        times, variances = estimate_moments(simulate_optimization(d_range, n_simulation))
    # SUPPOSITION LOI NORMALE
    quantile = - stats.norm.ppf((1 - p_confidence) / 2)
    delta = (variances ** 0.5) * quantile
    lower_estimation = times - delta
    higher_estimation = times + delta

    plt.plot(d_range, times, label="mean")
    plt.plot(d_range, lower_estimation, label="lower_bound")
    plt.plot(d_range, higher_estimation, label="upper_bound")

    plt.legend()
    plt.xlabel("Dimension")
    plt.ylabel("Annulation time")
    plt.title("Law of the annulation time in the separable case")

    if save:
        plt.savefig("Full annulation time along dimension ")
    plt.show()
    return times, variances


def plot_quantiles(dimensions, quants, deltas, title=""):
    x, y = np.meshgrid(deltas, dimensions)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    print(x.shape)
    print(y.shape)
    print(quants.shape)

    surf = ax.plot_surface(x, y, quants,
                           linewidth=0, cmap=cm.coolwarm, antialiased=False)

    fig.colorbar(surf)
    plt.title(title)
    plt.ylabel("Dimension")
    plt.xlabel("Probability")

    plt.show()


def mean_theory(d_range):
    p = np.arange(1, d_range) / d_range
    return np.sum(1 / (1 - p))


def variance_theory(d_range):
    p = np.arange(1, d_range) / d_range
    q = 1 - p
    return np.sum(p/q)


if __name__ == "__main__":
    DIM = 100
    N_MAX = 5 * DIM
    N_SIMULATION = 10000
    # D_RANGE = 100 * (10 ** np.arange(5))
    D_RANGE = []
    POWER_MAX = 4
    for power in range(POWER_MAX):
        for i, base_elem in enumerate([100, 200, 500, 800]):
            D_RANGE.append(base_elem * 10**power)
    D_RANGE.append(10 ** POWER_MAX * 100)
    print(D_RANGE)

    TO_BE_RUN = {"Theory vs Experimental moments": True,
                 "Quantiles": True
                 }

    # Generate runs and initializes the law
    simulations = simulate_optimization(D_RANGE, N_SIMULATION)
    law = loi_descente(DIM, N_MAX)

    if TO_BE_RUN["Theory vs Experimental moments"]:
        simulation_res = plot_simulation_attainment(D_RANGE, N_SIMULATION, p_confidence=0.95, simulations=simulations)
        # Theoretic trajectories with a gaussian behaviour
        mean_theory_ = np.array([mean_theory(d) for d in D_RANGE])
        plt.plot(D_RANGE, mean_theory_, label="Mean")

        # Plot
        var_theory_ = np.array([variance_theory(d) for d in D_RANGE])
        plt.plot(D_RANGE, mean_theory_ + 2 * var_theory_ ** 0.5, label="Upper-bound")
        plt.plot(D_RANGE, mean_theory_ - 2 * var_theory_ ** 0.5, label="Upper-Bound")
        plt.legend()
        plt.xlabel("Dimension")
        plt.ylabel("Annulation time")
        plt.show()

    if TO_BE_RUN["Quantiles"]:
        # Computation of quantiles
        quant, deltas = quantiles(simulations, 40)
        plot_quantiles(D_RANGE, quant, deltas, title="Quantiles quadratic case")
