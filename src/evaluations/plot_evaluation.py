import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def context_plot(plotter):
    def wrapper(*args, title="", x_label="cost_call", y_label="descent - sol", show=False, save=False, **kwargs):
        plotter(*args, **kwargs)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if save:
            plt.savefig(title)
        if show:
            plt.show()

    return wrapper


@context_plot
def plot_trajectories(runs):
    for run in runs:
        plt.semilogy(run)


@context_plot
def show_eaf(eaf):
    plt.imshow(eaf)


@context_plot
def plot_eaf(x, y, eaf, mode="", save_data=False):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if mode == "":
        x_plot = x
        y_plot = y
        eaf_plot = eaf

    elif mode == "y_log":
        x_plot = x
        y_plot = np.log(y)
        eaf_plot = eaf


    else:
        raise NotImplementedError("mode to be implemented")

    surf = ax.plot_surface(x_plot, y_plot, eaf_plot,
                           linewidth=0, cmap=cm.coolwarm, antialiased=False)

    # ax.set_yscale("log")
    fig.colorbar(surf)
    if save_data:
        with open("last_data", "w") as file:
            file.write(str(x) + "\\")
            file.write(str(y) + "\\")
            file.write(str(eaf))


def plot_ert(x, y, eaf, level=0.95, mode="", show=True):
    """x: costs, y: distances_objective"""
    filter_level = eaf < level
    indices = np.sum(filter_level, axis=0) - 1
    indices = np.maximum(indices, 0)

    x_simple = x[0, :]
    assert len(x_simple) == len(indices), "Mismatch shapes"
    y_level = np.array([y[ind, i] for i, ind in enumerate(indices)])
    print(y_level)
    if mode == "":
        x_label = "Cost"
        y_label = "estimation - objective"

    elif mode == "y_log":
        y_level = np.log(y_level)
        x_label = "Cost"
        y_label = "log(estimation - objective)"


    if show:
        plt.plot(x_simple, y_level)

        plt.title(f"ert level: {level}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    else:
        plt.plot(x_simple, y_level, label=f"level={level}")
        plt.title(f"ECDF")
        plt.xlabel(x_label)
        plt.ylabel(y_label)


def plot_erts(x, y, eaf, levels=[0.95], mode=""):
    for level in levels:
        plot_ert(x, y, eaf, level=level, mode=mode, show=False)
    plt.legend()
    plt.show()


def plot_quantiles_budget(x, y, eaf, budgets, dimension=10):
    deltas = np.log(y[:, 0])
    if isinstance(budgets, int):
        probas = eaf[:, budgets - 1]
        plt.plot(deltas, probas)

        plt.title(f"Quantiles at budget={budgets}")
    else:
        for ind, budget in enumerate(budgets):
            if ind == 0:
                dim = dimension
                budget = dim * np.log(dim)
            probas = eaf[:, int(budget) - 1]
            plt.plot(probas, deltas, label=f"budget={int(budget)}")
        plt.legend()
        plt.title("Quantiles at given budgets")
    plt.xlabel("log(val - solution)")
    plt.ylabel("Probability")
    plt.show()