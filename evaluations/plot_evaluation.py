import matplotlib.pyplot as plt
from matplotlib import cm


def context_plot(plotter):
    def wrapper(*args, title="", x_label="cost_call", y_label="descent - sol", show=False, save=False):
        plotter(*args)
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
def plot_eaf(x, y, eaf):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, eaf,
                           linewidth=0, cmap=cm.coolwarm, antialiased=False)
    # ax.set_yscale("log")
    fig.colorbar(surf)