import matplotlib.pyplot as plt
import numpy as np


def generate_grid(func, center, radius=10, resolution=1000):
    """Create a grids for 3d contouring as well as contours of a function"""
    x = np.linspace(center[0] - radius, center[0] + radius, resolution)
    y = np.linspace(center[1] - radius, center[1] + radius, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    return X, Y, Z


def levels_func(func, center=np.array([0, 0]), radius=10, resolution=1000, nb_phases=5,  cmap="Reds"):
    """Create the graph of contours of a function"""
    X, Y, Z = generate_grid(func, center, radius, resolution)
    graph = plt.contourf(X, Y, Z, nb_phases, cmap=cmap)
    plt.colorbar()
    plt.title("function visualization")

    return None


def get_xy(l_parameters):
    """Gets the list of abscisses and ordonates."""
    x, y = [], []
    for point in l_parameters:
        x.append(point[0][0])
        y.append(point[0][1])
    return x, y


def get_center_radius(x, y):
    """Compute the center and the radius of the points according to the list of abscisses and ordonates"""
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    point_min = np.array([x_min, y_min])
    point_max = np.array([x_max, y_max])
    center = (point_min + point_max) / 2

    radius_x = 2 * (center[0] - x_min)
    radius_y = 2 * (center[1] - y_min)
    radius = max(radius_x, radius_y)
    return center, radius


def visualize_descent(func,
                      l_parameters,
                      resolution=100,
                      cmap_levels="Reds",
                      color_trajectory="blue",
                      title="Visualisation"):
    """Trace the evolution of the descent of a R^2 -> R function
     --------------
     Parameters:
         func: function np.array([x, y]) of R^2 to R

         l_parameters: list of points of the descent

         resolution: resolution to build the colors of the graph

         cmap_levels: string for the levels

         color_trajectory: color of the descent

         title: string for the plot title
         """
    xy = get_xy(l_parameters)
    x, y = xy[0], xy[1]
    center, radius = get_center_radius(x, y)
    levels_func(func, center=center, radius=radius, resolution=resolution, cmap=cmap_levels)
    # plt.plot(x, y, color=color_trajectory)
    plt.title(title)
    plt.savefig("Lignes de niveau")
    plt.show()


if __name__ == '__main__':
    from problems import linear_regression

    func = linear_regression.linear_application
    levels_func(func)
    plt.show()


