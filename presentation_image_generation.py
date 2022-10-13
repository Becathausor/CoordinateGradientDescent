import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def carr(x):
    return x ** 2


def dcarr(x):
    return 2 * x


def descent_gradient(x_0, function, d_function, learning_rate, n_max = 10):
    descent = np.zeros(n_max + 1)
    val_descent = np.zeros(n_max + 1)
    descent[0], val_descent[0] = x_0, function(x_0)
    for k in range(n_max):
        descent[k + 1] = descent[k] - learning_rate * d_function(descent[k])
        val_descent[k + 1] = function(descent[k + 1])

    return descent, val_descent


def plot_gradient_descent(x_0, learning_rate, precision):
    x_func = np.linspace(-10, 10, precision)
    y_func = x_func ** 2
    plt.plot(x_func, y_func)

    x_des, y_des = descent_gradient(x_0, carr, dcarr, learning_rate)

    print(x_des, y_des)
    plt.plot(x_des, y_des)
    plt.show()


def plot_large_surface():
    x = np.linspace(-20, 20, 1000)
    y = np.linspace(-20, 20, 1000)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z,
                           linewidth=0, cmap=cm.coolwarm, antialiased=False)

    # ax.set_yscale("log")
    fig.colorbar(surf)
    plt.show()


def main():
    X_0 = 10
    LEARNING_RATE = 0.7
    PRECISION = 1000

    plot_gradient_descent(X_0, LEARNING_RATE, PRECISION)
    plot_large_surface()

    pass

if __name__ == '__main__':
    main()


