import numpy as np
import visualization2D as visu
import matplotlib.pyplot as plt


class GradientDescent:
    def __init__(self,
                 func,
                 df,
                 f_solution,
                 x_0,
                 learning_rate,
                 epochs_max=100,
                 mode='random',
                 lr_def="adaptative_index"):
        self.function = func
        self.d_function = df
        self.f_solution = f_solution
        self.parameters = np.array(x_0)

        self.learning_rate = learning_rate
        self.lr_def = lr_def

        self.value = self.function(self.parameters)
        self.derivative = self.d_function(self.parameters)

        self.nb_dimensions = len(self.parameters)
        self.index_descent = -1

        self.epochs_max = epochs_max

        self.cost_call = 0
        self.cost_call_derivative = 0

        self.create_history = True
        self.L_value = []
        self.L_parameters = []

    def set_values(self):
        """Compute the value of the funciton and the value of the derivative at the point of the parameters"""
        self.value = self.function(self.parameters)
        self.derivative = self.d_function(self.parameters)
        self.cost_call += 1
        self.cost_call_derivative += 1

    def descent_one_step(self):
        """Does a step in the descent according to the index of the next self.change_index"""
        self.set_values()
        if self.create_history:
            self.L_value.append(self.value)
            self.L_parameters.append(self.parameters)

        lr = np.array(self.learning_rate(-1), dtype="float64")
        self.parameters -= lr * self.derivative

    def optimize(self):
        """Operates epochs_max steps of the descent"""
        for epoch in range(self.epochs_max):
            self.descent_one_step()
        check = False
        if check:
            print("Check in CoordinateGradientDescent => optimize")
            print(self.value, self.parameters)
        return self.value, self.parameters

    def plot_history(self, scale="log_scale"):
        """Plots the distance to the solution during the descent with y-axis in log_scale by default"""

        x_sol, val_sol = self.f_solution()
        if scale == "log_scale":
            plt.semilogy(np.array(self.L_value) - val_sol)
            plt.xlabel('cost_call')
            plt.ylabel('value-val_sol')

        else:
            plt.plot(self.L_value)
            plt.xlabel('cost_call')
            plt.ylabel('value')

        plt.title(f'Coordinate descent, learning_rate={self.lr_def}')
        plt.savefig("GradientDescent, d={}".format(self.nb_dimensions))
        plt.show()




