import numpy as np
import visualization2D as visu
import matplotlib.pyplot as plt


class CoordinateGradientDescent:
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
        self.mode = mode
        if self.mode == "permutation":
            self.permutation = []
            self.generate_permutation()

    def generate_permutation(self):
        """Generates a permutation of S(d) where d is number of dimensions of the parameters"""
        self.permutation = list(range(self.nb_dimensions))
        np.random.shuffle(self.permutation)

    def set_values(self):
        """Compute the value of the funciton and the value of the derivative at the point of the parameters"""
        self.value = self.function(self.parameters)
        self.derivative = self.d_function(self.parameters)
        self.cost_call += 1
        self.cost_call_derivative += 1

    def change_index(self):
        """Change index according to the mode of the optimizer"""
        if self.mode == "random":
            self.index_descent = np.random.randint(0, self.nb_dimensions)
        elif self.mode == "cycle":
            self.index_descent = (self.index_descent + 1) % self.nb_dimensions
        elif self.mode == "permutation":
            if self.permutation:
                self.generate_permutation()
            self.index_descent = self.permutation.pop()
        else:
            raise NotImplementedError

    def descent_one_step(self):
        """Does a step in the descent according to the index of the next self.change_index"""
        self.change_index()
        self.set_values()
        if self.create_history:
            self.L_value.append(self.value)
            self.L_parameters.append(self.parameters)

        if self.lr_def == "adaptative_index":
            lr = np.array(self.learning_rate(self.index_descent), dtype="float64")
        else:
            lr = np.array(self.learning_rate(-1), dtype="float64")
        derivative = np.array(self.derivative[self.index_descent], dtype="float64")

        self.parameters[self.index_descent] -= lr * derivative

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

        plt.title(f'Coordinate descent, mode={self.mode}, learning_rate={self.lr_def}')
        plt.savefig("{}_mode".format(self.mode))
        plt.show()

    def visualize_descent(self,
                          resolution=1000,
                          cmap_levels="Reds",
                          color_trajectory="blue"):
        """Plots for a 2D problem the path of the descent around the solution"""
        visu.visualize_descent(self.function,
                               self.L_parameters,
                               resolution=resolution,
                               cmap_levels="Reds",
                               color_trajectory="blue",
                               title=f"Visualisation {self.mode} coordinate descent")
        plt.show()

        return None


