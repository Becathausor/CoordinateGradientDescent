from CoordinateGradientDescent import CoordinateGradientDescent


class CoordinateDescent(CoordinateGradientDescent):
    def __init__(self,
                 func,
                 optimize_coordinate,
                 f_solution,
                 x_0,
                 learning_rate=0.01,
                 epochs_max=100,
                 mode='random'):
        super().__init__(self, func, lambda _: _, f_solution, x_0, epochs_max=epochs_max, mode=mode)
        self.coordinate_optimizer = optimize_coordinate

    def descent_one_step(self):
        self.change_index()
        self.set_values()
        if self.create_history:
            self.L_value.append(self.value)
            self.L_parameters.append(self.parameters)
        self.parameters[self.index_descent] = self.coordinate_optimizer(
            self.parameters, self.index_descent)[self.index_descent]
