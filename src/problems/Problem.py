class Problem:
    def __init__(self, origine, **hyper_parameters):
        self.function = self.create_function(**hyper_parameters)
        self.df = self.create_derivative(**hyper_parameters)
        self.solution = self.create_solution(**hyper_parameters)
        self.origine = origine
        self.solution_hat = origine

    def create_function(self, **hyper_parameters):
        """Must return a function taking 1 argument, the point on which we want the valor of the model"""
        def primal_function(W):
            pass
        raise NotImplementedError

    def create_derivative(self, **hyper_parameters):
        """Must return a function taking 1 or 2 arguments for the parameter
        and the possible constraint parameter (see Lagrangian) """
        def d_f(W, alpha, **kwargs):
            pass
        raise NotImplementedError

    def create_solution(self, **hyper_parameters):
        """Must return a function taking no argument and returning a
        tuple of the solution and the valor of the function at the solution"""
        def solution():
            pass
        raise NotImplementedError


