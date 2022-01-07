import numpy as np
from Problem import Problem
import sklearn as sk


def linear_svm(*args, **kwargs):
    return sk.svm.SVC(*args, kernel="linear", **kwargs)


class Linear_SVM(Problem):
    def __init__(self, origine, **hyper_parameters):
        super().__init__(origine, **hyper_parameters)
        self.data = hyper_parameters["data"]
        self.features = self.data[:, :-1]
        self.labels = self.data[:, -1]
        self.model = linear_svm()

    def create_function(self, **hyper_parameters):
        """Create the function of the problem according to the extended vector of the hyperplane"""
        def distance_hyperplan(data_line, w, w0):
            label = data_line[-1]
            feature = data_line[:-1]
            return label * (np.dot(w, feature) + w0)

        def primal_prediction(W, *args, **kwargs):
            w = W[:len(self.origine)]
            w0 = W[len(self.origine)]
            return np.array(list(
                map(lambda data_line: distance_hyperplan(data_line, w, w0), self.data)
                ))

        return primal_prediction

    def create_derivative(self, **hyper_parameters):
        """ Create the derivative of the lagrangian problem"""
        def d_lagrangian(W, alpha, *args, **kwargs):
            w, w0 = W[:-1], W[-1]
            d_w = w - alpha * self.labels * self.features
            d_w0 = -alpha * self.labels
            d_alpha = self.labels * (np.dot(self.features, w.T) + w0)
            result = np.zeros(len(W) + len(self.data))
            result[:len(W)] = d_w
            result[len(W)] = d_w0
            result[len(W) + 1:] = d_alpha

            return result

    def create_solution(self, **hyper_parameters):
        self.model.fit(self.features, self.labels)

        def solution():
            return self.model.get_params(), self.model.score(self.features)
