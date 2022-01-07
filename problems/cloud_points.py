import numpy as np


def generate_gaussian_cloud(mean: float, covariance: float, n_points: int = 10):
    return np.random.multivariate_normal(mean, covariance, n_points)


def generate_data_gaussian(zeros_parameters: list, ones_parameters: list):
    gen_zeros = [generate_gaussian_cloud(*parameters, **hyper_parameters)
                     for (parameters, hyper_parameters) in zeros_parameters]
    X_zeros = np.concatenate(gen_zeros)

    gen_ones = [generate_gaussian_cloud(*parameters, **hyper_parameters)
                    for parameters, hyper_parameters in ones_parameters]
    X_ones = np.concatenate(gen_ones)

    n_0, d = X_zeros.shape
    n_1, d_check = X_ones.shape
    assert d == d_check, "Misfit dimensions"

    data = np.zeros((n_0 + n_1, d + 1))
    data[:n_0, :d] = X_zeros
    data[n_0:, :d] = X_ones
    data[n_0:, -1] += 1
    return data


# Quicktest
if __name__ == '__main__':
    DIMENSION = 53
    zeros_parameters = [((np.ones(DIMENSION), np.eye(DIMENSION)), {"n_points": 10}),
                        ((np.ones(DIMENSION) * 3, np.eye(DIMENSION)), {"n_points": 10})]
    ones_parameters = [((-1 * np.ones(DIMENSION), np.eye(DIMENSION)), {"n_points": 10}),
                       ((-3 * np.ones(DIMENSION), np.eye(DIMENSION)), {"n_points": 10})]

    data = generate_data_gaussian(zeros_parameters, ones_parameters)
