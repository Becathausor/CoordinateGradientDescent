# from cloud_points import main_test as generate_data
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

########################################################################################################################
########################################################################################################################

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
def generate_data():

    DIMENSION = 200
    LAM = 30
    zeros_parameters = [((np.ones(DIMENSION), LAM * np.eye(DIMENSION)), {"n_points": 200}),
                        ((3 * np.ones(DIMENSION), LAM * np.eye(DIMENSION)), {"n_points": 200})]
    ones_parameters = [((-1 * np.ones(DIMENSION), LAM * np.eye(DIMENSION)), {"n_points": 200}),
                       ((-3 * np.ones(DIMENSION), LAM * np.eye(DIMENSION)), {"n_points": 200})]

    data = generate_data_gaussian(zeros_parameters, ones_parameters)
    return data

########################################################################################################################
########################################################################################################################


# Data of the problem
DATA = generate_data()
FEATURES = DATA[:, :-1]
LABELS = (DATA[:, -1] - 0.5) * 2
NB_FEATURES, DIM = FEATURES.shape


# Parametrization
LAMBDA = 0.1
X_0 = np.random.normal(0, LAMBDA, NB_FEATURES + DIM + 1)
# X0_PRIMAL = X0[:DIM + 1]
# ALPHA0 = X0[DIM + 1:]

MODEL_SOLUTION = SVC(kernel='linear')
MODEL_SOLUTION.fit(FEATURES, LABELS)
SOL = MODEL_SOLUTION.coef_
SOL0 = MODEL_SOLUTION.intercept_


def product_hadamard_(vec, mat):
    res = np.array([vec * mat[ind] for ind, vec in enumerate(mat)])
    return res


lab_feat = product_hadamard_(LABELS, FEATURES)


def primal_function(X):
    return 0.5 * np.linalg.norm(X[:DIM]) ** 2

# TODO: Descente dans le Dual, gradient projeté
def d_lagrangian(x):
    dw = x[:DIM] - np.sum(product_hadamard_(x[DIM + 1:], lab_feat), axis=0)
    dw0 = - np.sum(x[DIM + 1:] * LABELS, axis=0)
    dalpha = LABELS * (np.dot(FEATURES, x[:DIM]) + x[DIM]) - 1

    dx = np.concatenate([dw, np.array([dw0]), dalpha])

    return dx


def constante_lipschitz(index=-1):
    L = 2 * sum(lab_feat ** 2)
    return L[index]


def solution():
    return SOL, 1 / np.linalg.norm(SOL)


def main():
    assert np.linalg.norm(hess - hess.T) < 0.001


if __name__ == '__main__':
    main()