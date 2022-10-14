import numpy as np
#import Problem as pb
import time

# DIM = int(input("Dimension:"))
DIM = 100
# np.random.seed(int(time.time()))
np.random.seed(100)
N = np.random.uniform(-10, 10, (DIM, DIM))
Y = np.random.uniform(-10, 10, DIM)
X_0 = np.zeros(DIM)
ALPHA = 1

print(N)


def conditionnement():
    return np.linalg.norm(N.T @ N) * np.linalg.norm(np.linalg.inv(N.T @ N))


def linear_application(x):
    return np.linalg.norm(N @ x - Y) ** 2 / 2


def d_linear_application(x):
    return N.T @ (N @ x - Y)


def linear_optimize_coordinate(x, ind):
    x_star = x.copy()
    x_star[ind] = 0
    y__ = Y - N.T @ x_star

    x_hat = np.linalg.inv(N.T @ N) @ (N.T @ Y).T
    return x_hat


def solution():
    x_sol = (np.linalg.inv(N.T @ N) @ (N.T @ Y.T).T)
    return x_sol, linear_application(x_sol)


def constante_lipschitz(index=-1):
    vec_lip = np.sum(N ** 2, axis=0)
    if index == -1:
        return np.max(vec_lip)
    else:
        return vec_lip[index]


def ridge(x):
    return np.linalg.norm(x) ** 2 / 2 + linear_application(x)


def d_ridge(x):
    return ALPHA * x + d_linear_application(x)


def solution_ridge():
    x_sol = np.linalg.inv(ALPHA * np.eye(DIM) + N.T @ N) @ (N.T @ Y).T
    return x_sol, ridge(x_sol)

"""
class linear_regression(pb.Problem):
    def create_function(self, N: np.array, Y: np.array):
        return lambda x:  np.linalg.norm(N @ x - Y) ** 2 / 2

    def create_derivative(self, N: np.array, Y: np.array):
        return lambda x, j: N.T[:, len(x)] @ (N[len(x)] @ x - Y[j])


"""