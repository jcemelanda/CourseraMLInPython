import numpy as np


def warmup():
    """
        % ============= YOUR CODE HERE ==============
        % Instructions: Return the 5x5 identity matrix
        %
    """

    A = np.eye(5)

    return A


def computeCost(X, y, theta):
    m = X.shape[0]

    J = np.sum(np.square(X @ theta - y)) / (2 * m)

    return J


def gradientDescent(X, y, theta, alpha, iterations):
    m = X.shape[0]

    J_history = np.zeros((iterations, 1))

    for i in range(iterations):
        J_history[i] = computeCost(X, y, theta)
        theta -= alpha * (X.T @ (X @ theta - y)) / m

    return theta, J_history