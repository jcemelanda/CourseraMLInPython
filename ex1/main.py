from exercise import *
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':

    a1 = warmup()
    print(f'Answer:\n\n{a1}')
    assert np.equal(a1, np.eye(5)).all(), 'warmup() does not return the identity matrix'

    input('Press Enter to continue...')

    data1 = np.genfromtxt(fname='ex1data1.txt', delimiter=',' , dtype=float)
    X = data1[:, 0].reshape(-1, 1)
    y = data1[:, 1].reshape(-1, 1)
    n = X.shape[0]

    plt.plot(X, y, 'rx')
    plt.show(block=False)

    input('Press Enter to continue...')
    X = np.hstack((np.ones(shape=(n, 1)), X))
    theta = np.zeros(shape=(2, 1))

    iterations = 1500
    alpha = 0.01

    print('Test cost function')

    J = computeCost(X, y, theta)

    print(f'With theta = [0 ; 0]\nCost computed = {J}')
    print('Expected cost value (approx) 32.07')

    input('Press Enter to continue...')

    J = computeCost(X, y, np.matrix('-1; 2'))
    print(f'With theta = [-1 ; 2]\nCost computed = {J}')
    print('Expected cost value (approx) 54.24')

    input('Press Enter to continue...')

    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

    print('Theta found by gradient descent:\n')
    print(theta)
    print('Expected theta values (approx)\n')
    print(' -3.6303\n  1.1664\n\n')

    input('Press Enter to continue...')

    plt.plot(X[:, 1], X @ theta, 'b-')
    plt.show(block=False)

    input('Press Enter to continue...')

    predict1 = np.matrix('1, 3.5') @ theta
    print('For population = 35,000, we predict a profit of ', predict1*10000)
    predict2 = np.matrix('1, 7') @ theta
    print('For population = 70,000, we predict a profit of ', predict2*10000)

    input('Press Enter to continue...')

    print('Visualizing J(theta_0, theta_1) ...')

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    J_vals = np.zeros(shape=(len(theta0_vals), len(theta1_vals)))

    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.matrix(([theta0_vals[i]], [theta1_vals[j]]))
            J_vals[i, j] = computeCost(X, y, t)

    J_vals = J_vals.T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    plt.show(block=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
    plt.show(block=False)

    input('Press Enter to continue...')
