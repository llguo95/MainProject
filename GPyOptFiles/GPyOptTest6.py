import GPyOpt
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt

seed(123)

# func = GPyOpt.objective_examples.experimentsNd.alpine1(input_dim=7)
# X = np.ones((7, 1))
# print(X)
# print(func.f(X))
#
# def f(X, input_dim, sd):
#     X = np.array([np.reshape(X, input_dim)])
#     print(X)
#     n = X.shape[0]
#     fval = (X * np.sin(X) + 0.1 * X).sum(axis=1)
#     if sd == 0:
#         noise = np.zeros(n).reshape(n, 1)
#     else:
#         noise = np.random.normal(0, sd, n)
#     return fval.reshape(n, 1) + noise
#
# print(f(X, 7, 0))

class alpine1:
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def f(self, X):
        X = np.array([np.reshape(X, self.input_dim)])
        # print(X)
        n = X.shape[0]
        fval = np.abs(X * np.sin(X) + 0.1 * X).sum(axis=1)
        return fval.reshape(n, 1)

func = alpine1(input_dim=7)

mixed_domain = [{'name': 'var1', 'type': 'continuous', 'domain': (-5, 5), 'dimensionality': 3},
                {'name': 'var3', 'type': 'discrete', 'domain': (0, 8, 10), 'dimensionality': 2},
                {'name': 'var4', 'type': 'discrete', 'domain': (1, 2, 100), 'dimensionality': 1},
                {'name': 'var5', 'type': 'continuous', 'domain': (-1, 2)}]

myBopt = GPyOpt.methods.BayesianOptimization(f=func.f,  # Objective function
                                             domain=mixed_domain,  # Box-constraints of the problem
                                             initial_design_numdata=1,  # Number data initial design
                                             acquisition_type='MPI',  # Expected Improvement
                                             exact_feval=True)  # True evaluations, no sample noise

max_iter = 35  ## maximum number of iterations
max_time = 60  ## maximum allowed time
eps = 1e-4  ## tolerance, max distance between consicutive evaluations.

myBopt.run_optimization(max_iter, eps=eps, verbosity=True)

print(myBopt.x_opt)
print(myBopt.Y_best)

myBopt.plot_convergence()
myBopt.plot_acquisition()

plt.show()
