import numpy as np
from matplotlib import pyplot as plt
from GPyOpt.methods import BayesianOptimization

# --- Fixed seed for consistent results
np.random.seed(123)


# --- Define your problem
# def R(x): return (1 - x[:, 0]) ** 2 + 10 * (x[:, 1] - x[:, 0] ** 2) ** 2
#
#
# domainR = [{'name': 'x', 'type': 'continuous', 'domain': (0, 2)},
#            {'name': 'y', 'type': 'continuous', 'domain': (0, 2)}]
#
# # --- Solve your problem
# numberOfInitialPoints = 5
# numberOfIterations = 15
#
# myBoptR = BayesianOptimization(f=R, acquisition_type='EI', domain=domainR, normalize_Y=True,
#                                initial_design_numdata=numberOfInitialPoints)
# myBoptR.run_optimization(max_iter=numberOfIterations, max_time=5000)
# myBoptR.plot_acquisition()
#
# # --- Printing some interesting stuff
# print("Next acquisition minimizing argument: ")
# print(myBoptR.suggest_next_locations())
#
# print("Progression of the approximated minimum function value:")
# print(myBoptR.Y_best)

def f(X, input_dim, sd):
    X = np.reshape(X, input_dim)
    n = X.shape[0]
    fval = (X * np.sin(X) + 0.1 * X).sum(axis=1)
    if sd == 0:
        noise = np.zeros(n).reshape(n, 1)
    else:
        noise = np.random.normal(0, sd, n)
    return fval.reshape(n, 1) + noise

input_dim = 1
sd = 0
X = np.linspace(0, 1, 11).reshape(-1, 1)
print(X)
print(f(X, input_dim, sd))