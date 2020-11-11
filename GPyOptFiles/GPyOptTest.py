# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
from matplotlib import pyplot as plt
from numpy.random import seed
import numpy as np
seed(123)

# --- Define your problem
def f(x): return (6*x - 2) ** 2 * np.sin(12 * x - 4)
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)}]

# --- Solve your problem
myBopt = BayesianOptimization(f=f, domain=domain, initial_design_numdata=1, acquisition_type='MPI')
myBopt.run_optimization(max_iter=40, verbosity=True)
myBopt.plot_acquisition()
myBopt.plot_convergence()

print(myBopt.x_opt)
print(myBopt.Y_best)

plt.show()