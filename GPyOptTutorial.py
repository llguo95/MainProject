# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import seed
seed(123)

# --- Define your problem
def f(x): return (6*x-2)**2*np.sin(12*x-4)
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)}]

# --- Solve your problem
myBopt = BayesianOptimization(f=f, domain=domain)
myBopt.run_optimization(max_iter=15)

myBopt.plot_acquisition()

dommesh = np.linspace(0, 1, 100)
plt.plot(dommesh, f(dommesh), '--')

myBopt.plot_convergence()

plt.show()
