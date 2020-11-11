# import GPy
# import GPyOpt
# from numpy.random import seed
# import matplotlib
#
# def myf(x):
#     return 2*x**2
#
# bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1, 1)}]
#
# max_iter = 10
#
# myProblem = GPyOpt.methods.BayesianOptimization(myf, bounds)
#
# myProblem.run_optimization(max_iter)
#
# print(myProblem.x_opt)
#
# print(myProblem.fx_opt)

# import GPy
# import GPyOpt
# import matplotlib.pyplot as plt
#
# # Create the true and perturbed Forrester function and the boundaries of the problem
# f_true = GPyOpt.objective_examples.experiments1d.forrester()          # noisy version
# bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]  # problem constraints
#
# f_true.plot()
#
# # Creates GPyOpt object with the model and anquisition fucntion
# myBopt = GPyOpt.methods.BayesianOptimization(f=f_true.f,            # function to optimize
#                                              domain=bounds,        # box-constraints of the problem
#                                              acquisition_type='EI',
#                                              exact_feval = True) # Selects the Expected improvement
#
# # Run the optimization
# max_iter = 15     # evaluation budget
# max_time = 60     # time budget
# eps = 10e-6  # Minimum allows distance between the las two observations
#
# myBopt.run_optimization(max_iter, max_time, eps)
#
# myBopt.plot_acquisition()
#
# myBopt.plot_convergence()

import GPy
import GPyOpt
from matplotlib import pyplot as plt
from numpy.random import seed

seed(123)

# create the object function
f_true = GPyOpt.objective_examples.experiments2d.rosenbrock()
f_sim = GPyOpt.objective_examples.experiments2d.rosenbrock(sd=0.1)
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': f_true.bounds[0]},
          {'name': 'var_2', 'type': 'continuous', 'domain': f_true.bounds[1]}]
f_true.plot()

# Creates three identical objects that we will later use to compare the optimization strategies
myBopt2D = GPyOpt.methods.BayesianOptimization(f_sim.f,
                                               domain=bounds)

# runs the optimization for the three methods
max_iter = 20  # maximum time 40 iterations
max_time = 60  # maximum time 60 seconds

myBopt2D.run_optimization(max_iter, max_time, verbosity=True)

myBopt2D.plot_acquisition()

myBopt2D.plot_convergence()

print(myBopt2D.x_opt)
print(myBopt2D.Y_best)
plt.show()
