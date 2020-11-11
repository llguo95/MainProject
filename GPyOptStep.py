import numpy as np
import GPyOpt
from matplotlib import pyplot as plt

def forrester(x):
    return -(6 * x - 2) ** 2 * np.sin(12 * x - 4)

plt.plot(np.linspace(0, 1.05, 100), forrester(np.linspace(0, 1.05, 100)))

domain_myF = [{'name': 'var1', 'type': 'continuous', 'domain': (0, 1.05)}]

X_init = np.array([[0.], [0.5], [1.05]])
Y_init = forrester(X_init)

iter_count = 5
current_iter = 0
X_step = X_init
Y_step = Y_init
print(Y_step)
files = []
while current_iter < iter_count:
    print("iteration ", current_iter)
    bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=domain_myF, X=X_step, Y=Y_step)
    x_next = bo_step.suggest_next_locations()

    y_next = forrester(x_next)

    X_step = np.vstack((X_step, x_next))
    Y_step = np.vstack((Y_step, y_next))

    if current_iter < 10:
        strxFile = 'opt_0%d' % current_iter
    else:
        strxFile = 'opt_%d' % current_iter
    files.append(strxFile + ".png")
    bo_step.plot_acquisition(filename=strxFile)
    current_iter += 1

print(files)

plt.show()