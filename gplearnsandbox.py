import numpy as np
import matplotlib.pyplot as plt
import gplearn.functions as gf
import gplearn.genetic as gl
rng = np.random.RandomState()

xmin = -0.1
xmax = 0.2

N = 600
Ng = 1000

k = 6.817
c = 0.222
k_P = 0.085
gamma = 0.073
n = 6

def sigma(eps):
    return np.exp(-30*eps)
    # return np.exp(-k*eps/c)*(np.exp(k*eps/c) - 1)*c# + (k_P + gamma*(1 - np.exp(eps))**n)*eps

inTrain = np.linspace(xmin, xmax, N).reshape(N, 1)
domaingrid = np.linspace(xmin, xmax, Ng)

outTrain = np.ravel(sigma(inTrain))

def _protected_exponent(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.)

def _protected_negexponent(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(-x), 0.)


pexp = gf.make_function(_protected_exponent, 'exp', 1)
pnexp = gf.make_function(_protected_negexponent, 'nexp', 1)
f_s = ['add', 'sub', 'mul', 'div', pexp, pnexp, 'neg']

est_gp = gl.SymbolicRegressor(init_depth=(2, 4), population_size=3000,
                              tournament_size=20, const_range=(-40, 40),
                              generations=20, stopping_criteria=0.01,
                              p_crossover=0.7, p_subtree_mutation=0.1,
                              warm_start=True,
                              p_hoist_mutation=0.05, p_point_mutation=0.1,
                              max_samples=0.9, verbose=1, random_state=0,
                              function_set=f_s)
est_gp.fit(inTrain, outTrain)

y_gp = est_gp.predict(np.c_[domaingrid.ravel()]).reshape(domaingrid.shape)

print(est_gp.program)

plt.plot(inTrain, outTrain)
plt.plot(domaingrid, y_gp)
plt.show()