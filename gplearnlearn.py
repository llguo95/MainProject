import numpy as np
import matplotlib.pyplot as plt
import gplearn.functions as gf
import gplearn.genetic as gl
import pandas as pd

df = pd.read_excel(r'C:\Users\leoguo\Downloads\leds.xlsx')
dfcolumntime = pd.DataFrame(df, columns = ["Time (Hrs)"])
dfcolumnunit = pd.DataFrame(df, columns = ["Test Samples"])
columntime = pd.DataFrame.to_numpy(dfcolumntime)
columnunit = pd.DataFrame.to_numpy(dfcolumnunit)
X_train = np.array(columntime[1:], dtype = 'float64')
X_train = X_train/max(X_train)
X_traintemp = X_train
X_train = X_train[0:10]
y_train = np.ravel(np.array(columnunit[1:], dtype = 'float64'))
y_traintemp = y_train
y_train = y_train[0:10]

rng = np.random.RandomState()

xmin = min(X_train)
xmax = max(X_train)

# xmin = 1
# xmax = 3

X_int = np.linspace(xmin, xmax, 200)

N = 20

# X_train = rng.uniform(xmin, xmax, N).reshape(N, 1)
# y_train = np.ravel(X_train)

def protexp(x):
    return np.exp(-np.abs(x))

nexp = gf.make_function(protexp, 'negabsexp', 1)
f_s = ['add', 'sub', 'mul', 'div', 'inv', 'abs', nexp, 'log']

est_gp = gl.SymbolicRegressor(init_depth=(3, 6), population_size=4000,
                              tournament_size=20,
                              generations=30, stopping_criteria=0.01,
                              p_crossover=0.7, p_subtree_mutation=0.1,
                              p_hoist_mutation=0.05, p_point_mutation=0.1,
                              max_samples=0.9, verbose=1,
                              parsimony_coefficient=0.01, random_state=0,
                              function_set=f_s)
est_gp.fit(X_train, y_train)

y_gp = est_gp.predict(np.c_[X_traintemp.ravel()]).reshape(X_traintemp.shape)

print(est_gp.program)

plt.plot(X_traintemp, y_traintemp, '.')
plt.plot(X_traintemp, y_gp)
plt.title('SR on raw data')

plt.show()
