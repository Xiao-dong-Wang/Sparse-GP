import numpy as np
import sys
import toml
from get_dataset import *
import matplotlib.pyplot as plt
import GPy

np.random.seed(1)

argv = sys.argv[1:]
conf = toml.load(argv[0])

name = conf['funct']
funct = get_funct(name)
num = conf['num']
bounds = np.array(conf['bounds'])
num_inducing = conf['num_inducing']

data = init_dataset(funct, num, bounds)
train_x = data['train_x']
train_y = data['train_y']

# conventional GP regression
k1 = GPy.kern.RBF(1, ARD=True)
m1 = GPy.models.GPRegression(X=train_x, Y=train_y, kernel=k1)
m1.kern.variance = np.var(train_y)
m1.kern.lengthscale = np.std(train_x)
m1.likelihood.variance = 0.01 * np.var(train_y)
m1.optimize('bfgs')

# sparse GP regression via FITC approximation
z = np.random.uniform(bounds[0,0], bounds[0,1], (num_inducing, 1))
k2 = GPy.kern.RBF(1, ARD=True)
m2 = GPy.models.SparseGPRegression(X=train_x, Y=train_y, kernel=k2, Z=z)
m2.inference_method=GPy.inference.latent_function_inference.FITC()
m2.optimize('bfgs')

inducing_points = np.zeros((num_inducing, 1))
for i in range(num_inducing):
    inducing_points[i,0] = m2.inducing_inputs[i]
print('inducing points', inducing_points)

# Test data
data_star = get_test(funct, 400, bounds)
X_star = data_star['test_x']
y_star = data_star['test_y']

y_pred1, y_var1 = m1.predict(X_star)
y_pred2, y_var2 = m2.predict(X_star)

plt.figure(figsize=(6,6))
plt.subplot(2,1,1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
#plt.plot(X_star.flatten(), y_star.flatten(), 'b-', label = "Exact", linewidth=2)
plt.plot(X_star.flatten(), y_pred1.flatten(), 'b--', label = "Predictive mean", linewidth=2)
lower1 = y_pred1 - 2.0*np.sqrt(y_var1)
upper1 = y_pred1 + 2.0*np.sqrt(y_var1)
plt.fill_between(X_star.flatten(), lower1.flatten(), upper1.flatten(), 
                 facecolor='skyblue', alpha=0.5, label="Two std band")
plt.plot(train_x, train_y, 'k+', label = "Data points")
ax = plt.gca()
ax.set_xlim([0.9*bounds[0,0], 1.1*bounds[0,1]])
ax.set_ylim([-1.5, 1.5])
ax.set_title('Conventional GP Regression')
plt.legend()
plt.ylabel('f(x)')

plt.subplot(2,1,2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
#plt.plot(X_star.flatten(), y_star.flatten(), 'b-', label = "Exact", linewidth=2)
plt.plot(X_star.flatten(), y_pred2.flatten(), 'b--', label = "Predictive mean", linewidth=2)
lower2 = y_pred2 - 2.0*np.sqrt(y_var2)
upper2 = y_pred2 + 2.0*np.sqrt(y_var2)
plt.fill_between(X_star.flatten(), lower2.flatten(), upper2.flatten(), 
                 facecolor='skyblue', alpha=0.5, label="Two std band")
plt.plot(inducing_points, -1.5*np.ones(num_inducing), 'r^', label = 'Inducing points location', ms=8)
ax = plt.gca()
ax.set_xlim([0.9*bounds[0,0], 1.1*bounds[0,1]])
ax.set_ylim([-1.5, 1.5])
ax.set_title('Sparse GP Regression')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')

plt.subplots_adjust(hspace=0.3)
plt.show()


