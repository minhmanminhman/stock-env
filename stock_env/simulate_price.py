import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt

n = 5000
h = 5
pe = 50
sig = 0.1

# trading params
tick_size = 0.1
lot_size = 100
n_action = 5
M = 10

# calculated params
dt = 1 / n
lmbda = np.log(2) / h
action_space = lot_size * np.arange(-n_action, n_action+1)
holdings = np.arange(-M, M+1)

npr.seed(0)
# 1. Eucler method
xt = np.zeros(n, dtype=np.float64)
for i in range(1, n):
    xt[i] = xt[i-1] - lmbda * xt[i-1] * dt + sig * np.sqrt(dt) * npr.normal()
price = np.exp(xt) * pe
plt.plot(price)

npr.seed(0)
# 2. Solution in terms of integral
t = np.arange(0, 1, dt)
w = np.zeros(n)
x0 = 0

# calculate integral
for i in range(1, n):
    w[i] = w[i-1] + np.sqrt(dt) * np.exp(lmbda * t[i-1]) * npr.normal()

# intergral solution
ex = np.exp(-lmbda * t)
x = x0*ex + sig*ex*w

# transform price
price = np.exp(x) * pe
plt.plot(price)

npr.seed(0)
# 2. Solution in terms of integral
t = np.arange(0, 1, dt)
w = np.zeros(n)
x0 = 0

ex = np.exp(-lmbda * t)
w = np.sqrt(np.diff(np.exp(2*lmbda*t) - 1)) * npr.normal(size=n-1)
w = np.insert(w, 0, 0)
x = x0*ex + 0*(1-ex) + sig*ex*np.cumsum(w) / np.sqrt(2*lmbda)

# transform price
price = np.exp(x) * pe
plt.plot(price)