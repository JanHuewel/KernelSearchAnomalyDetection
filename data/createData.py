import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = dict()

def f0(x):
    return 0.0

def f1(x):
    return (0.1 * x) #+ 3*np.sin(x/3)

def f2(x):
    return (0.01 * x - 10)

def f3(x):
    return -0.02 * x + 5

def f4(x):
    return 10.0

def f5(x):
    return 0.5 #* np.sin(x/5)

def f(x):
    return (x<1400) * f0(x) \
           + (400<=x<=500) * f3(x)

data["X"] = np.linspace(0, 1000, 30)
data["sigma"] = 0.02
data["Y"] = np.vectorize(f)(data["X"]) + np.random.normal(0.0, data["sigma"], len(data["X"]))

df = pd.DataFrame(data)

df.to_csv('dd_test_basic_anomaly0.csv')

plt.scatter(x = data["X"], y = data["Y"], marker = ".", linewidth = 0.001)
plt.show()
