import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = dict()

def f0(x):
    return 3

def f1(x):
    return 0.01 * x - 2

def f2(x):
    return (-0.01 * x + 9)

def f3(x):
    return -0.02 * x + 5

def f4(x):
    return 0.02 * x - 13

def f5(x):
    return 0.5 #* np.sin(x/5)

def f(x):
    return (x <= 500) * f0(x) \
           + (500 < x <=550) * f1(x) \
           + (550 < x <=600) * f2(x) \
           + (600 < x) * f0(x)

def anomaly(x):
    return (500 < x < 600) * 1.0

data["X"] = np.linspace(0, 1000, 1000)
data["sigma"] = 0.1
data["Y"] = np.vectorize(f)(data["X"]) + np.random.normal(0.0, data["sigma"], len(data["X"]))
data["Anomaly"] = np.vectorize(anomaly)(data["X"])

df = pd.DataFrame(data)

df.to_csv('dd_test_basic_anomaly5.csv')

plt.scatter(x = data["X"], y = data["Y"], marker = ".", linewidth = 0.001)
plt.axvspan(list(data["Anomaly"]).index(1), list(data["Anomaly"]).index(0, list(data["Anomaly"]).index(1)), facecolor="red", alpha=0.4)
plt.show()
