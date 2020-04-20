# Libraries we'll be using
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from computeCost.py import computeCost
#from gradientDescent.py import gradientDescent


data = pd.read_csv('ex1data1.txt', header=None)
X = data.iloc[:, 0]
y = data.iloc[:, 1]
m = len(y)
print(data.head())
plt.scatter(X,y)                                               # PLot the data on a graph
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
X = X[:,np.newaxis]
y = y[:,np.newaxis]
ones = np.ones((m,1))
X = np.hstack((ones,X))
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01

input('Press enter if you have completed computeCost file, else Ctrl+C then enter to exit')


# Func defined in other files will be imported here

from computeCost import computeCost
from gradientDescent import gradientDescent


J = computeCost(X,y,theta)
print('Cost function J value:',J)
input('Press enter if you have completed gradientDescent file, else Ctrl+C then enter to exit')
theta = gradientDescent(X,y,theta,alpha,iterations)
J = computeCost(X,y,theta)
print('New Cost Function value:', J)
