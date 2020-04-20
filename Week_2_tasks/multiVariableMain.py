import numpy as np
import pandas as pd
data = pd.read_csv('ex1data2.txt', sep = ',', header=None)
X = data.iloc[:,0:2]            # read first two columns
y = data.iloc[:,2]              # read the third column into y
m = len(y)                      # no. of training samples
print(data.head())              # view first few rows of data
ones = np.ones((m,1))             
X = np.hstack((ones,X))
alpha = 0.01
iterations = 500
theta = np.zeros((3,1))
y = y[:,np.newaxis]

# Functions defined in other files will be imported here

from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalization import featureNormalization

X = featureNormalization(X)
J = computeCost(X,y,theta)
print('Cost function J value:',J)
input('Press enter if you have completed gradient descent file, else Ctrl+C then enter to exit')
theta = gradientDescent(X, y, theta, alpha, iterations)
J = computeCost(X,y,theta)
print('New Cost function value:',J)

