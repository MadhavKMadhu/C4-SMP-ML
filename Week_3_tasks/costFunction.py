import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):
    '''Returns Cost for theta, X, y'''

    m = y.size
    h = sigmoid(np.dot(X,theta))
    J = [-(1/m) * (np.sum((y.T)*np.log(h) + (1-(y.T))*np.log(1-h)))]

    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient(theta, X, y):
    '''Calculate Gradient Descent for Logistic Regression'''
    m = y.size
    theta = theta.reshape(-1,1)
    h = sigmoid(np.dot(X,theta))
    grad = ((1/m) * np.dot(X.T, (h-y)) )
    return(grad.flatten())
