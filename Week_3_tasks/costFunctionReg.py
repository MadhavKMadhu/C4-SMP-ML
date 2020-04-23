import numpy as np
from sigmoid import sigmoid

def costFunctionReg(theta, reg, X, y):
    '''Returns the cost in a Regularized manner'''

    m = y.size
    h = sigmoid(X.dot(theta))
    theta_J = theta[1:]
    regparameter = (1/ (2*m)) * np.sum(np.power(theta_J,2)) 
    J = -1*(1/m)*((np.log(h).T).dot(y)+np.log(1-h).T.dot(1-y)) + regparameter
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])


def gradientReg(theta, reg, X, y):
    '''Returns the Gradient for input values of theta, reg, X, y'''

    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1/m)*X.T.dot(h-y) + (reg/m)*np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return(grad.flatten())
