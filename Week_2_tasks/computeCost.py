import numpy as np

def computeCost(X,y,theta):
    m = len(y)
    predicitons = np.dot(X,theta)
    square_err = np.power( (predicitons-y) , 2 )

    return 1/(2*m) * np.sum(square_err)
