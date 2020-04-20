import numpy as np
from computeCost import computeCost

def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    for i in range(num_iters):
        predictions = np.dot(X,theta)
        error = np.dot(X.T ,(predictions-y) )
        descent = alpha * 1/m * error
        theta = theta - descent
        if(i%100 == 0):
            print('Cost function after',i,' iteration is ', computeCost(X,y,theta))
        
    return theta
