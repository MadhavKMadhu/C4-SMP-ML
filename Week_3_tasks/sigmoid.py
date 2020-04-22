import numpy as np

def sigmoid(z):
    '''Returns sigmoid of z'''
    sgm = 1 / (1 + np.exp(-z))
    return sgm
