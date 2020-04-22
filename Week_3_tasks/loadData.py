import numpy as np

def loadData(file, delimiter):
    data = np.loadtxt(file, delimiter=delimiter)    # Load the file using numpy
    print('Dimensions: ', data.shape)              
    print(data[1:6, :])                              # Print first six samples of data
    return(data)
