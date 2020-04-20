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

# FInal Solution
/madhav/C4-SMP-ML/Week2/singleVariableMain.py$ /usr/bin/python3 /home/
        0        1
0  6.1101  17.5920
1  5.5277   9.1302
2  8.5186  13.6620
3  7.0032  11.8540
4  5.8598   6.8233
Press enter if you have completed computeCost file, else Ctrl+C then enter to exit
Cost function J value: 32.072733877455676
Press enter if you have completed gradientDescent file, else Ctrl+C then enter to exit
Cost function after 0  iteration is  6.737190464870006
Cost function after 100  iteration is  5.476362817272741
Cost function after 200  iteration is  5.173634551165021
Cost function after 300  iteration is  4.962606493117519
Cost function after 400  iteration is  4.8155014941166865
Cost function after 500  iteration is  4.712956453749759
Cost function after 600  iteration is  4.6414735988143185
Cost function after 700  iteration is  4.591643801766726
Cost function after 800  iteration is  4.5569080784097515
Cost function after 900  iteration is  4.532694243543437
Cost function after 1000  iteration is  4.515815084502823
Cost function after 1100  iteration is  4.50404883551784
Cost function after 1200  iteration is  4.495846731678218
Cost function after 1300  iteration is  4.490129148489064
Cost function after 1400  iteration is  4.486143493324961
New Cost Function value: 4.483388256587725
