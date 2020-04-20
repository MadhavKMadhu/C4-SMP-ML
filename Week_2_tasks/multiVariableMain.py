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

# Final Solution
madhav@madhav-Inspiron-5559:~/C4-SMP-ML/Week2$  env DEBUGPY_LAUNCHER_PORT=42923 /usr/bin/python3 /home/madhav/.vscode/extensions/ms-python.python-2020.3.71659/pythonFiles/lib/python/debugpy/no_wheels/debugpy/launcher /home/madhav/C4-SMP-ML/Week2/multiVariableMain.py 
      0  1       2
0  2104  3  399900
1  1600  3  329900
2  2400  3  369000
3  1416  2  232000
4  3000  4  539900
Cost function J value: 65591548106.45744
Press enter if you have completed gradient descent file, else Ctrl+C then enter to exit
Cost function after 0  iteration is  62068007680.283844
Cost function after 100  iteration is  2679101950.7045717
Cost function after 200  iteration is  2372180772.7146063
Cost function after 300  iteration is  2284765774.850247
Cost function after 400  iteration is  2221957435.2096915
New Cost function value: 2176910653.2626677

