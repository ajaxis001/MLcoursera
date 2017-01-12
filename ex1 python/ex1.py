## Machine Learning Online Class - Exercise 1: Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from warmUpExercise import warmUpExercise 
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

plt.close('all')

#==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... \n')
print('5x5 identity matrix : \n')
warmUpExercise()
print('\n\n')

#======================= Part 2: Plotting =======================
df = pd.read_csv('C:\Users\hp\Documents\Projects Folder\ML learning\Coursera ML Andrew NG\ex1 python\ex1data1.txt', header=None)
dftmtx = df.as_matrix()

tSet = np.loadtxt('C:\Users\hp\Documents\Projects Folder\ML learning\Coursera ML Andrew NG\ex1 python\ex1data1.txt', delimiter = ',')

#np.shape(tSet) gives (97L,)
# this is just a vector and not a proper matrix we have to add an 
# extra dimension before we can treat it as a mx1 matrix we do it by the following methods
# x = tSet[:,0][:,np.newaxis] OR x.shape = (97,1)

x = tSet[:,0][:,np.newaxis]
y = tSet[:,1][:,np.newaxis]

m = len(y)

plotData(x,y,1)

#=================== Part 3: Gradient descent ===================
X = np.hstack((np.ones((m,1)) , x)) # concatenating the two 
#sz_X = np.shape(X)

theta = np.zeros((2,1)) # init fitting params
iterations = 1500
alpha = 0.01

print('The initial cost is = {}'.format(computeCost(X,y,theta)))

# running gradient descent
theta, J_history = gradientDescent(X,y,theta,alpha,iterations)

# plotting variation in J w.r.t number of iterations
plt.figure(2)
plt.plot(np.arange(1,iterations+1),J_history,'r.')
plt.xlabel('Number of Iterations')
plt.ylabel('cost function (J)')
plt.ion()

plt.show(block = True)