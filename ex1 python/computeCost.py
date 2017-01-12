import numpy as np

def computeCost(X,y,theta):
    m = len(y)
    J = 0
    
    J = np.dot((np.dot(X,theta)-y).T , (np.dot(X,theta)-y))/(2*m)
    
    return J