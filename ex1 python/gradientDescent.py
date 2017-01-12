import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y) # number of training examples
    J_history = np.zeros((iterations,1))
    theta_old = theta
    
    for iter in np.arange(1,iterations):
        d_J = (np.dot(np.dot(X.T,X),theta_old) - np.dot(X.T,y))/m  # differentiating cost function using matrix calculus
        theta_new = theta_old - (alpha * d_J)
        theta_old = theta_new 
        J_history[iter-1] = computeCost(X,y,theta_new)
        #print("J[{a}] = {b}".format(a=iter,b=J_history[iter-1]))
        
    theta = theta_new
    return theta ,J_history
        