import numpy as np
import copy
from RNN import RNN
"""
NOTE: This code is a rewrite of code written by Elias Stihl. Since the code for calculating numerical gradients
in Matlab was given, I assume it is fine to use someone else's python translation of said code.
"""

def compute_gradients_numerically(X,Y,step_size,name, rnn, h_prev):
    grad = np.zeros_like(rnn.parameters[name])
    for i in range(rnn.parameters[name].shape[0]):
        for j in range(rnn.parameters[name].shape[1]):
            rnn_try = copy.deepcopy(rnn)
            rnn_try.parameters[name][i][j] = rnn.parameters[name][i][j] - step_size
            
            l1 = rnn_try.compute_loss(X, Y, h_prev)
            rnn_try.parameters[name][i][j] = rnn.parameters[name][i][j] + step_size
            l2 = rnn_try.compute_loss(X, Y, h_prev)
            grad[i][j] = (l2-l1) / (2 * step_size)
    return grad