""" 
This file runs the training for a small subset of the training data to ensure that the model 
can overfit on that data. This is to ensure that the gradient calculations are correct.
"""

import numpy as np
from data_normalization import prepare_data, normalize_data
from matlab_functions import displayimages
from functions import initialize_model, MiniBatchGD
import scipy.io

np.random.seed(0)

d = 3072 # Number of features
K = 10 # Number of classes
m = 50 # Number of nodes

llambda = 0
n_epochs = 40
num_batches = 10
eta = 0.001
num_images = 100
batch_size = 10

# Y is one-hot encoded, y is not
train_X, train_Y, train_y = prepare_data("../lab1/cifar-10-batches-py/data_batch_1")
validation_X, validation_Y, validation_y = prepare_data("../lab1/cifar-10-batches-py/data_batch_2")
test_X, test_Y, test_y = prepare_data("../lab1/cifar-10-batches-py/test_batch")

train_X, validation_X, test_X = normalize_data(train_X, validation_X, test_X)

train_X = train_X[:, 0:100]
train_Y = train_Y[:, 0:100]
train_y = train_y[0:100]

train_X_batch_acc = []
train_Y_batch_acc = []

for i in range(num_images // num_batches):
    i_start = i * num_batches
    i_end = (i + 1) * num_batches
    train_X_batch_acc.append(train_X[:, i_start:i_end])
    train_Y_batch_acc.append(train_Y[:, i_start:i_end])

train_X_batch_acc = np.asarray(train_X_batch_acc)
train_Y_batch_acc = np.asarray(train_Y_batch_acc)

print("train_X", np.shape(train_X))
print("train_Y", np.shape(train_Y))
print("train_y", np.shape(train_y))

W_1, W_2, b_1, b_2 = initialize_model(K=K, d=d, m=m)

GDParams = [llambda, eta, batch_size, num_batches, n_epochs]

W, b, train_epoch_costs, train_epoch_loss, epoch_accuracy, \
        validation_epoch_costs, validation_epoch_loss, validation_epoch_accuracy = MiniBatchGD(X=train_X_batch_acc, Y=train_Y_batch_acc, GDParams=GDParams, W_1=W_1, W_2=W_2, b_1=b_1, b_2=b_2, train_X=train_X, train_Y=train_Y, train_y=train_y, validation_X=validation_X, validation_Y=validation_Y, validation_y=validation_y)

W_1, W_2 = W 
b_1, b_2 = b