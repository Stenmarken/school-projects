import numpy as np
from data_normalization import prepare_data, normalize_data
from functions import ComputeGradients, ComputeGradientDifference
from matlab_functions import ComputeGradsNum, ComputeGradsNumSlow

K = 10 # Number of distinct labels
d = 3072 # Number of datapoints for each image
n = 10000 # Number of images
mu, sigma = 0, 0.01 # mean and standard deviation
llambda = 0.0
num_batches = 100
num_images = 10000
batch_size = num_images // num_batches
eta = 0.001
llambda = 0
n_epochs = 5

# Y is one-hot encoded, y is not
train_X, train_Y, train_y = prepare_data("cifar-10-batches-py/data_batch_1")
validation_X, validation_Y, validation_y = prepare_data("cifar-10-batches-py/data_batch_2")
test_X, test_Y, test_y = prepare_data("cifar-10-batches-py/test_batch")

train_X, validation_X, test_X = normalize_data(train_X, validation_X, test_X)

W_values = np.random.normal(mu, sigma, K*d)
b_values = np.random.normal(mu, sigma, K)

W = W_values.reshape((K, d))
b = b_values.reshape((K, 1))

for i in range(100):
    first_image_X = np.reshape(train_X[0:3072, i], (3072, 1))
    first_image_Y = np.reshape(train_Y[:, i], (10, 1))
    W_changed = np.reshape(W[:, 0:3072], (K, 3072))

    print("W_changed", np.shape(W_changed))
    print("first_image_X", np.shape(first_image_X))
    print("first_image_Y", np.shape(first_image_Y))
    print("b", np.shape(b))

    grad_W_an, grad_b_an = ComputeGradients(X=first_image_X, Y=first_image_Y, P=[], W=W_changed, llambda=llambda, b=b, use_sigmoid=True)
    h = 1e-6

    grad_W_num, grad_b_num = ComputeGradsNum(X=first_image_X, Y=first_image_Y, P=[], W=W_changed, b=b, lamda=llambda, h=h, use_compute_cost_multiple=True)
    
    grad_W_an /= 10
    grad_b_an /= 10

    print("grad_W_an", np.shape(grad_W_an))
    print("grad_W_num", np.shape(grad_W_num))

    W_difference = ComputeGradientDifference(grad_W_an, grad_W_num)
    print("W_difference:", W_difference)
    
    b_difference = ComputeGradientDifference(grad_b_an, grad_b_num)
    print("b_difference:", b_difference)

    assert b_difference < h
    assert W_difference < h