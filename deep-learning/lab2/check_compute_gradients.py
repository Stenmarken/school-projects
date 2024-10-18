import numpy as np
from data_normalization import prepare_data, normalize_data
from functions import Two_Dimensional_Network, ComputeGradientDifference
#ComputeGradients, ComputeGradientDifference, initialize_model
from ComputeGradsNum import ComputeGradsNum

np.random.seed(0)

d = 3072 # Number of features
K = 10 # Number of classes
m = 50 # Number of nodes

llambda = 0
n_epochs = 40
num_batches = 100
eta = 0.001

# Y is one-hot encoded, y is not
train_X, train_Y, train_y = prepare_data("../lab1/cifar-10-batches-py/data_batch_1")
validation_X, validation_Y, validation_y = prepare_data("../lab1/cifar-10-batches-py/data_batch_2")
test_X, test_Y, test_y = prepare_data("../lab1/cifar-10-batches-py/test_batch")

train_X, validation_X, test_X = normalize_data(train_X, validation_X, test_X)

network = Two_Dimensional_Network(K=K, d=d, m=m)


for i in range(1, 30, 1):
    print("W_1", np.shape(network.W_1))
    print("W_2", np.shape(network.W_2))
    first_image_X = train_X[(i-1)*20:20*i, (i-1)*2:2*i]
    first_image_Y = train_Y[:, (i-1)*2:2*i]
    #network.W_1 = network.W_1[:, 0:20]
    h = 1e-5

    d_L_d_W_2_an, d_L_d_b_2_an, d_L_d_W_1_an, d_L_d_b_1_an = network.ComputeGradients(X=first_image_X, Y=first_image_Y)
    [d_L_d_W_1_num, d_L_d_W_2_num], [d_L_d_b_1_num, d_L_d_b_2_num] = ComputeGradsNum(X=first_image_X, Y=first_image_Y, W = [network.W_1, network.W_2], b = [network.b_1, network.b_2], lambda_ = llambda, h = h)

    W_2_difference = ComputeGradientDifference(d_L_d_W_2_an, d_L_d_W_2_num)
    W_1_difference = ComputeGradientDifference(d_L_d_W_1_an, d_L_d_W_1_num)
    b_2_difference = ComputeGradientDifference(d_L_d_b_2_an, d_L_d_b_2_num)
    b_1_difference = ComputeGradientDifference(d_L_d_b_1_an, d_L_d_b_1_num)
    print("Iteration:", i+1)
    print("W_2_difference:", W_2_difference)
    print("W_1_difference", W_1_difference)
    print("b_2_difference:", b_2_difference)
    print("b_1_difference", b_1_difference)

    #assert b_difference < h
    assert W_2_difference < 1e-5
    assert W_1_difference < 1e-5
    assert b_2_difference < 1e-5
    assert b_1_difference < 1e-5