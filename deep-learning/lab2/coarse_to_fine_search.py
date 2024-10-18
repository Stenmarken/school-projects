import numpy as np
from data_normalization import prepare_data, normalize_data, forty_five_k_training_five_k_validation
from matlab_functions import displayimages
from functions import Two_Dimensional_Network
import matplotlib.pyplot as plt
import sys

np.random.seed(2)

d = 3072 # Number of features
K = 10 # Number of classes
m = 50 # Number of nodes

llambda = 0.01
n_epochs = 40
num_batches = 100
eta = 0.001
num_training_images = 45000
batch_size = 100
n_s = np.floor(num_training_images / batch_size)
eta_min = 1e-5
eta_max = 1e-1
num_cycles = 2
l_min = -5
l_max = -1
coarse_search = False

if coarse_search:
    llambdas = np.logspace(l_min, l_max, 8)
    file_path = "coarse_search.txt"
    num_cycles = 2
else:
    file_path = "fine_search.txt"
    llambdas = np.random.uniform(0.00013894954943731373, 0.007196856730011514, 20)
    #llambdas = np.linspace(0.00013894954943731373, 0.007196856730011514, 8)
    num_cycles = 3
print("llambdas", llambdas)


test_X, test_Y, test_y = prepare_data("../lab1/cifar-10-batches-py/test_batch")

paths = ["../lab1/cifar-10-batches-py/data_batch_1", "../lab1/cifar-10-batches-py/data_batch_2", 
         "../lab1/cifar-10-batches-py/data_batch_3", "../lab1/cifar-10-batches-py/data_batch_4",
         "../lab1/cifar-10-batches-py/data_batch_5"]

train_X, train_Y, train_y, validation_X, validation_Y, validation_y = forty_five_k_training_five_k_validation(paths)

train_X, validation_X, test_X = normalize_data(train_X, validation_X, test_X)

network = Two_Dimensional_Network(K=K, d=d, m=m)

train_X_batch_acc = []
train_Y_batch_acc = []

for i in range(num_training_images // num_batches):
    i_start = i * num_batches
    i_end = (i + 1) * num_batches
    train_X_batch_acc.append(train_X[:, i_start:i_end])
    train_Y_batch_acc.append(train_Y[:, i_start:i_end])

train_X_batch_acc = np.asarray(train_X_batch_acc)
train_Y_batch_acc = np.asarray(train_Y_batch_acc)


network.set_parameters(X=train_X_batch_acc, Y=train_Y_batch_acc, train_X=train_X, train_Y=train_Y, train_y=train_y, validation_X=validation_X, validation_Y=validation_Y, validation_y=validation_y,
                       llambda=llambda, eta=eta, batch_size=batch_size, eta_min=eta_min, eta_max=eta_max, n_s=n_s, test_X=test_X, test_y=test_y, num_cycles=num_cycles)

with open(file_path, "w") as f:
    s = " ".join(["llambda", str(llambda), "l_min", str(l_min), "l_max", str(l_max), "n_s", str(n_s),
                  "num_cycles", str(num_cycles), "\n"])
    f.write(s)
    for l in llambdas:
        network.llambda = l
        t_list = network.MiniBatchGD()

        s = " ".join(["Llambda", str(l), "Test accuracy", str(max(network.validation_accuracy)), "\n"])
        network.reset_network()
        f.write(s)