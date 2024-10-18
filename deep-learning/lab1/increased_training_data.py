import numpy as np
from data_normalization import prepare_data, normalize_data, prepare_five_batches
from functions import ComputeAccuracy, MiniBatchGD
import matplotlib.pyplot as plt
from matlab_functions import displayimages, view_W_matrix
np.random.seed(0)

llambda = 0.01
n_epochs = 40
eta = 0.001
batch_size = 100
step_decay = 30

K = 10 # Number of distinct labels
d = 3072 # Number of datapoints for each image
n = 10000 # Number of images
mu, sigma = 0, 0.01 # mean and standard deviation
num_training_images = 49000
num_batches = (num_training_images // batch_size)
print("num_batches", num_batches)


paths = ["cifar-10-batches-py/data_batch_1", "cifar-10-batches-py/data_batch_2", 
         "cifar-10-batches-py/data_batch_3", "cifar-10-batches-py/data_batch_4",
         "cifar-10-batches-py/data_batch_5"]

train_X, train_Y, train_y, validation_X, validation_Y, validation_y = prepare_five_batches(paths)

test_X, test_Y, test_y = prepare_data("cifar-10-batches-py/test_batch")

train_X, validation_X, test_X = normalize_data(train_X, validation_X, test_X)

#images = np.reshape(validation_X.T, (1000, 32, 32, 3), order='F').transpose((2,1,3,0))

#displayimages(images[:,:,:,:100])

W_values = np.random.normal(mu, sigma, K*d)
b_values = np.random.normal(mu, sigma, K)

W = W_values.reshape((K, d))
b = b_values.reshape((K, 1))

train_X_batch_acc_100 = []
train_Y_batch_acc_100 = []

train_X_batch_acc_50 = []
train_Y_batch_acc_50 = []

train_X_batch_acc_5 = []
train_Y_batch_acc_5 = []

for i in range(49000//100):
    i_start = i * 100
    i_end = (i + 1) * 100
    train_X_batch_acc_100.append(train_X[:, i_start:i_end])
    train_Y_batch_acc_100.append(train_Y[:, i_start:i_end])

train_X_batch_acc_100 = np.asarray(train_X_batch_acc_100)
train_Y_batch_acc_100 = np.asarray(train_Y_batch_acc_100)

params = [llambda, eta, batch_size, num_batches, n_epochs]

X = train_X_batch_acc_100
Y = train_Y_batch_acc_100

W_star, b_star, train_epoch_costs, \
train_epoch_loss, epoch_accuracy, \
validation_epoch_costs, validation_epoch_loss,\
validation_epoch_accuracy = MiniBatchGD(X=X, Y=Y, 
                        GDParams=params, W=W, b=b, train_X=train_X, train_Y=train_Y,
                        train_y=train_y, validation_X=validation_X, validation_Y=validation_Y, validation_y=validation_y, 
                        use_compute_cost_multiple = False,
                        num_epochs_decay_step=step_decay)

test_accuracy = ComputeAccuracy(X=test_X, y=test_y, W=W_star, b=b_star)
validation_accuracy = ComputeAccuracy(X=validation_X, y=validation_y, W=W_star, b=b_star)

print(f"Test accuracy: {test_accuracy}")



