import numpy as np
from data_normalization import prepare_data, normalize_data, prepare_five_batches
from functions import ComputeAccuracy, MiniBatchGD
import matplotlib.pyplot as plt
from matlab_functions import displayimages, view_W_matrix
from itertools import product


def combinations():
    parameter_grid = {
        'llambda': [0.1, 0.01, 0.001],
        'eta': [0.001, 0.005, 0.01],
        'batch_size' : [5, 50, 100]
        }
    # Each list in param_combinations is of form [llambda, eta, batch_size]
    param_combinations = list(product(parameter_grid['llambda'], 
                                      parameter_grid['eta'], 
                                      parameter_grid['batch_size']))
    return param_combinations


np.random.seed(0)

llambda = 0
n_epochs = 40
eta = 0.001
# Divide the learning rate by 10 every num_epochs_decay_step epochs
num_epochs_decay_step = 20 
batch_size = 100

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

for i in range(49000//50):
    i_start = i * 50
    i_end = (i + 1) * 50
    train_X_batch_acc_50.append(train_X[:, i_start:i_end])
    train_Y_batch_acc_50.append(train_Y[:, i_start:i_end])

train_X_batch_acc_50 = np.asarray(train_X_batch_acc_50)
train_Y_batch_acc_50 = np.asarray(train_Y_batch_acc_50)

for i in range(49000//5):
    i_start = i * 5
    i_end = (i + 1) * 5
    train_X_batch_acc_5.append(train_X[:, i_start:i_end])
    train_Y_batch_acc_5.append(train_Y[:, i_start:i_end])

train_X_batch_acc_5 = np.asarray(train_X_batch_acc_5)
train_Y_batch_acc_5 = np.asarray(train_Y_batch_acc_5)

GDParams = combinations()
parameter_scores = []

for i, params in enumerate(GDParams):
    print(f"Parameter combination #{i+1}")
    params += (num_batches, n_epochs)

    if params[2] == 5:
        X = train_X_batch_acc_5
        Y = train_Y_batch_acc_5
    elif params[2] == 50:
        X = train_X_batch_acc_50
        Y = train_Y_batch_acc_50
    else: 
        X = train_X_batch_acc_100
        Y = train_Y_batch_acc_100

    W_star, b_star, train_epoch_costs, \
    train_epoch_loss, epoch_accuracy, \
    validation_epoch_costs, validation_epoch_loss,\
    validation_epoch_accuracy = MiniBatchGD(X=X, Y=Y, 
                                GDParams=params, W=W, b=b, train_X=train_X, train_Y=train_Y,
                                train_y=train_y, validation_X=validation_X, validation_Y=validation_Y, validation_y=validation_y,
                                num_epochs_decay_step=num_epochs_decay_step)

    test_accuracy = ComputeAccuracy(X=test_X, y=test_y, W=W_star, b=b_star)
    validation_accuracy = ComputeAccuracy(X=validation_X, y=validation_y, W=W_star, b=b_star)
    parameter_scores.append((validation_accuracy, test_accuracy, params))
    print("Accuracy:", test_accuracy)

parameter_scores.sort(reverse = True)
with open("parameters.txt", "w") as f:
    f.write("Validation, testing, llambda, eta, batch_size, num_batches, num_epochs\n")
    for val_acc, test_acc,  parameter_set in parameter_scores:
        s = " ".join(["Validation:", str(val_acc), "Testing:", str(test_acc), "Parameters:", str(parameter_set) ])
        f.write(s)
        f.write("\n")

# plt.plot(train_epoch_costs, label='Training cost')
# plt.plot(validation_epoch_costs, label="Validation cost")
# plt.legend()
# plt.title("Plot of training and validation cost")
# plt.xlabel('Epoch number')
# plt.ylabel('Cost')
# plt.title(f"Cost \n lambda={llambda}, n_epochs={n_epochs}, batch_size={batch_size}, eta={eta}")
# plt.show()
# plt.figure() 

# plt.plot(train_epoch_loss, label='Training loss')
# plt.plot(validation_epoch_loss, label="Validation loss")
# plt.legend()
# plt.suptitle(f"Loss \n lambda={llambda}, n_epochs={n_epochs}, batch_size={batch_size}, eta={eta}")
# plt.xlabel('Epoch number')
# plt.ylabel('Loss')
# plt.show()
# plt.figure() 

#plt.plot(epoch_accuracy)
#plt.plot(validation_epoch_accuracy)
#plt.xlabel('Epoch number')
#plt.ylabel('Accuracy')
#plt.show()
#plt.figure() 