import numpy as np
from data_normalization import prepare_data, normalize_data
from functions import ComputeAccuracy, MiniBatchGD, ComputeAccuracyAndHistogram, plot_side_by_side
import matplotlib.pyplot as plt
from matlab_functions import displayimages, view_W_matrix
import scipy.io

np.random.seed(0)

llambda = 0.0
n_epochs = 40
num_batches = 100
eta = 0.02


K = 10 # Number of distinct labels
d = 3072 # Number of datapoints for each image
n = 10000 # Number of images
mu, sigma = 0, 0.01 # mean and standard deviation
num_images = 10000
batch_size = 100

# Y is one-hot encoded, y is not
train_X, train_Y, train_y = prepare_data("cifar-10-batches-py/data_batch_1")
validation_X, validation_Y, validation_y = prepare_data("cifar-10-batches-py/data_batch_2")
test_X, test_Y, test_y = prepare_data("cifar-10-batches-py/test_batch")

train_X, validation_X, test_X = normalize_data(train_X, validation_X, test_X)

#images = np.reshape(train_X.T, (10000, 32, 32, 3), order='F').transpose((2,1,3,0))

#displayimages(images[:,:,:,:100])

W_values = np.random.normal(mu, sigma, K*d)
b_values = np.random.normal(mu, sigma, K)

W = W_values.reshape((K, d))
b = b_values.reshape((K, 1))

train_X_batch_acc = []
train_Y_batch_acc = []

for i in range(num_images // num_batches):
    i_start = i * num_batches
    i_end = (i + 1) * num_batches
    train_X_batch_acc.append(train_X[:, i_start:i_end])
    train_Y_batch_acc.append(train_Y[:, i_start:i_end])

train_X_batch_acc = np.asarray(train_X_batch_acc)
train_Y_batch_acc = np.asarray(train_Y_batch_acc)

GDParams = [llambda, eta, batch_size, num_batches, n_epochs]

W_star, b_star, train_epoch_costs, \
train_epoch_loss, epoch_accuracy, \
validation_epoch_costs, validation_epoch_loss,\
validation_epoch_accuracy = MiniBatchGD(X=train_X_batch_acc, Y=train_Y_batch_acc, 
                             GDParams=GDParams, W=W, b=b, train_X=train_X, train_Y=train_Y,
                             train_y=train_y, validation_X=validation_X, validation_Y=validation_Y, validation_y=validation_y, 
                             use_compute_cost_multiple= True)

#accuracy = ComputeAccuracy(X=test_X, y=test_y, W=W_star, b=b_star)
accuracy, correctly_labeled, incorrectly_labeled = ComputeAccuracyAndHistogram(X=test_X, y=test_y, W=W_star, b=b_star)

sum_correctly_labeled = sum(correctly_labeled)
sum_incorrectly_labeled = sum(incorrectly_labeled)

correctly_labeled = list(map(lambda x: x / sum_correctly_labeled, correctly_labeled))
incorrectly_labeled = list(map(lambda x: x / sum_incorrectly_labeled, incorrectly_labeled))

plot_side_by_side(correctly_labeled=correctly_labeled, incorrectly_labeled=incorrectly_labeled,
                  plot_title=f"Histogram for multiple binary cross-entropy loss, llambda={llambda}, eta={eta}, batch_size={batch_size}\n, num_batches={num_batches}, n_epochs={n_epochs}")

#labels = [i for i in range(10)]

#fig, ax = plt.subplots()
#ax.bar(labels, correctly_labeled)
#plt.show()

print("Accuracy:", accuracy)

view_W_matrix(W_star, (llambda, n_epochs, num_batches, eta))

plt.plot(train_epoch_costs, label='Training cost')
plt.plot(validation_epoch_costs, label="Validation cost")
plt.legend()
plt.title("Plot of training and validation cost")
plt.xlabel('Epoch number')
plt.ylabel('Cost')
plt.title(f"Cost \n lambda={llambda}, n_epochs={n_epochs}, num_batches={num_batches}, eta={eta}")
plt.show()
plt.figure() 

plt.plot(train_epoch_loss, label='Training loss')
plt.plot(validation_epoch_loss, label="Validation loss")
plt.legend()
plt.suptitle(f"Loss \n lambda={llambda}, n_epochs={n_epochs}, num_batches={num_batches}, eta={eta}")
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()
plt.figure() 

plt.plot(epoch_accuracy, label="Training accuracy")
plt.plot(validation_epoch_accuracy, label="Validation accuracy")
plt.legend()
plt.xlabel('Epoch number')
plt.suptitle(f"Multiple binary cross-entropy. Accuracy \n lambda={llambda}, n_epochs={n_epochs}, num_batches={num_batches}, eta={eta}")
plt.ylabel('Accuracy')
plt.show()
plt.figure() 

