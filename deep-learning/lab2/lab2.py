import numpy as np
from data_normalization import prepare_data, normalize_data
from matlab_functions import displayimages
from functions import Two_Dimensional_Network
import matplotlib.pyplot as plt

np.random.seed(2)

d = 3072 # Number of features
K = 10 # Number of classes
m = 50 # Number of nodes

llambda = 0.01
n_epochs = 40
num_batches = 100
eta = 0.001
num_images = 10000
batch_size = 100
n_s = 800
eta_min = 1e-5
eta_max = 1e-1
num_cycles = 3

# Y is one-hot encoded, y is not
train_X, train_Y, train_y = prepare_data("../lab1/cifar-10-batches-py/data_batch_1")
validation_X, validation_Y, validation_y = prepare_data("../lab1/cifar-10-batches-py/data_batch_2")
test_X, test_Y, test_y = prepare_data("../lab1/cifar-10-batches-py/test_batch")

train_X, validation_X, test_X = normalize_data(train_X, validation_X, test_X)

network = Two_Dimensional_Network(K=K, d=d, m=m)

train_X_batch_acc = []
train_Y_batch_acc = []

for i in range(num_images // num_batches):
    i_start = i * num_batches
    i_end = (i + 1) * num_batches
    train_X_batch_acc.append(train_X[:, i_start:i_end])
    train_Y_batch_acc.append(train_Y[:, i_start:i_end])

train_X_batch_acc = np.asarray(train_X_batch_acc)
train_Y_batch_acc = np.asarray(train_Y_batch_acc)

print("train_X_batch_acc", np.shape(train_X_batch_acc))
print("validation_X", np.shape(validation_X))

network.set_parameters(X=train_X_batch_acc, Y=train_Y_batch_acc, train_X=train_X, train_Y=train_Y, train_y=train_y, validation_X=validation_X, validation_Y=validation_Y, validation_y=validation_y,
                       llambda=llambda, eta=eta, batch_size=batch_size, eta_min=eta_min, eta_max=eta_max, n_s=n_s, test_X=test_X, test_y=test_y, num_cycles=num_cycles)

t_list = network.MiniBatchGD()

test_acc = network.ComputeAccuracy(use_training_data=False, use_test_data=True)

print("Accuracy on test dataset:", test_acc)

plt.plot(network.learning_rates)
plt.show()
plt.figure()

plt.plot(t_list, network.train_costs, label='Training cost')
plt.plot(t_list, network.validation_costs, label="Validation cost")
plt.legend()
plt.title("Plot of training and validation cost")
plt.xlabel('Update step')
plt.ylabel('Cost')
plt.title(f"Cost \n lambda={llambda}, n_s={n_s}, batch_size={batch_size}, eta_min={eta_min}, eta_max={eta_max}")
plt.show()
plt.figure() 

plt.plot(t_list, network.train_losses, label='Training loss')
plt.plot(t_list, network.validation_losses, label="Validation loss")
plt.legend()
plt.title(f"Loss \n lambda={llambda}, n_s={n_s}, batch_size={batch_size}, eta_min={eta_min}, eta_max={eta_max}")
plt.xlabel('Update step')
plt.ylabel('Loss')
plt.show()
plt.figure() 

plt.plot(t_list, network.train_accuracy, label="Training accuracy")
plt.plot(t_list, network.validation_accuracy, label="Validation accuracy")
plt.legend()
plt.title(f"Accuracy \n lambda={llambda}, n_s={n_s}, batch_size={batch_size}, eta_min={eta_min}, eta_max={eta_max}")
plt.xlabel('Update step')
plt.ylabel('Accuracy')
plt.show()
plt.figure() 