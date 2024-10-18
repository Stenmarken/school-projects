from matlab_functions import load_batch
import matplotlib.pyplot as plt
import numpy as np

def one_hot_encode_labels(labels):
    Y = []

    for label in labels:
        arr = [0] * 10
        arr[label] = 1
        Y.append(arr)
    
    Y_np = np.asanyarray(Y)
    return Y_np.T

def prepare_data(path):
    dataset = load_batch(path)
    
    X = np.array(dataset[b'data']).T
    X = X.reshape(3072, 10000)

    X = X.astype(float) / 255 # Images with pixel values in between 0 and 1
    y = np.array(dataset[b'labels']) # Labels (0-9)

    Y = one_hot_encode_labels(y)
    return (X, Y, y)    

def normalize_data(train_X, validation_X, test_X):
    mean_X = np.reshape(np.mean(train_X, axis=1), (3072, 1))
    std_X = np.reshape(np.std(train_X, axis=1), (3072, 1))
    
    train_X -= mean_X
    train_X /= std_X
    validation_X -= mean_X
    validation_X /= std_X
    test_X -= mean_X
    test_X /= std_X 
    return (train_X, validation_X, test_X)


def forty_five_k_training_five_k_validation(paths, forty_nine_k_validation = False):
    first_batch = load_batch(paths[0])
    X = np.array(first_batch[b'data']).T
    y = np.array(first_batch[b'labels']) 

    for path in paths[1:4]:
        dataset = load_batch(path)
        X = np.concatenate((X, np.array(dataset[b'data']).T), axis=1)
        y = np.append(y, np.array(dataset[b'labels']))

    last_batch = load_batch(paths[-1])
    X_val = np.array(last_batch[b'data']).T
    y_val = np.array(last_batch[b'labels'])

    cutoff = 5000
    if forty_five_k_training_five_k_validation:
        cutoff = 9000

    X = np.concatenate((X, X_val[:, :cutoff]), axis=1)
    y = np.concatenate((y, y_val[:cutoff]))
    X_val = X_val[:, cutoff:]
    y_val = y_val[cutoff:]

    X = X.astype(float) / 255 
    X_val = X_val.astype(float) / 255 

    Y = one_hot_encode_labels(y)
    Y_val = one_hot_encode_labels(y_val)
    print("X", np.shape(X))
    print("Y", np.shape(Y))
    print("y", np.shape(y))
    print("X_val", np.shape(X_val))
    print("Y_val", np.shape(Y_val))
    print("y_val", np.shape(y_val))
    return X, Y, y, X_val, Y_val, y_val