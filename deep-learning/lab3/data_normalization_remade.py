from matlab_functions_remade import load_batch
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