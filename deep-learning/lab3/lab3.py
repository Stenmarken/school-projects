from data_normalization_remade import prepare_data, normalize_data
from N_dimensional_network_remade import N_dimensional_network
import yaml
import numpy as np

def read_parameters(path):
    with open(path, "r") as f:
        parameters = yaml.safe_load(f)
    for param in ['d', 'K', 'm']:
        assert param in parameters.keys()
    return parameters

def read_data(parameters):
    # Y is one-hot encoded, y is not
    X_train, Y_train, y_train = prepare_data("../lab1/cifar-10-batches-py/data_batch_1")
    X_val, Y_val, y_val = prepare_data("../lab1/cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = prepare_data("../lab1/cifar-10-batches-py/test_batch")
    
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    train_X_batch_acc = []
    train_Y_batch_acc = []

    for i in range(10000 // parameters['batch_size']):
        i_start = i * parameters['batch_size']
        i_end = (i + 1) * parameters['batch_size']
        train_X_batch_acc.append(X_train[:, i_start:i_end])
        train_Y_batch_acc.append(Y_train[:, i_start:i_end])

    train_X_batch_acc = np.asarray(train_X_batch_acc)
    train_Y_batch_acc = np.asarray(train_Y_batch_acc)

    #print("X_train.shape", X_train.shape)
    #print("Y_train.shape", Y_train.shape)
    #print("train_X_batch_acc.shape", train_X_batch_acc.shape) 
    #print("train_Y_batch_acc.shape", train_Y_batch_acc.shape)

    return ((X_train, Y_train, y_train),
            (X_val, Y_val, y_val),
            (X_test, Y_test, y_test),
            (train_X_batch_acc, train_Y_batch_acc))

def main():
    parameters = read_parameters("configs/parameters.yaml")
    train_data, val_data, test_data, data_batches = read_data(parameters)
    n_dim = N_dimensional_network(parameters=parameters, 
                                          data_batches=data_batches,
                                          train_data=train_data, 
                                          val_data=val_data,
                                          test_data=test_data)
    n_dim.mini_batch_gd()

def coarse_to_fine_search():
    llambdas = np.logspace(-5, -1, 8)
    for llambda in reversed(llambdas):
        parameters = read_parameters("configs/parameters.yaml")
        train_data, val_data, test_data, data_batches = read_data(parameters)
        n_dim = N_dimensional_network(parameters=parameters, 
                                            data_batches=data_batches,
                                            train_data=train_data, 
                                            val_data=val_data,
                                            test_data=test_data)
        n_dim.llambda = llambda
        print(f"llambda: {llambda}")
        n_dim.mini_batch_gd()
        print("\n\n")
    
def sensitivity_to_initialization():
    sigs = [1e-1, 1e-3, 1e-4]
    for sig in sigs:
        parameters = read_parameters("configs/parameters.yaml")
        train_data, val_data, test_data, data_batches = read_data(parameters)
        n_dim = N_dimensional_network(parameters=parameters, 
                                            data_batches=data_batches,
                                            train_data=train_data, 
                                            val_data=val_data,
                                            test_data=test_data,
                                            sig=sig)
        print(f"sig: {sig}")
        n_dim.mini_batch_gd()
        print("\n\n")

if __name__ == "__main__":
    main()
    #coarse_to_fine_search()
    #sensitivity_to_initialization()