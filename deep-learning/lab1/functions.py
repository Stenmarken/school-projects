import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(x): 
    return np.exp(x) / (np.exp(x) + 1)

def EvaluateClassifier(X, W, b, use_sigmoid=False):
    s = np.matmul(W, X)
    if s.ndim == 1:
        s = np.reshape(s, (s.size, 1))
    s_som = np.add(s, b)
    if use_sigmoid:
        return sigmoid(s_som)
    else:
        return softmax(s_som)

def ComputeMultipleBinaryCrossEntropyLoss(X, Y, W, b):
    num_classes = 10
    s = 0
    p = EvaluateClassifier(X=X, W=W, b=b, use_sigmoid=True)
    for k in range(num_classes):
        s += (1 - Y[k]) * np.log(1 - p[k]) + Y[k] * np.log(p[k])
    res = s[0] # s becomes for some reason a numpy array with shape (1, )
    return (-res) / num_classes

def ComputeCostMultiple(X, Y, W, b, llambda):
    """
    Assumes the labels of Y are one-hot encoded
    """
    # Check that the labels of Y are one-hot encoded
    assert np.shape(Y)[0] == 10

    sum_l_cross = 0
    loss = 0
    cost = 0
    _, X_cols = np.shape(X)
    for i, column in enumerate(X.T):
        l_cross_i = ComputeMultipleBinaryCrossEntropyLoss(X=column, Y=Y.T[i], W=W, b=b)
        sum_l_cross += l_cross_i
        loss += l_cross_i
    
    loss /= X_cols
    cost = loss + llambda * np.sum(W ** 2)
    return cost, loss

def ComputeCost(X, Y, W, b, llambda):
    """
    Assumes the labels of Y are one-hot encoded
    """
    # Check that the labels of Y are one-hot encoded
    assert np.shape(Y)[0] == 10

    sum_l_cross = 0
    loss = 0
    cost = 0
    _, X_cols = np.shape(X)
    for i, column in enumerate(X.T):
        p_i = EvaluateClassifier(X=column, W=W, b=b)
        y_i = np.transpose(Y[:, i])
        p_i = np.reshape(p_i, (10,))
        l_cross_i = -np.dot(y_i, np.log(p_i))
        sum_l_cross += l_cross_i
        loss += l_cross_i
    
    loss /= X_cols
    cost = loss + llambda * np.sum(W ** 2)
    return cost, loss
    
def ComputeAccuracy(X, y, W, b):
    num_correct = 0
    _, X_cols = np.shape(X)
    for i, column in enumerate(X.T):
        p_i = EvaluateClassifier(X=column, W=W, b=b)
        print("p_i", np.shape(p_i))
        index = np.argmax(p_i)
        if y[i] == index:
            num_correct += 1
    return num_correct / X_cols

def ComputeAccuracyAndHistogram(X, y, W, b):
    num_correct = 0
    correctly_labeled = [0] * 10
    incorrectly_labeled = [0] * 10
    _, X_cols = np.shape(X)
    for i, column in enumerate(X.T):
        p_i = EvaluateClassifier(X=column, W=W, b=b)
        index = np.argmax(p_i)
        if y[i] == index:
            correctly_labeled[y[i]] += 1
            num_correct += 1
        else:
            incorrectly_labeled[y[i]] += 1
    return num_correct / X_cols, correctly_labeled, incorrectly_labeled

def ComputeGradients(X, Y, P, W, llambda, b, use_sigmoid = False):
    """ 
    Should this function take the bias vector b as input?
    """
    _, n = np.shape(X)
    ones_col = np.ones(n).reshape((n, 1))
    intermediary = np.matmul(W, X) + np.matmul(b, ones_col.T)
    # Forward pass
    if use_sigmoid:
        p_batch = sigmoid(intermediary)
    else:
        p_batch = softmax(intermediary)
    
    # Backward pass
    g_batch = -(Y - p_batch)
    grad_L_W = np.matmul(g_batch, X.T) / n
    grad_L_b = np.matmul(g_batch, ones_col) / n
    
    grad_W = grad_L_W + 2 * llambda * W
    grad_b = grad_L_b
    if use_sigmoid:
        grad_W /= 10
        grad_b /= 10
    return [grad_W, grad_b]

def ComputeGradientDifference(g_a, g_n, eps = 1e-12):
    return np.sum(np.absolute(g_a - g_n)) / np.sum(np.maximum(eps, np.absolute(g_a) + np.absolute(g_n)))

# Code taken from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def log_epoch(epoch_num, train_X, train_Y, train_y, validation_X, validation_Y, validation_y, W, b, llambda,
              use_compute_cost_multiple = False):
        print("Epoch", epoch_num)
        if use_compute_cost_multiple:
            train_cost, train_loss = ComputeCostMultiple(X=train_X, Y=train_Y, W = W, b=b, llambda=llambda)
            validation_cost, validation_loss = ComputeCostMultiple(X=validation_X, Y=validation_Y, W = W, b=b, llambda=llambda)
        else:
            train_cost, train_loss = ComputeCost(X=train_X, Y=train_Y, W = W, b=b, llambda=llambda)
            validation_cost, validation_loss = ComputeCost(X=validation_X, Y=validation_Y, W = W, b=b, llambda=llambda)
        print("Training cost:", train_cost)
        print("Training loss", train_loss)
        print("Validation cost:", validation_cost)
        print("Validation loss", validation_loss)
        training_accuracy = ComputeAccuracy(X=train_X, y=train_y, W=W, b=b)
        validation_accuracy = ComputeAccuracy(X=validation_X, y=validation_y, W=W, b=b)
        print("Training accuracy:", training_accuracy)
        print("Validation accuracy:", validation_accuracy, "\n")
        return (train_cost, train_loss, training_accuracy, 
                validation_cost, validation_loss, validation_accuracy)

def MiniBatchGD(X, Y, GDParams, W, b, train_X, train_Y, train_y, validation_X, validation_Y, 
                validation_y, num_epochs_decay_step=1e6, use_compute_cost_multiple = False):
    """ Performs Mini batch gradient descent with respect to W and b

    Args:
        X (numpy(n_batch * a * b)): n_batch is the number of batches, a the number of features, 
            b the number of images in a batch.
        Y (numpy(n_batch * K * b)): n_batch is the number of batches, 
        K the number of possible labels, b is the number of images in a batch.
        GDParams ([n_batch, eta, n_pochs]): n_batch is the number of batches, 
            eta is the learning rate, n_pochs is the number of epochs
        W (numpy): Weight matrix
        b (numpy): bias vector
        llambda (float): Regularisation multiplies

    Returns:
        W (numpy) : Updated weight matrix
        b (numpy) : Updated bias vector
    """
    train_epoch_costs = []
    train_epoch_loss = []
    validation_epoch_costs = []
    validation_epoch_loss = []
    epoch_accuracy = []
    validation_epoch_accuracy = []

    llambda, eta, batch_size, num_batches, num_epochs = GDParams
    for epoch_index in range(num_epochs):
        if (epoch_index + 1) % num_epochs_decay_step == 0:
            eta *= 0.1  
        X, Y = unison_shuffled_copies(X, Y)
        for i in range(num_batches):
            d_J_d_W, d_J_d_b = ComputeGradients(X=X[i], Y=Y[i], P=[], W=W, llambda=llambda, b=b, use_sigmoid=use_compute_cost_multiple)
            
            W = W - eta * d_J_d_W
            b = b - eta * d_J_d_b

        train_cost, train_loss, training_accuracy, validation_cost, \
            validation_loss, validation_accuracy = log_epoch(epoch_index+1, train_X, train_Y, train_y, validation_X, validation_Y, validation_y, W, b, llambda, use_compute_cost_multiple)
        
        train_epoch_costs.append(train_cost)
        train_epoch_loss.append(train_loss)
        validation_epoch_costs.append(validation_cost)
        validation_epoch_loss.append(validation_loss)
        epoch_accuracy.append(training_accuracy)
        validation_epoch_accuracy.append(validation_accuracy)

    return W, b, train_epoch_costs, train_epoch_loss, epoch_accuracy, \
        validation_epoch_costs, validation_epoch_loss, validation_epoch_accuracy

"""
This code is a modified version of the top answer to this Stack overflow question: 
https://stackoverflow.com/questions/10369681/how-to-plot-bar-graphs-with-same-x-coordinates-side-by-side-dodged   
"""
def plot_side_by_side(correctly_labeled, incorrectly_labeled, plot_title):
    print("Correctly labeled")
    for number in correctly_labeled:
        print(f"{number:.4f} &")
    print("\nIncorrectly labeled")
    for number in incorrectly_labeled:
        print(f"{number:.4f} &")
    
    # Numbers of pairs of bars you want
    N = 10

    # Position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10,5))

    # Width of a bar 
    width = 0.3       

    # Plotting
    plt.bar(ind, correctly_labeled , width, label='Correctly labeled')
    plt.bar(ind + width, incorrectly_labeled, width, label='Incorrectly labeled')

    plt.xlabel('Image index')
    plt.ylabel('Probability for the ground truth class')
    plt.title(plot_title)

    # xticks()
    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    plt.xticks(ind + width / 2, list(map(str, ind)))

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    plt.show()