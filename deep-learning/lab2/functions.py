import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

np.random.seed(2)

class Two_Dimensional_Network():
    def __init__(self, K, d, m, dropout_rate=0.0):
        W_1 = np.random.normal(0.0, 1 / (np.sqrt(d)), m * d)
        W_2 = np.random.normal(0.0, 1 / (np.sqrt(m)), K * m)
        self.dropout_rate = dropout_rate

        self.W_1 = W_1.reshape((m, d))
        self.W_2 = W_2.reshape((K, m))
        self.b_1 = np.zeros((m, 1))
        self.b_2 = np.zeros((K, 1))
        self.train_costs = []
        self.train_losses = []
        self.train_accuracy = []
        
        self.validation_costs = []
        self.validation_losses = []
        self.validation_accuracy = []

    def reset_network(self):
        self.train_costs = []
        self.train_losses = []
        self.train_accuracy = []
        
        self.validation_costs = []
        self.validation_losses = []
        self.validation_accuracy = []

    def set_parameters(self, X, Y, train_X, train_Y, train_y, validation_X, validation_Y, 
                validation_y, test_X, test_y, llambda, eta, batch_size, eta_min=1e-5, eta_max=1e-1, n_s=500, num_cycles=1, 
                printings_per_cycle = 9) -> None:
        self.X = X
        self.Y = Y
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_y = train_y
        self.validation_X = validation_X
        self.validation_Y = validation_Y
        self.validation_y = validation_y
        self.test_X = test_X
        self.test_y = test_y
        self.llambda = llambda
        self.eta = eta
        self.batch_size = batch_size
        self.eta_min = eta_min 
        self.eta_max = eta_max
        self.n_s = n_s
        self.num_cycles = num_cycles
        self.num_batches = int(np.shape(train_X)[1] / self.batch_size)
        self.printings_per_cycle = printings_per_cycle

    def EvaluateClassifier(self, x):
        Ws = [self.W_1, self.W_2]
        bs = [self.b_1, self.b_2]
        ss = []
        hs = []
        s_1 = Ws[0] @ x + bs[0]
        h = np.maximum(s_1, 0)
        ss.append(s_1)
        hs.append(h)

        for i in range(1, len(Ws)-1):
            s_i = Ws[i] @ h + bs[i]
            print("s_i.shape", s_i.shape)
            h = np.maximum(s_i, 0)
            ss.append(s_i)
            hs.append(h)

        s_k = Ws[-1] @ h + bs[-1]
        p = softmax(s_k)
        ss.append(s_k)

        return p, ss[1], hs, s_1[0]

    def ComputeCost(self, use_training_data = True):
        if use_training_data:
            X = self.train_X
            Y = self.train_Y
            y = self.train_y
        else:
            X = self.validation_X
            Y = self.validation_Y
            y = self.validation_y
        _, X_cols = np.shape(X)
        p, s, h, s_1 = self.EvaluateClassifier(X)
        loss = np.sum(-np.multiply(Y, np.log(p))) / X_cols 
        cost = loss + self.llambda * (np.sum(self.W_1 ** 2) + np.sum(self.W_2 ** 2))
        return cost, loss
    
    def ComputeAccuracy(self, use_training_data = True, use_test_data=False):
        if use_training_data:
            X = self.train_X
            y = self.train_y
        elif use_test_data:
            X = self.test_X
            y = self.test_y
        else:
            X = self.validation_X
            y = self.validation_y
        _, X_cols = np.shape(X)
        p, s, h, s_1 = self.EvaluateClassifier(X)
        indices = np.argmax(p, axis=0)
        return np.sum(y == indices) / X_cols       

    def apply_dropout(self, W):
        mask = np.ndarray(np.shape(W))

    def ComputeGradients(self, X, Y):
        _, n_b = X.shape

        Ws = [self.W_1, self.W_2]
        dLdW = []
        dLdb = []
        
        p, s, hs, s_1 = self.EvaluateClassifier(X)
        g_batch = -(Y - p)

        for i in range(len(Ws)-1, 0, -1):
            dLdW.insert(0, (1/n_b)*g_batch @ hs[i-1].T)
            print("(1/n_b)*g_batch @ hs[i-1].T", ((1/n_b)*g_batch @ hs[i-1].T).shape)
            dLdb.insert(0, (1/n_b)*g_batch @ np.ones((n_b, 1)))
            g_batch = Ws[-1].T @ g_batch
            g_batch = g_batch * np.int64(hs[i-1] > 0)
            
        dLdW.insert(0, (1/n_b)*g_batch @ X.T)
        dLdb.insert(0, (1/n_b)*g_batch @ np.ones((n_b, 1)))
        
        assert dLdW[0].shape == self.W_1.shape
        assert dLdW[1].shape == self.W_2.shape
        assert dLdb[0].shape == self.b_1.shape
        assert dLdb[1].shape == self.b_2.shape

        return dLdW, dLdb

    def ComputeGradients2(self, X, Y):
        _, n_b = X.shape 
        p, s, h, s_1 = self.EvaluateClassifier(X)
        g_batch = -(Y - p)
        d_L_d_W_2 = (1 / n_b) * g_batch @ h[0].T
        d_L_d_b_2 = (1 / n_b) * g_batch @ np.ones((n_b, 1))

        g_batch = self.W_2.T @ g_batch
        g_batch = g_batch * np.int64(h[0] > 0)
        d_L_d_W_1 = (1 / n_b) * g_batch @ X.T
        d_L_d_b_1 = (1 / n_b) * g_batch @ np.ones((n_b, 1))
        return d_L_d_W_2, d_L_d_b_2, d_L_d_W_1, d_L_d_b_1

    def log_epoch(self, t):
            print("Update step", t)
            train_cost, train_loss = self.ComputeCost(use_training_data=True)
            self.train_costs.append(train_cost)
            self.train_losses.append(train_loss)
            validation_cost, validation_loss = self.ComputeCost(use_training_data=False)
            self.validation_costs.append(validation_cost)
            self.validation_losses.append(validation_loss)
            print("Training cost:", train_cost)
            print("Training loss", train_loss)
            print("Validation cost:", validation_cost)
            print("Validation loss", validation_loss)
            training_accuracy = self.ComputeAccuracy(use_training_data=True)
            self.train_accuracy.append(training_accuracy)
            validation_accuracy = self.ComputeAccuracy(use_training_data=False)
            self.validation_accuracy.append(validation_accuracy)
            print("Training accuracy:", training_accuracy)
            print("Validation accuracy:", validation_accuracy, "\n")
    
    def cyclical_learning_rate(self, t):
        def find_first(l_set, bool_function):
            for l in l_set:
                if bool_function(l):
                    return l
            return None

        def check_t_first(l, n_s=self.n_s, t=t):
            return 2 * l * n_s <= t and t <= (2*l + 1) * n_s
        
        def check_t_second(l, n_s=self.n_s, t=t):
            return (2*l + 1)*n_s <= t and t <= 2*(l + 1) * n_s

        l_set = [i for i in range(10)] # TODO: Fix this hard coded value

        l = find_first(l_set, check_t_first)

        if l != None:
            self.eta = self.eta_min + ((t - 2*l*self.n_s)/self.n_s) * (self.eta_max - self.eta_min)
            return
        l = find_first(l_set, check_t_second)
        if l != None:
            self.eta = self.eta_max - ((t - (2*l+1)*self.n_s) / self.n_s) * (self.eta_max - self.eta_min) 
            return

    def update_matrices(self, dLdW, dLdb):
        Ws = [self.W_1, self.W_2]
        bs = [self.b_1, self.b_2]

        for i in range(len(Ws)):
            Ws[i] = Ws[i] - (self.eta * dLdW[i])
            bs[i] = bs[i] - (self.eta * dLdb[i])

        self.W_1 = Ws[0]
        self.W_2 = Ws[1]
        self.b_1 = bs[0]
        self.b_2 = bs[1]

    def MiniBatchGD(self):
        t_list = []
        self.learning_rates = []
        
        t = 0
        #t_list.append(t)
        a = 1
        val = np.floor(2*self.n_s / (self.printings_per_cycle))
        while (t < 2 * self.num_cycles * self.n_s):
            X, Y = unison_shuffled_copies(self.X, self.Y)

            for i in range(self.num_batches):
                if t % val == 0: 
                    self.log_epoch(t)
                    t_list.append(t)
                    print("Printing #", a)
                    a += 1

                dLdW, dLdb= self.ComputeGradients(X=X[i], Y=Y[i])
                self.cyclical_learning_rate(t)
                assert(self.eta <= self.eta_max and self.eta >= self.eta_min)
                self.learning_rates.append(self.eta)

                self.update_matrices(dLdW, dLdb)

                #self.W_1 = self.W_1 - self.eta * d_L_d_W_1
                #self.W_2 = self.W_2 - self.eta * d_L_d_W_2
                #self.b_1 = self.b_1 - self.eta * d_L_d_b_1
                #self.b_2 = self.b_2 - self.eta * d_L_d_b_2

                t += 1
            
        return t_list

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(x): 
    return np.exp(x) / (np.exp(x) + 1)

def ComputeGradientDifference(g_a, g_n, eps = 1e-12):
    return np.sum(np.absolute(g_a - g_n)) / np.sum(np.maximum(eps, np.absolute(g_a) + np.absolute(g_n)))

# Code taken from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    print("a", len(a))
    print("b", len(b))
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]