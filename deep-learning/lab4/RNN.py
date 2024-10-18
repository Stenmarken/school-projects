from typing import Dict
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

class RNN:
    def __init__(self, seed=0) -> None:
        np.random.seed(seed)
        self.file_path = "goblet_book.txt"
        self.construct_maps()
        self.initialize_model()

    def initialize_model(self):
        self.m = 100
        self.seq_length = 25
        self.sig = 0.01
        self.num_epochs = 2
        self.update_steps_one_epoch = math.floor(self.num_chars / self.seq_length)
        self.epsilon = 1e-8
        self.has_computed_first_loss = False
        self.gamma = 0.9
        self.eta = 0.001

        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))
        self.U = np.random.normal(0, 1, (self.m, self.K)) * self.sig
        self.W = np.random.normal(0, 1, (self.m, self.m)) * self.sig
        self.V = np.random.normal(0, 1, (self.K, self.m)) * self.sig

        self.parameters = {"V" : self.V, "W" : self.W, "U" : self.U}
        self.e = 0 # Index for where we are in the training data
        self.last_computed_h = np.zeros((self.m, 1)) # The last computed hidden state, initially 0-vector
        self.h_prev = np.zeros((self.m, 1))
        self.V_m = np.zeros_like(self.V)
        self.W_m = np.zeros_like(self.W)
        self.U_m = np.zeros_like(self.U)
        self.b_m = np.zeros_like(self.b)
        self.c_m = np.zeros_like(self.c)

    def read_text_file(self, file_path):
        with open(file_path, "r") as f:
            f_str = f.read()
            char_set = list(set(f_str))
        return f_str, char_set

    def generate_one_hot_encoding(self, ind):
        x_0 = [0] * self.K
        x_0[ind] = 1
        x_0 = np.reshape(x_0, (self.K, 1))
        return x_0
    
    def one_hot_encoding_to_char(self, o_vec):
        return self.ind_to_char[np.argwhere(o_vec == 1)[0][0]]

    def construct_maps(self):
        self.data_string, self.char_set = self.read_text_file(self.file_path)
        self.num_chars = len(self.data_string)
        self.K = len(self.char_set)
        self.char_to_ind : Dict[str, int] = {}
        self.ind_to_char : Dict[int, str] = {}

        for c, i in enumerate(self.char_set):
            self.char_to_ind[i] = c
            self.ind_to_char[c] = i

    def synthesize_text(self, h_0, x_0, n):
        h_prev = h_0
        x_t = x_0
        stored_indices = []
        for i in range(n):
            x_t = np.reshape(x_t, (self.K, 1))
            a_t = np.matmul(self.W, h_prev) + np.matmul(self.U, x_t) + self.b
            h_t = np.tanh(a_t)
            o_t = np.matmul(self.V, h_t) + self.c 
            p_t = self.softmax(o_t)
            next_index = self.generate_next_t(p_t)
            stored_indices.append(next_index)
            x_t = self.generate_one_hot_encoding(next_index)
            h_prev = h_t

        sequence = "".join(list(map(lambda x: self.ind_to_char[x], stored_indices)))
        return sequence

    def generate_next_t(self, p):
        cp = np.cumsum(p)
        a = np.random.rand()
        ixs = np.where(cp - a > 0)[0]
        ii = ixs[0]
        return ii

    def ada_grad(self):
        self.rnn_num_step_size = 1e-4
        smooth_loss = 0.0

        smooth_loss_acc = []

        for _ in range(self.num_epochs):
            print("New epoch")
            h_prev = np.zeros((self.m, 1))
            i = 0
            for j in range(0, self.num_chars - self.seq_length, self.seq_length):
                X, Y = self.construct_training_data(start_index=j)
                a, h, p = self.forward_pass(X=X, h_prev=h_prev)

                d_L_d_V, d_L_d_W, d_L_d_U, d_L_d_b, d_L_d_c = self.compute_grads(X=X, Y=Y, a=a, h=h, p=p)

                self.V_m = self.gamma * self.V_m + (1 - self.gamma) * (d_L_d_V**2)
                self.W_m = self.gamma * self.W_m + (1 - self.gamma) * (d_L_d_W**2)
                self.U_m = self.gamma * self.U_m + (1 - self.gamma) * (d_L_d_U**2)
                self.b_m = self.gamma * self.b_m + (1 - self.gamma) * (d_L_d_b**2)
                self.c_m = self.gamma * self.c_m + (1 - self.gamma) * (d_L_d_c**2)

                self.V = self.V - (self.eta / (np.sqrt(self.V_m + self.epsilon)) * d_L_d_V)
                self.W = self.W - (self.eta / (np.sqrt(self.W_m + self.epsilon)) * d_L_d_W)
                self.U = self.U - (self.eta / (np.sqrt(self.U_m + self.epsilon)) * d_L_d_U)
                self.b = self.b - (self.eta / (np.sqrt(self.b_m + self.epsilon)) * d_L_d_b)
                self.c = self.c - (self.eta / (np.sqrt(self.c_m + self.epsilon)) * d_L_d_c)
                
                loss = self.compute_loss(X=X, Y=Y, hprev = h_prev)
                h_prev = h[:,-1].reshape((self.m, 1))
                if not self.has_computed_first_loss:
                    smooth_loss = loss
                    self.has_computed_first_loss = True
                else: 
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                smooth_loss_acc.append(smooth_loss)
                if i % 1000 == 0:
                    print("i:", i, "smooth_loss:", smooth_loss, "loss:", loss)
                if i % 10000 == 0:
                    x_0 = X[:, 0]
                    sequence = self.synthesize_text(h_0 = h_prev, x_0=x_0, n=200)
                    print(sequence)
                    print("\n")
                i += 1
        self.plot_results(smooth_loss_acc)
        sequence = self.synthesize_text(h_0 = h_prev, x_0=x_0, n=1000)
        print(sequence)
        print("\n")
        
    def plot_results(self, smooth_loss_acc):
        plt.plot(smooth_loss_acc)
        plt.xlabel('Iteration')
        plt.ylabel('Smooth loss')
        plt.show()
        plt.figure()
    
    def compute_grads(self, X, Y, a, h, p):
        num_cols = X.shape[1]
        d_L_d_V = np.zeros((self.K, self.m))
        for t in range(num_cols): 
            h_t_T = h[:, t].reshape(1, -1)
            d_L_d_V += np.matmul(-(Y[:, t].reshape(-1, 1) - p[:, t].reshape(-1, 1)), h_t_T)
        
        d_L_d_c = np.zeros(self.c.shape)
        for t in range(num_cols):
            d_L_d_c += -(Y[:, t].reshape(-1, 1) - p[:, t].reshape(-1, 1))

        d_L_a = self.compute_h_t(X_arr=X, Y_arr=Y, p=p, a=a)

        d_L_d_b = np.sum(d_L_a, axis=1, keepdims=True)

        d_L_d_W = np.zeros(np.shape(self.W))
        h_inserted = np.insert(h, 0, np.zeros(self.m), axis=1)
        for t in range(num_cols):
            d_L_d_W += d_L_a[:, t].reshape(-1, 1) @ h_inserted[:, t].reshape(1, -1)

        d_L_d_U = np.zeros(np.shape(self.U))
        for t in range(num_cols):
            d_L_d_U += d_L_a[:, t].reshape(-1, 1) @ X[:, t].reshape(1, -1)

        gradients = [d_L_d_V, d_L_d_W, d_L_d_U, d_L_d_b, d_L_d_c]
        for i in range(len(gradients)):
            gradients[i] = np.clip(gradients[i], -5, 5)
        return gradients

    def compute_h_t(self, X_arr, Y_arr, p, a):
        num_cols = X_arr.shape[1]
        d_L_d_o = -(Y_arr - p)

        d_L_d_h = np.hstack((np.zeros((self.m, num_cols-1)), 
                             (d_L_d_o[:, num_cols-1] @ self.V).reshape(-1, 1)))
        d_L_d_a = np.hstack((np.zeros((self.m, num_cols-1)), 
                             (d_L_d_h[:, num_cols-1] @ np.diag(1-np.square(np.tanh(a[:, num_cols-1])))).reshape(-1, 1) ))

        for t in range(num_cols-2, -1, -1):
            d_L_d_h[:, t] = d_L_d_o[:, t] @ self.V + d_L_d_a[:, t+1] @  self.W
            d_L_d_a[:, t] = d_L_d_h[:, t] @ np.diag(1-np.square(np.tanh(a[:, t])))
        return d_L_d_a
    
    def compute_loss(self, X, Y, hprev):
        P = self.forward_pass(X=X, h_prev=hprev)[2]
        return -np.sum(Y * np.log(P))

    def forward_pass(self, X, h_prev):
        num_cols = X.shape[1]
        a = np.zeros((self.m, num_cols))
        h = np.zeros((self.m, num_cols))
        p = np.zeros((self.K, num_cols))
        h_t = h_prev
        for i in range(num_cols):
            a_t = self.W @ h_t + self.U @ X[:, i].reshape(-1, 1) + self.b
            a[:, i] = a_t.reshape(self.m)
            h_t = np.tanh(a_t)
            h[:, i] = h_t.reshape(self.m)
            p_t = self.softmax(self.V @ h_t + self.c)
            p[:, i] = p_t.reshape(self.K)
        return a, h, p

    def construct_training_data(self, start_index=0):
        X_chars = self.data_string[start_index : start_index+self.seq_length]
        Y_chars = self.data_string[start_index+1 : start_index+self.seq_length+1]
        
        X_arr = np.hstack(list(map(lambda x: self.generate_one_hot_encoding(self.char_to_ind[x]), X_chars)))
        Y_arr = np.hstack(list(map(lambda x: self.generate_one_hot_encoding(self.char_to_ind[x]), Y_chars)))
        return X_arr, Y_arr

    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def clip_gradients(self, gradients):
        gradient_list = []
        for gradient in gradients:
            gradient_list.append(np.maximum(np.minimum(gradient, 5), -5))
        return gradient_list
    
    """
    NOTE: This code is a rewrite of code written by Elias Stihl. Since the code for calculating numerical gradients
    in Matlab was given, I assume it is fine to use someone else's python translation of said code.
    """
    def compute_gradients_numerically(self, X,Y,step_size,name, h_prev):
        grad = np.zeros_like(self.parameters[name])
        for i in range(self.parameters[name].shape[0]):
            for j in range(self.parameters[name].shape[1]):
                rnn_try = copy.deepcopy(self)
                assert np.array_equal(rnn_try.W, self.W) == True
                rnn_try.parameters[name][i][j] = self.parameters[name][i][j] - step_size
                
                l1,_,_,_,_ = rnn_try.forward_pass(X, Y, h_prev)
                rnn_try.parameters[name][i][j] = self.parameters[name][i][j] + step_size
                l2,_,_,_,_ = rnn_try.forward_pass(X, Y, h_prev)
                grad[i][j] = (l2-l1) / (2 * step_size)
        return grad
    
    def ComputeGradientDifference(self, g_a, g_n, eps = 1e-4):
        return np.sum(np.absolute(g_a - g_n)) / np.sum(np.maximum(eps, np.absolute(g_a) + np.absolute(g_n)))

if __name__ == "__main__":
    rnn = RNN()
    rnn.ada_grad()