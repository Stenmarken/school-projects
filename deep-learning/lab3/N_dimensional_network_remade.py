import numpy as np
import numpy
import matplotlib.pyplot as plt

class N_dimensional_network():
    """
    W_1: (m x d), (50, 3072)
    W_2: (K x m), (10, 50)
    b_1: (m x 1), (50, 1)
    b_2: (K x 1), (10, 1)

    """
    def __init__(self, parameters, data_batches, train_data, val_data, test_data, sig=None) -> None:
        self.set_parameters(parameters)
        self.sig = sig
        self.initialize_network()
        self.X, self.Y = data_batches
        assert self.X.shape[0] == self.Y.shape[0]
        self.num_batches = self.X.shape[0]
        self.X_train, self.Y_train, self.y_train = train_data
        self.X_val, self.Y_val, self.y_val = val_data
        self.X_test, self.Y_test, self.y_test = test_data
        #self.get_network_info()

    def get_network_info(self):
        for i, W in enumerate(self.Ws):
            print(f"i: {i}, W: {W.shape}")
        
        for i, gamma in enumerate(self.gammas):
            print(f"i: {i}, gamma: {gamma.shape}")
            
    def create_matrices_normal_distribution(self, sig):
        d = 3072
        if self.debug_mode:
            d = 20
        prev_nodes = -1
        Ws = []
        bs = []
        gammas = []
        betas = []
        for i in range(self.num_layers):
            if i == 0:

                Ws.append(np.random.normal(0, sig, (self.nodes[0], d)))
                bs.append(np.ones((self.nodes[0], 1)))
                
                prev_nodes = self.nodes[0]
                # Initialization rationale taken from https://stackoverflow.com/questions/62216100/initialization-of-gamma-and-beta-in-batch-normalization-in-neural-networks
                gammas.append(np.ones((self.nodes[i], 1)))
                betas.append(np.zeros((self.nodes[i], 1)))
            elif i == self.num_layers - 1:
                Ws.append(np.random.normal(0, sig, (self.K, prev_nodes)))
                bs.append(np.zeros((self.K, 1)))
                
            else:
                Ws.append(np.random.normal(0, sig, (self.nodes[i], prev_nodes)))
                bs.append(np.zeros((self.nodes[i], 1)))
                
                # Initialization rationale taken from https://stackoverflow.com/questions/62216100/initialization-of-gamma-and-beta-in-batch-normalization-in-neural-networks
                gammas.append(np.ones((self.nodes[i], 1)))
                betas.append(np.zeros((self.nodes[i], 1)))
                prev_nodes = self.nodes[i]
        assert len(Ws) == len(bs) and len(gammas) == len(betas)
        return Ws, bs, gammas, betas

    def create_matrices(self):
        d = 3072
        if self.debug_mode:
            d = 20
        prev_nodes = -1
        Ws = []
        bs = []
        gammas = []
        betas = []
        for i in range(self.num_layers):
            if i == 0:
                Ws.append(self.initialize_matrix((self.nodes[0], d)))
                bs.append(np.ones((self.nodes[0], 1)))
                
                prev_nodes = self.nodes[0]
                # Initialization rationale taken from https://stackoverflow.com/questions/62216100/initialization-of-gamma-and-beta-in-batch-normalization-in-neural-networks
                gammas.append(np.ones((self.nodes[i], 1)))
                betas.append(np.zeros((self.nodes[i], 1)))
            elif i == self.num_layers - 1:
                Ws.append(self.initialize_matrix((self.K, prev_nodes)))
                bs.append(np.zeros((self.K, 1)))
                
            else:
                Ws.append(self.initialize_matrix((self.nodes[i], prev_nodes)))
                bs.append(np.zeros((self.nodes[i], 1)))
                
                # Initialization rationale taken from https://stackoverflow.com/questions/62216100/initialization-of-gamma-and-beta-in-batch-normalization-in-neural-networks
                gammas.append(np.ones((self.nodes[i], 1)))
                betas.append(np.zeros((self.nodes[i], 1)))
                prev_nodes = self.nodes[i]
        assert len(Ws) == len(bs) and len(gammas) == len(betas)
        return Ws, bs, gammas, betas

    def initialize_network(self):
        self.Ws = []
        if self.use_normal_distribution:
            self.Ws, self.bs, self.gammas, self.betas = self.create_matrices_normal_distribution(sig=self.sig)
        else:
            self.Ws, self.bs, self.gammas, self.betas = self.create_matrices()
        self.train_costs = []
        self.train_losses = []
        self.validation_costs = []
        self.validation_losses = []
        self.train_accuracy = []
        self.validation_accuracy = []
        self.has_initialized_mu_and_variance = False

    def set_parameters(self, parameters):
        self.m = parameters['m']
        self.K = parameters['K']
        self.d = parameters['d']
        self.llambda = parameters['llambda']
        self.n_s = parameters['n_s']
        self.printings_per_cycle = parameters['printings_per_cycle']
        self.num_cycles = parameters['num_cycles']
        self.eta_min = parameters['eta_min']
        self.eta_max = parameters['eta_max']
        self.batch_size = parameters['batch_size']
        self.debug_mode = parameters['debug_mode']
        self.num_layers = parameters['num_layers']
        self.initialization_technique = parameters['initialization_technique']
        self.nodes = parameters['nodes']
        self.alpha = parameters['alpha']
        self.use_normal_distribution = parameters['use_normal_distribution']

    def exponential_moving_average(self, mu, variances):
        assert len(mu) == len(variances)
        for i in range(len(mu)):
            if not self.has_initialized_mu_and_variance:
                self.mu_avg = mu
                self.variances_avg = variances
                self.has_initialized_mu_and_variance = True
            else:
                self.mu_avg[i] = self.alpha * self.mu_avg[i] + (1-self.alpha) * mu[i]
                self.variances_avg[i] = self.alpha * self.variances_avg[i] + (1-self.alpha) * variances[i]

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def compute_cost(self, X, Y):
        if self.has_initialized_mu_and_variance:
            p, _, _, _, _, _ = self.forward_pass(x=X, precomp_mu=self.mu_avg, precomp_var=self.variances_avg)
        else:
            p, _, _, _, _, _ = self.forward_pass(x=X)

        loss = -np.sum(Y * np.log(p)) / np.shape(Y)[1]
        cost = loss
        for W in self.Ws:
            cost += self.llambda * np.sum(W**2)
        return cost, loss
    
    def compute_accuracy(self, X, y):
        if self.has_initialized_mu_and_variance:
            p, _, _, _, _, _ = self.forward_pass(X, precomp_mu=self.mu_avg, precomp_var=self.variances_avg)
        else:
            p, _, _, _, _, _ = self.forward_pass(X)
        indices = np.argmax(p, axis=0)
        return np.sum(y == indices) / X.shape[1]

    def compute_gradients(self, X, Y):
        _, n_b = X.shape
        p, ss, xs, s_hats, mus, variances = self.forward_pass(X)
        self.exponential_moving_average(mu=mus, variances=variances)
        g_batch = -(Y - p)
        dLdW = [0] * len(self.Ws)
        dLdb = [0] * len(self.bs)
        dLdGamma = [0] * len(self.gammas)
        dLdBeta = [0] * len(self.betas)
        assert len(self.gammas) == len(self.Ws) - 1
        assert len(self.betas) == len(self.bs) - 1

        for i in range(len(self.Ws)-1, -1, -1):
            if i == len(self.Ws)-1:
                dLdW[i] = (1/n_b)*g_batch @ xs[i-1].T + 2*self.llambda*self.Ws[i]
                dLdb[i] = (1/n_b)*g_batch @ np.ones((n_b, 1))
                g_batch = self.Ws[i].T @ g_batch
                g_batch = np.multiply(g_batch, np.int64(xs[i-1] > 0))
            else:
                dLdGamma[i] = (1/n_b) * np.multiply(g_batch, s_hats[i]) @ np.ones((n_b, 1))
                dLdBeta[i] = (1/n_b) * g_batch @ np.ones((n_b, 1))

                g_batch = np.multiply(g_batch, (self.gammas[i] @ np.ones((1, n_b))) )
                g_batch = self.batch_norm_backward_pass(G_batch=g_batch, s_batch=ss[i], mu=mus[i], variance=variances[i], n_b=n_b)
                if i == 0:
                    dLdW[i] = (1/n_b)*g_batch @ X.T + 2*self.llambda*self.Ws[i]
                else:
                    dLdW[i] = (1/n_b)*g_batch @ xs[i-1].T + 2*self.llambda*self.Ws[i]
                dLdb[i] = (1/n_b)*g_batch @ np.ones((n_b, 1))

                if i > 0:
                    g_batch = self.Ws[i].T @ g_batch
                    g_batch = np.multiply(g_batch, np.int64(xs[i-1] > 0))
        for i in range(len(self.Ws)):
            assert dLdW[i].shape == self.Ws[i].shape, f"Shape mismatch on index {i}: (dLdW = {dLdW[i].shape}) != (self.Ws = {self.Ws[i].shape}) "
            assert dLdb[i].shape == self.bs[i].shape, f"Shape mismatch on index {i}: (dLdb = {dLdb[i].shape}) != (self.bs = {self.bs[i].shape})"
        for i in range(len(self.betas)):
            assert dLdBeta[i].shape == self.betas[i].shape, f"Shape mismatch on index {i}: (dLdBeta = {dLdBeta[i].shape}) != (self.betas = {self.betas[i].shape})"
            assert dLdGamma[i].shape == self.gammas[i].shape, f"Shape mismatch on index {i}: (dLdGamma = {dLdGamma[i].shape}) != (self.gamma = {self.gammas[i].shape})"
        
        return dLdW, dLdb, dLdGamma, dLdBeta
    
    def batch_normalize(self, s, mu, var):
        eps = 2e-52
        return np.diag(np.power(var.reshape(-1,) + eps, -0.5)) @ (s - mu)

    def forward_pass(self, x, precomp_mu = [], precomp_var = []):
        if bool(precomp_mu) != bool(precomp_var):
            raise "precomp_mu and/or precomp_var are incorrectly set"
        xs = [0]* (self.num_layers - 1)
        ss = [0]*self.num_layers
        ss_hat = [0] * (self.num_layers - 1)
        mus = [0] * (self.num_layers - 1)
        variances = [0] * (self.num_layers - 1)
        k = self.num_layers - 1

        for i in range(k):
            s_i = self.Ws[i] @ x + self.bs[i]
            ss[i] = s_i
            if precomp_mu != [] and precomp_var != []:
                mu = precomp_mu[i]
                v_j = precomp_var[i]
            else:
                mu = np.mean(s_i, axis=1, keepdims=True)
                v_j = np.var(s_i, axis=1, keepdims=True, ddof=0)
            mus[i] = mu
            variances[i] = v_j

            s_i_hat = np.divide((s_i - mu), np.sqrt(v_j + 1e-6))
            ss_hat[i] = s_i_hat

            s_i_tilde = np.multiply(self.gammas[i], s_i_hat) + self.betas[i]
            x = np.maximum(s_i_tilde, 0)
            xs[i] = x
        s_k = self.Ws[-1] @ x + self.bs[-1]
        ss[-1] = s_k
        p = self.softmax(s_k)
        return p, ss, xs, ss_hat, mus, variances
    
    def update_matrices(self, dLdW, dLdb, dLdGamma, dLdBeta):
        for i in range(len(self.Ws)):
            self.Ws[i] -= self.eta * dLdW[i]
            self.bs[i] -= self.eta * dLdb[i]
            if i != (len(self.Ws) - 1):
                self.gammas[i] = self.gammas[i] - (self.eta * dLdGamma[i])
                self.betas[i] = self.betas[i] - (self.eta * dLdBeta[i])

    def log_epoch(self, t):
        print("Update step", t)
        train_cost, train_loss = self.compute_cost(self.X_train, self.Y_train)
        self.train_costs.append(train_cost)
        self.train_losses.append(train_loss)
        validation_cost, validation_loss = self.compute_cost(self.X_val, self.Y_val)
        self.validation_costs.append(validation_cost)
        self.validation_losses.append(validation_loss)
        print("Training cost:", train_cost)
        print("Training loss", train_loss)
        print("Validation cost:", validation_cost)
        print("Validation loss", validation_loss)
        training_accuracy = self.compute_accuracy(self.X_train, self.y_train)
        self.train_accuracy.append(training_accuracy)
        validation_accuracy = self.compute_accuracy(self.X_val, self.y_val)
        self.validation_accuracy.append(validation_accuracy)
        print("Training accuracy:", training_accuracy)
        print("Validation accuracy:", validation_accuracy, "\n")
    
    def show_graphs(self, t_list):
        print("Accuracy on the test dataset:", self.compute_accuracy(self.X_test, self.y_test))

        plt.plot(self.learning_rates)
        plt.show()
        plt.figure()

        plt.plot(t_list, self.train_costs, label='Training cost')
        plt.plot(t_list, self.validation_costs, label="Validation cost")
        plt.legend()
        plt.title("Plot of training and validation cost")
        plt.xlabel('Update step')
        plt.ylabel('Cost')
        plt.title(f"Cost \n lambda={self.llambda}, n_s={self.n_s}, batch_size={self.batch_size}, eta_min={self.eta_min}, eta_max={self.eta_max}\nNodes={self.nodes}, init={self.initialization_technique}")
        plt.show()
        plt.figure() 

        plt.plot(t_list, self.train_losses, label='Training loss')
        plt.plot(t_list, self.validation_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Loss \n lambda={self.llambda}, n_s={self.n_s}, batch_size={self.batch_size}, eta_min={self.eta_min}, eta_max={self.eta_max}\n Nodes={self.nodes}, init={self.initialization_technique}")
        plt.xlabel('Update step')
        plt.ylabel('Loss')
        plt.show()
        plt.figure() 

        plt.plot(t_list, self.train_accuracy, label="Training accuracy")
        plt.plot(t_list, self.validation_accuracy, label="Validation accuracy")
        plt.legend()
        plt.title(f"Accuracy \n lambda={self.llambda}, n_s={self.n_s}, batch_size={self.batch_size}, eta_min={self.eta_min}, eta_max={self.eta_max}")
        plt.xlabel('Update step')
        plt.ylabel('Accuracy')
        plt.show()
        plt.figure() 

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

    def mini_batch_gd(self):
        t_list = []
        self.learning_rates = []
        
        t = 0
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

                dLdW, dLdb, dLdGamma, dLdBeta = self.compute_gradients(X=X[i], Y=Y[i])
                #self.eta = 0.001
                self.cyclical_learning_rate(t)          
                assert(self.eta <= self.eta_max and self.eta >= self.eta_min)
                self.learning_rates.append(self.eta)

                self.update_matrices(dLdW, dLdb, dLdGamma, dLdBeta)

                t += 1
        #print("Accuracy on the test dataset:", self.compute_accuracy(self.X_test, self.y_test))
        #print("Accuracy on the validation dataset:", self.compute_accuracy(self.X_val, self.y_val))
        self.show_graphs(t_list)

    def initialize_matrix(self, shape, sqrt_val=0.0):
        """
        sqrt_val is included since sometimes we want 1/np.sqrt(m) and other times
        1/np.sqrt(d)
        """
        if self.initialization_technique == "xavier":
            return xavier_initialization(shape)
        elif self.initialization_technique == "he":
            return he_initialization(shape)
        else:
            if sqrt_val == 0.0:
                raise "sqrt_val is 0.0 but still we use standard initialization which causes division by 0"
            return np.random.normal(0, 1/np.sqrt(sqrt_val), shape)
        
    def batch_norm_backward_pass(self, G_batch, s_batch, mu, variance, n_b):
        eps = 2e-52
        one_column = np.ones((n_b, 1))
        sigma_1 = np.power(variance + eps, -0.5)
        sigma_2 = np.power(variance + eps, -1.5)
        G_1 = np.multiply(G_batch, (sigma_1 @ one_column.T))
        G_2 = np.multiply(G_batch, (sigma_2 @ one_column.T))
        D = s_batch - (mu @ one_column.T)
        c = np.multiply(G_2, D) @ one_column
        return G_1 - (1/n_b)*(G_1 @ one_column) @ one_column.T - (1/n_b)*np.multiply(D, c @ one_column.T)


# Code taken from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def xavier_initialization(shape):
    n_in = shape[1]
    return np.random.normal(loc = 0, scale = (1/np.sqrt(n_in)), size = shape)

def he_initialization(shape):
    n_in = shape[1]
    return np.random.normal(loc = 0, scale = (1/np.sqrt(2/n_in)), size = shape)