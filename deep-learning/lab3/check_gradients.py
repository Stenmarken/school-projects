import numpy as np
from N_dimensional_network_remade import N_dimensional_network
from lab3.lab3 import read_parameters, read_data
from compute_grads_num import compute_gradients_num
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def ComputeGradientDifference(g_a, g_n, eps = 1e-12):
    return np.sum(np.absolute(g_a - g_n)) / np.sum(np.maximum(eps, np.absolute(g_a) + np.absolute(g_n)))

np.random.seed(0)

llambda = 0

parameters = read_parameters("configs/check_gradients.yaml")
train_data, val_data, test_data, data_batches = read_data(parameters)
network = N_dimensional_network(parameters=parameters, 
                                          data_batches=data_batches,
                                          train_data=train_data, 
                                          val_data=val_data,
                                          test_data=test_data)

W_i_differences = [[] for _ in range(len(network.Ws))]
b_i_differences = [[] for _ in range(len(network.bs))] 

for k in range(1, 2, 1):
    first_image_X = network.X_train[(k-1)*20:20*k, 0:30]
    first_image_Y = network.Y_train[:, 0:30]
    h = 1e-5
    d_W_an, d_b_an, dLdGamma_an, dLdBeta_an = network.compute_gradients(X=first_image_X, Y=first_image_Y)

    d_W_num, d_b_num, dLdGamma_num, dLdBeta_num = compute_gradients_num(
        first_image_X, first_image_Y, network.Ws, network.bs, network.gammas, network.betas, network.llambda, 1e-5, network.num_layers)
    assert len(d_W_num) == len(d_W_an), f"len(d_W_num: {len(d_W_num)} len(d_W_an): {len(d_W_an)}"
    for i in range(len(d_W_an)):
        W_diff = ComputeGradientDifference(d_W_an[i], d_W_num[i])
        b_diff = ComputeGradientDifference(d_b_an[i], d_b_num[i])
        if i != len(d_W_an) - 1:
            gamma_diff = ComputeGradientDifference(dLdGamma_an[i], dLdGamma_num[i])
            beta_diff = ComputeGradientDifference(dLdBeta_an[i], dLdBeta_num[i])
        else:
            gamma_diff = 0.0
            beta_diff = 0.0

        W_i_differences[i].append(W_diff)
        b_i_differences[i].append(b_diff)
        print(f"Layer {i+1}")
        print(f"W_diff: {W_diff}\nb_diff: {b_diff}\ngamma_diff: {gamma_diff}\nbeta_diff: {beta_diff}\n\n")
        assert W_diff < 1e-4
        assert b_diff < 1e-4
        assert gamma_diff < 1e-4
        assert beta_diff < 1e-4

for i in range(len(W_i_differences)):
    plt.plot(W_i_differences[i], label=f'W_{i+1}')
    plt.plot(b_i_differences[i], label=f'b_{i+1}')

plt.axhline(y=0.00001, color='r', linestyle='--', label='Cutoff')
plt.ylim(0, 1.5e-5)
plt.title('Plot showing differences between analytic and numeric computations \nfor W_i and b_i and the cutoff line 1e-5')
plt.xlabel('Index')
plt.ylabel('Value')

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*1e4:.1f}e-4'))

plt.legend()
plt.show()
