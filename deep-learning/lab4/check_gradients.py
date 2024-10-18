import numpy as np
from RNN import RNN 
from ComputeGradsNum import compute_gradients_numerically

def ComputeGradientDifference(g_a, g_n, eps = 1e-4):
    return np.sum(np.absolute(g_a - g_n)) / np.sum(np.maximum(eps, np.absolute(g_a) + np.absolute(g_n)))

def check_gradients(num_sequences = 30):
    threshold = 1e-4

    for i in range(num_sequences):
        print("i", i)
        rnn_num = RNN(seed=0)
        rnn_an = RNN(seed=0)

        assert np.array_equal(rnn_an.W, rnn_num.W) == True
        assert np.array_equal(rnn_an.V, rnn_num.V) == True
        assert np.array_equal(rnn_an.U, rnn_num.U) == True

        X, Y = rnn_num.construct_training_data(start_index=i)
        h = 1e-4
        h_prev = np.zeros((rnn_num.m, 1))

        U_num = compute_gradients_numerically(X=X,Y=Y,step_size=h,name="U", rnn=rnn_num, h_prev=h_prev)
        W_num = compute_gradients_numerically(X=X,Y=Y,step_size=h,name="W", rnn=rnn_num, h_prev=h_prev)
        V_num = compute_gradients_numerically(X=X,Y=Y,step_size=h,name="V", rnn=rnn_num, h_prev=h_prev)
        print("V_num", V_num)
        a, h, p = rnn_an.forward_pass(X=X, h_prev=h_prev)
        d_L_d_V, d_L_d_W, d_L_d_U, d_L_d_b, d_L_d_c = rnn_an.compute_grads(X=X, Y=Y, a=a, h=h, p=p)

        diff = ComputeGradientDifference(d_L_d_V, V_num)
        assert diff < threshold, f"V_diff: {diff}"
        diff = ComputeGradientDifference(d_L_d_W, W_num)
        assert diff < threshold, f"W_diff: {diff}"
        diff = ComputeGradientDifference(d_L_d_U, U_num) 
        assert diff < threshold, f"U_diff: {diff}"

if __name__ == "__main__":
    check_gradients(num_sequences=30)
