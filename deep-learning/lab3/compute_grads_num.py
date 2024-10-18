import numpy as np

def compute_b_gamma_beta(i, arr, X, Y, Ws, bs, gammas, betas, llambda, num_layers, h):
        for j in range(arr.shape[1]):
            arr_copy = arr[i, j]
            arr[i] -= h 
            c1 = compute_cost(X, Y, Ws, bs, gammas, betas, llambda, num_layers)
            arr[i] = arr_copy + h
            c2 = compute_cost(X, Y, Ws, bs, gammas, betas, llambda, num_layers)
            return (c2-c1) / (2*h), arr_copy

def compute_gradients_num(X, Y, Ws, bs, gammas, betas, llambda, h, num_layers):
    grad_Ws = [0] * len(Ws)
    grad_bs = [0] * len(bs)
    grad_gammas = [0] * len(gammas)
    grad_betas = [0] * len(betas)

    for k, W in enumerate(Ws):
        dW = np.zeros(W.shape)
        for j in range(len(W)):
            for i in range(len(W[0])):
                W_copy = W[j, i]
                W[j, i] -= h 
                c1 = compute_cost(X, Y, Ws, bs, gammas, betas, llambda, num_layers)
                W[j, i] +=  2*h
                c2 = compute_cost(X, Y, Ws, bs, gammas, betas, llambda, num_layers)
                dW[j, i] = (c2-c1) / (2*h)
                W[j, i] = W_copy
        grad_Ws[k] = dW
    
    for k, b in enumerate(bs):
        db = np.zeros(b.shape)
        for i in range(len(b)):
            db[i], b[i] = compute_b_gamma_beta(i, b, X, Y, Ws, bs, gammas, betas, llambda, num_layers, h)
        grad_bs[k] = db

    for k, gamma in enumerate(gammas):
        dGamma = np.zeros(gamma.shape)
        for i in range(len(gamma)):
            dGamma[i], gamma[i] = compute_b_gamma_beta(i, gamma, X, Y, Ws, bs, gammas, betas, llambda, num_layers, h)

        grad_gammas[k] = dGamma

    for k, beta in enumerate(betas):
        dBeta = np.zeros(beta.shape)
        for i in range(len(beta)):
            dBeta[i], beta[i] = compute_b_gamma_beta(i, beta, X, Y, Ws, bs, gammas, betas, llambda, num_layers, h)

        grad_betas[k] = dBeta

    return grad_Ws, grad_bs, grad_gammas, grad_betas

def compute_cost(X, Y, Ws, bs, gammas, betas, llambda, num_layers):
    p, _, _, _, _, _ = forward_pass(x=X, Ws=Ws, bs=bs, gammas=gammas, betas=betas, num_layers=num_layers)
    loss = -np.sum(Y * np.log(p)) / np.shape(Y)[1]
    cost = loss
    for W in Ws:
        cost += llambda * np.sum(W**2)
    return cost

def forward_pass(x, Ws, bs, gammas, betas, num_layers, precomp_mu = [], precomp_var = []):
    if bool(precomp_mu) != bool(precomp_var):
        raise "precomp_mu and/or precomp_var are incorrectly set"
    xs = [0]* (num_layers - 1)
    ss = [0]*num_layers
    ss_hat = [0] * (num_layers - 1)
    mus = [0] * (num_layers - 1)
    variances = [0] * (num_layers - 1)
    k = num_layers - 1

    for i in range(k):
        s_i = Ws[i] @ x + bs[i]
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

        s_i_tilde = np.multiply(gammas[i], s_i_hat) + betas[i]
        x = np.maximum(s_i_tilde, 0)
        xs[i] = x
    s_k = Ws[-1] @ x + bs[-1]
    ss[-1] = s_k
    p = softmax(s_k)
    return p, ss, xs, ss_hat, mus, variances

def softmax(s):
    exp_s = np.exp(s - np.max(s, axis=0))
    return exp_s / np.sum(exp_s, axis=0)