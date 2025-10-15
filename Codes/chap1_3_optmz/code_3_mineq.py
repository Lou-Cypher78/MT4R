import numpy as np

def opt_minls_gd(e_func, J_func, H, x_0, epsilon=1e-4, alpha=1e-2):
    x_k = x_0
    while True:
        dotf = J_func(x_k) @ H @ e_func(x_k)
        if np.linalg.norm(dotf) < epsilon:
            break
        x_k = x_k - alpha * dotf
    return x_k


def Gauss_Newton(e_func, J_func, H, x_0, epsilon=1e-4, alpha=1e-2):
    x_k = x_0
    while True:
        J_k = J_func(x_k)
        H_n = J_k.T @ H @ J_k
        g_n = J_k.T @ H @ e_func(x_k)
        if np.linalg.norm(g_n) < epsilon:
            break
        x_k = x_k - np.linalg.inv(H_n) @ g_n
    return x_k


def Levenberg_Marquardt(e_func, J_func, H, x_0, epsilon=1e-4, rmin=1e-5, rmax=1, lambda_0=1):
    x_k, lambda_k = x_0, lambda_0
    while True:
        J_k = J_func(x_k)
        g_l = J_k.T @ H @ e_func(x_k)
        if np.linalg.norm(g_l) < epsilon:
            break
        H_l = J_k.T @ H @ J_k + lambda_k * np.eye(x_0.shape[0])
        delta_x = - np.linalg.inv(H_l) @ g_l
        x_new = x_k + delta_x
        f_k = e_func(x_k)[None, :] @ H @ e_func(x_k)
        f_new = e_func(x_new)[None, :] @ H @ e_func(x_new)
        rho = np.abs(f_k - f_new) / np.linalg.norm(J_k @ delta_x)
        if rho > rmax and lambda_k < 1e2:
            lambda_k = 2 * lambda_k
        elif rho < rmin and lambda_k > 1e-10:
            lambda_k = 0.5 * lambda_k
        x_k = x_new
    return x_k