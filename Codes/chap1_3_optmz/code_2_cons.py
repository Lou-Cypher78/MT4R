import numpy as np
from code_1_nocons import GR_line_search, opt_nc_LSGD

def opt_eqc_gd_line(df_func, ddf_func, g_func, dg_func, ddg_func, x_0, epsilon=1e-4):
    lambda_0 = np.ones_like(g_func(x_0))
    x_ext_k = np.concatenate([x_0, lambda_0])
    n_x, n_lam = x_0.shape[0], lambda_0.shape[0]
    def J_func(x_ext):
        x, lambda_ = x_ext[:n_x], x_ext[n_x:]
        gradL = np.concatenate([
            (df_func(x)[None, :] + lambda_[None, :] @ dg_func(x))[0],
            g_func(x)
        ])
        return 0.5 * (gradL[None, :] @ gradL[:, None])[0,0]
    while True:
        x_k, lambda_k = x_ext_k[:n_x], x_ext_k[n_x:]
        gradL = np.concatenate([
            (df_func(x_k)[None, :] + lambda_k[None, :] @ dg_func(x_k))[0],
            g_func(x_k)
        ]) # (m+n,)
        tmp = ddf_func(x_k) + np.einsum("i,ijk->jk", lambda_k, ddg_func(x_k)) # (n, n)
        hessL = np.block([
            [tmp, dg_func(x_k).T],
            [dg_func(x_k), np.zeros((n_lam, n_lam))]
        ]) 
        dotJ = hessL @ gradL
        if np.linalg.norm(dotJ) < epsilon:
            break
        t = - dotJ
        h = GR_line_search(J_func, x_ext_k, t)
        x_ext_k = x_ext_k + h * t
    return x_ext_k[:n_x]



def opt_neqc_LSGD(f_func, df_func, h_func, dh_func, x_0, lambda_0=100.0, gamma=0.8, epsilon=1e-4):
    x_k, lambda_b = x_0, lambda_0
    while True:
        def fb_func(x):
            return f_func(x) - lambda_b * np.sum(np.log(-h_func(x)))
        def dfb_func(x):
            t = (1/h_func(x))[None, :] @ dh_func(x)
            return (df_func(x) - lambda_b * t)[0]
        x_new = opt_nc_LSGD(fb_func, dfb_func, x_k)
        print(x_k, lambda_b, x_new)
        x_k, lambda_b = x_new, lambda_b * gamma
        if lambda_b < epsilon:
            break
    return x_k