import numpy as np
import scipy

def filter_KF_disc(x_0, us, zs, A, B, C, Qs, Rs, P_0, N_K:int):
    assert len(us) == N_K, len(zs) == N_K
    assert len(Rs) == N_K, len(Qs) == N_K
    Ks, xs, Pk = [], [x_0], P_0
    for k in range(N_K):
        xk_ = A @ xs[k] + B @ us[k]
        Pk_ = A @ Pk @ A.T + Qs[k]
        tmp = np.linalg.inv(Rs[k] + C @ Pk_ @ C.T)
        Ks.append(Pk_ @ C.T @ tmp)
        xs.append(xk_ + Ks[k] @ (zs[k] - C @ xk_))
        tmp2 = np.eye(x_0.shape[0]) - Ks[k] @ C
        Pk = Ks[k] @ Rs[k] @ Ks[k].T + tmp2 @ Pk_ @ tmp2.T
    return xs


def filter_EKF(x_0, us, zs, f_func, df_func, h_func, dh_func, Qs, Rs, P_0, N_K:int):
    assert len(us) == N_K, len(zs) == N_K
    assert len(Rs) == N_K, len(Qs) == N_K
    Ks, xs, Pk = [], [x_0], P_0
    for k in range(N_K):
        xk_ = f_func(xs[k], us[k])
        Fk = df_func(xs[k], us[k])
        Pk_ = Fk @ Pk @ Fk.T + Qs[k]
        Hk = dh_func(xs[k])
        tmp = np.linalg.inv(Rs[k] + Hk @ Pk_ @ Hk.T)
        Ks.append(Pk_ @ Hk.T @ tmp)
        xs.append(xk_ + Ks[k] @ (zs[k] - h_func(xs[k])))
        tmp2 = np.eye(x_0.shape[0]) - Ks[k] @ Hk
        Pk = Ks[k] @ Rs[k] @ Ks[k].T + tmp2 @ Pk_ @ tmp2.T
    return xs


def filter_ESKF(x_0, us, zs, f_func, df_func, h_func, dh_func, add_func, Qs, Rs, P_0, N_K:int):
    assert len(us) == N_K, len(zs) == N_K
    assert len(Rs) == N_K, len(Qs) == N_K
    Ks, xs, Pk = [], [x_0], P_0
    for k in range(N_K):
        xk_ = f_func(xs[k], us[k])
        Fk = df_func(xs[k], us[k])
        Pk_ = Fk @ Pk @ Fk.T + Qs[k]
        Hk = dh_func(xs[k])
        tmp = np.linalg.inv(Rs[k] + Hk @ Pk_ @ Hk.T)
        Ks.append(Pk_ @ Hk.T @ tmp)
        delta_x = Ks[k] @ (zs[k] - h_func(xs[k]))
        xs.append(add_func(xk_, delta_x))
        tmp2 = np.eye(delta_x.shape[0]) - Ks[k] @ Hk
        Pk = Ks[k] @ Rs[k] @ Ks[k].T + tmp2 @ Pk_ @ tmp2.T
    return xs


def filter_UKF(x_0, us, zs, f_func, h_func, Qs, Rs, P_0, alpha, kappa, beta, N_K:int):
    assert len(us) == N_K, len(zs) == N_K
    assert len(Rs) == N_K, len(Qs) == N_K
    n, m = x_0.shape[0], zs[0].shape[0]
    lambda_ = alpha**2 *(n + kappa) - n
    wms, wcs = np.zeros([n+1]), np.zeros([n+1])
    wms[0] = lambda_ / (n + lambda_)
    wcs[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    wms[1:], wcs[1:] = 0.5 / (n + lambda_), 0.5 / (n + lambda_)
    Ks, xs, Pk = [], [x_0], P_0
    for k in range(N_K):
        points, fs, hs = np.zeros([2*n+1, n]), np.zeros([2*n+1, n]), np.zeros([2*n+1, m])
        ps = scipy.linalg.sqrtm((n + lambda_) * Pk)
        points[0] = xs[k]
        points[1:1+n], points[1+n:] = ps + xs[k], - ps + xs[k]
        xk_, zk_est = np.zeros_like(x_0), np.zeros_like(zs[0])
        for i in range(2*n+1):
            fs[i] = f_func(points[i], us[k])
            hs[i] = h_func(points[i])
            if i <= n:
                xk_ += wms[i] * fs[i]
                zk_est += wms[i] * hs[i]
            else:
                xk_ += wms[i-n] * fs[i]
                zk_est += wms[i-n] * hs[i]
        Pk_, Hk = Qs[k], np.zeros([m, n])
        for i in range(2*n+1):
            Pi = (fs[i] - xk_)[:, None] @ (fs[i] - xk_)[None, :] 
            Hi = (hs[i] - zk_est)[:, None] @ (fs[i] - xk_)[None, :] 
            if i <= n:
                Pk_ += wcs[i] * Pi
                Hk += wcs[i] * Hi
            else:
                Pk_ += wcs[i-n] * Pi
                Hk += wcs[i-n] * Hi
        tmp = np.linalg.inv(Rs[k] + Hk @ Pk_ @ Hk.T)
        Ks.append(Pk_ @ Hk.T @ tmp)
        xs.append(xk_ + Ks[k] @ (zs[k] - h_func(xs[k])))
        tmp2 = np.eye(x_0.shape[0]) - Ks[k] @ Hk
        Pk = Ks[k] @ Rs[k] @ Ks[k].T + tmp2 @ Pk_ @ tmp2.T
    return xs

