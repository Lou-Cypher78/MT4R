import numpy as np


def oc_LQR_disc(x_0, As, Bs, Rs, Qs, S, N:int):
    assert len(As) == N, len(Bs) == N
    assert len(Rs) == N, len(Qs) == N
    Fs, P_next = [], S
    for k in range(N-1, -1, -1):
        tmp = np.linalg.inv(Rs[k] + Bs[k].T @ P_next @ Bs[k])
        Fs.append(tmp @ Bs[k].T @ P_next @ As[k])
        P_next = As[k].T @ P_next @ (As[k] + Bs[k] @ Fs[-1]) + Qs[k]
    Fs, x_k, u_opt = Fs[::-1], x_0, []
    for k in range(N):
        u_opt[k] = Fs[k] @ x_k
        x_k = As[k] @ x_k + Bs[k] @ u_opt[k]
    return u_opt



def oc_LQR_track_disc(x_0, xds, A, B, Rs, Qs, S, N:int, lambda_=0.5):
    assert len(Rs) == N and len(Qs) == N
    assert len(xds) == N
    Ae_s, Be_s, Qe_s = [], [], []
    for k in range(N-1, -1, -1):
        Ad_k = lambda_ * np.eye(x_0.shape[0]) 
        Ad_k += ((xds[k+1] - lambda_ * xds[k])[:, None]  @ xds[k+1][None, :]) / np.sum(xds[k]**2)
        Ae_s.append(np.diag([A, Ad_k]))
        Be_s.append(np.column_stack([B, np.zeros_like(B)]))
        Qe_s.append(np.diag([Qs[k], np.zeros_like(Qs[k])]))
    Se = np.diag([S, np.zeros_like(S)])
    xe_0 = np.row_stack([x_0, xds[0]])
    u_opt = oc_LQR_disc(xe_0, Ae_s, Be_s, Rs, Qe_s, Se, N)
    return u_opt


