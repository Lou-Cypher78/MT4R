import numpy as np
import scipy

def vo_ICP(N, p1s, p2s):
    assert p1s.shape == (N, 3) and p2s.shape == (N, 3)
    p1_mean, p2_mean = np.mean(p1s, axis=0), np.mean(p2s, axis=0)
    W = (p2s - p2_mean).T @ (p1s - p1_mean)
    U, sigma, VT = np.linalg.svd(W)
    R_2_to_1 = VT.T @ U.T
    if np.linalg.det(R_2_to_1) < 0:
        VT[2, :] = - VT[2, :]
        R_2_to_1 = - VT.T @ U.T
    t_12 = p1_mean - R_2_to_1 @ p2_mean
    T_2_to_1 = np.row_stack([
        np.column_stack([R_2_to_1, t_12]),
        np.array([0, 0, 0, 1])
    ])
    return T_2_to_1


def norm_R(R):
    U, sigma, VT = np.linalg.svd(R)
    R = U @ VT
    if np.linalg.det(R) < 0:
        VT[2, :] = - VT[2, :]
        R = U @ VT
    assert abs(np.linalg.det(R) - 1) < 1e-7
    assert np.linalg.norm(R.T @ R - np.eye(3)) < 1e-5
    return R

def vo_DLT(N, p1s, nu_2s, K):
    assert nu_2s.shape == (N, 3) and p1s.shape == (N, 3)
    assert K.shape == (3, 3)
    M = np.zeros([2*N, 12])
    p1s_ext = np.column_stack([p1s, np.ones(N)])
    for i in range(N):
        A = np.array([[1, 0, -nu_2s[i, 0]], [0, 1, -nu_2s[i, 1]]]) @ K
        M[2*i:2*i+2, :4] = A[:, 0:1] @ p1s_ext[i:i+1, :]
        M[2*i:2*i+2, 4:8] = A[:, 1:2] @ p1s_ext[i:i+1, :]
        M[2*i:2*i+2, 8:] = A[:, 2:3] @ p1s_ext[i:i+1, :]
    t_long = scipy.linalg.null_space(M.T @ M)[:, 0]
    Rt_1_to_2 = np.column_stack([
        t_long[:4], t_long[4:8], t_long[8:]
    ]).T
    R_1_to_2, t_21 = Rt_1_to_2[:, :3], Rt_1_to_2[:, 3]
    Rdet = np.linalg.det(R_1_to_2)
    if Rdet < 0:
        R_1_to_2, t_21 = - R_1_to_2, - t_21
    R_1_to_2 = norm_R(R_1_to_2)
    T_2_to_1 = np.row_stack([
        np.column_stack([R_1_to_2.T, - R_1_to_2.T @ t_21]),
        np.array([0, 0, 0, 1])
    ])
    return T_2_to_1

