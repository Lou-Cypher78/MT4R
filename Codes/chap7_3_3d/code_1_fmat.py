import numpy as np
import scipy
import scipy.linalg

def triangulate_points(N, nu_1s, nu_2s, T_2_to_1, K):
    assert nu_1s.shape == (N, 3) and nu_2s.shape == (N, 3)
    assert T_2_to_1.shape == (4, 4) and K.shape == (3, 3)
    p1s, p2s = np.zeros([N, 3]), np.zeros([N, 3])
    invK = np.linalg.inv(K)
    R_2_to_1, t_12_1 = T_2_to_1[:3, :3], T_2_to_1[:3, 3]
    for i in range(N):
        nu1, nu2 = nu_1s[i], nu_2s[i]
        c = invK @ nu1
        a = np.cross(c, R_2_to_1 @ invK @ nu2)
        b = np.cross(c, t_12_1)
        z2 = - b[None, :] @ a[:, None] / np.sum(a**2)
        d = R_2_to_1 @ (z2 * invK @ nu2) + t_12_1
        z1 = d[None, :] @ c[:, None] / np.sum(c**2)
        p1s[i], p2s[i] = z1 * invK @ nu1, z2 * invK @ nu2
    return p1s, p2s


def E_to_Rt(N, nu_1s, nu_2s, E, K):
    assert nu_1s.shape == (N, 3) and nu_2s.shape == (N, 3)
    assert E.shape == (3, 3) and K.shape == (3, 3)
    U, sigma, VT = np.linalg.svd(E)
    W = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ])
    R1, R2 = U @ W @ VT, U @ W.T @ VT
    if np.linalg.det(R1) < 0:
        R1 = - R1
    if np.linalg.det(R2) < 0:
        R2 = - R2
    t1, t2 = U[:, 2], - U[:, 2]
    candidates = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
    for R, t in candidates:
        T_2_to_1 = np.zeros([4,4])
        T_2_to_1[:3, :3], T_2_to_1[:3, 3], T_2_to_1[3, 3] = R, t, 1
        p1s, p2s = triangulate_points(N, nu_1s, nu_2s, T_2_to_1, K)
        if np.all(p1s[:, 2] > 0) and np.all(p2s[:, 2] > 0):
            return T_2_to_1
    return None



def calc_vo_Fmat_8p(N, nu_1s, nu_2s, K):
    assert nu_1s.shape == (N, 3) and nu_2s.shape == (N, 3)
    assert K.shape == (3, 3)
    A = np.zeros([N, 9])
    for i in range(N):
        u2, v2, nu1 = nu_2s[i, 0], nu_2s[i, 1], nu_1s[i]
        A[i, :] = np.concatenate([u2*nu1, v2*nu1, nu1])
    f = scipy.linalg.null_space(A)[:, 0]
    F_ = np.array([
        [f[0], f[3], f[6]],
        [f[1], f[4], f[7]],
        [f[2], f[5], f[8]],
    ]) / np.max(f)
    U, sigma, VT = np.linalg.svd(F_)
    F = U @ np.diag([sigma[0], sigma[1], 0]) @ VT
    E = K.T @ F @ K
    T_2_to_1 = E_to_Rt(N, nu_1s, nu_2s, E, K)
    return T_2_to_1

