import numpy as np
import scipy
from code_2_icpdlt import vo_ICP, vo_DLT


def get_inside_receiving_radius(points):
    assert points.shape == (4, 3)
    def triangle_area(a, b, c):
        # 0.5 * || (b - a) × (c - a) ||
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    def tetrahedron_volume(a, b, c, d):
        # 1/6 * | (b - a) · ((c - a) × (d - a)) |
        return abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0
    A, B, C, D = points
    area_ABC = triangle_area(A, B, C)
    area_ABD = triangle_area(A, B, D)
    area_ACD = triangle_area(A, C, D)
    area_BCD = triangle_area(B, C, D)
    total_area = area_ABC + area_ABD + area_ACD + area_BCD
    volume = tetrahedron_volume(A, B, C, D)
    return (3 * volume) / total_area


def vo_EPnP(N, p1s, nu_2s, K):
    assert p1s.shape == (N, 3) and nu_2s.shape == (N, 3)
    assert K.shape == (3, 3)
    # calculate 2D control points
    f1_s = np.zeros([4, 3])
    f1_s[0] = np.mean(p1s, axis=0)
    P_bar = p1s - f1_s[0][None, :]
    lambdas, vs = np.linalg.eig(P_bar.T @ P_bar)
    for i in (1, 2, 3):
        f1_s[i] = f1_s[0] + vs[i-1] * np.sqrt(lambdas[i-1] / N)
    # calculate projection coefficents
    F_r = np.row_stack([f1_s.T, np.array([1,1,1,1])])
    P1_ext = np.column_stack([p1s, np.ones([N])])
    A = np.linalg.inv(F_r) @ P1_ext.T
    # calculate 3D control points
    M, I = np.zeros([2*N, 12]), np.eye(3)
    for i in range(N):
        Vi = np.array([ [1, 0, -nu_2s[i, 0]], [0, 1, -nu_2s[i, 1]]])
        M[2*i:2*i+2, :] = Vi @ K @ np.column_stack([
            A[0,i]*I, A[1,i]*I, A[2,i]*I, A[3,i]*I
        ])
    f = scipy.linalg.null_space(M.T @ M)[:, 0] # <- scale-free!
    # scale correction
    f2_s = np.row_stack([f[:3], f[3:6], f[6:9], f[9:]])
    r1 = get_inside_receiving_radius(f1_s)
    r2 = get_inside_receiving_radius(f2_s)
    f2_s = f2_s / r2 * r1
    # restore 3D points & output
    p2s = A.T @ f2_s # restore all 3D points
    return vo_ICP(N, p1s, p2s)


def skew(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])



def vo_BA_GN(N, p1s, nu_2s, K, alpha=1e-4, epsilon_1=5e-1, epsilon_2=1e-6):
    assert p1s.shape == (N, 3) and nu_2s.shape == (N, 3)
    assert K.shape == (3, 3)
    T_2_to_1 = np.eye(4)
    # T_2_to_1 = vo_DLT(N, p1s, nu_2s, K)
    k = 0
    while True:
        R_2_to_1, t_12 = T_2_to_1[:3, :3], T_2_to_1[:3, 3]
        J_k, e_k = np.zeros([2*N, 6]), np.zeros([2*N])
        for i in range(N):
            p2 = R_2_to_1.T @ (p1s[i] - t_12)
            m2 = p2 / (p2[2] + epsilon_2)
            nu2_est = K @ m2
            e_k[2*i:2*i+2] = (nu_2s[i] - nu2_est)[:2]
            J1 = np.array([[-1,0,0], [0,-1,0]])
            J2 = K
            J3 = np.array([ [1, 0, -p2[0] / p2[2]],
                [0, 1, -p2[1] / p2[2]],
                [0, 0, 0],
            ]) / (p2[2] + epsilon_2)
            J4 = np.column_stack([skew(R_2_to_1.T @ (p1s[i] - t_12)), -R_2_to_1.T])
            J_k[2*i:2*i+2, :] = J1 @ J2 @ J3 @ J4
        delta_x = - alpha * np.linalg.inv(J_k.T @ J_k) @ J_k.T @ e_k
        print(f"Iter {k}: t = {T_2_to_1[:3, 3]}, e_k: {np.linalg.norm(e_k)/N}")
        if np.linalg.norm(e_k)/N < epsilon_1:
            break
        T_2_to_1[:3, :3] = R_2_to_1 @ (np.eye(3) + skew(delta_x[:3]))
        T_2_to_1[:3, 3] = t_12 + delta_x[3:]
        k += 1
    return T_2_to_1


