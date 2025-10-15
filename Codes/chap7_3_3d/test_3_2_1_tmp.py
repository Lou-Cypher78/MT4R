import numpy as np
from scipy.spatial.transform import Rotation as R


def hat(v):
    """李代数 so(3) 的帽子运算符"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def rodrigues(phi):
    """将李代数向量 phi 转换为旋转矩阵"""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3)
    axis = phi / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def norm_R(R):
    U, sigma, VT = np.linalg.svd(R)
    R = U @ VT
    if np.linalg.det(R) < 0:
        VT[2, :] = - VT[2, :]
        R = U @ VT
    assert abs(np.linalg.det(R) - 1) < 1e-7
    assert np.linalg.norm(R.T @ R - np.eye(3)) < 1e-5
    return R

# 1. 生成真实旋转 R_true 和平移 t
np.random.seed(42)
angle = np.deg2rad(30)
axis = np.array([0.0, 0.0, 1.0])
R_true = R.from_rotvec(angle * axis).as_matrix()
t = np.array([1.0, 0.5, -0.3])

# 2. 生成 N 个随机点 p1
N = 10
p1 = np.random.randn(N, 3)
p2_true = (R_true.T @ (p1 - t).T).T  # 每一行是一个点

# 4. 初始化估计的旋转 R_est
phi_est = np.deg2rad([1.0, 1, -0.3])  # 初始李代数扰动
R_est = rodrigues(phi_est)

lr = 1e-2
max_iters = 100
for iter in range(max_iters):
    p2_est = (R_est.T @ (p1 - t).T).T
    e = (p2_est - p2_true).reshape(-1)
    J = np.zeros((3 * N, 3))
    for i in range(N):
        J[3*i:3*(i+1), :] = hat(p2_est[i])  
    # delta_phi = - lr * J.T @ e   # Gradient Decent
    delta_phi = - np.linalg.inv(J.T @ J) @ J.T @ e # Gauss Newton
    R_update = rodrigues(delta_phi)
    R_est = norm_R(R_est @ R_update)
    cost = np.dot(e, e)
    print(f"Iter {iter}: cost = {cost:.6f}, |delta_phi| = {np.linalg.norm(delta_phi):.6f}")
    if np.linalg.norm(delta_phi) < 1e-6:
        break
