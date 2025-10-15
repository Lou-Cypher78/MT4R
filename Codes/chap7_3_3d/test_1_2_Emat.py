import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
from code_1_fmat import E_to_Rt

np.random.seed(42)
K = np.array([
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
])

def skew(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])


def generate_test_data():
    # 随机生成R, t (2 to 1)
    theta = np.deg2rad(-30)  # 30度旋转
    R_true = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    t_true = np.array([0.5, 0.1, 0])  # 基线
    # print("R_true:", R_true)
    # print("t_true:", t_true)
    # 生成E和T
    T_2_to_1 = np.eye(4)
    T_2_to_1[:3, :3] = R_true
    T_2_to_1[:3, 3] = t_true
    E = skew(t_true) @ R_true
    T_1_to_2 = np.linalg.inv(T_2_to_1)
    # 三维点
    N = 10
    P_world = np.random.uniform(-1, 1, (N, 3))
    P_world[:, 2] = np.abs(P_world[:, 2])  # 保证在相机前方
    # 投影到两个相机坐标系
    def project(P, T):
        P_h = np.hstack([P, np.ones((P.shape[0], 1))])  # 齐次坐标
        P_cam = (T @ P_h.T).T[:, :3]
        return (K @ (P_cam.T / P_cam[:, 2])).T  # 转为归一化图像坐标
    nu_1s = project(P_world, np.eye(4))
    nu_2s = project(P_world, T_1_to_2)
    return N, nu_1s, nu_2s, E, R_true, t_true, K



def test_E_to_Rt():
    N, nu_1s, nu_2s, E, R_true, t_true, K = generate_test_data()
    T_est = E_to_Rt(N, nu_1s, nu_2s, E, K)
    assert T_est is not None, "E_to_Rt 返回 None，说明没有找到合适的解"
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    # 验证旋转矩阵是否接近（最多有符号不确定性）
    R_diff = R_est @ R_true.T
    angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    # 验证平移向量方向是否一致（单位向量，忽略尺度）
    t_true /= np.linalg.norm(t_true)
    t_est /= np.linalg.norm(t_est)
    cos_angle = np.dot(t_true, t_est)
    angle_t = np.arccos(np.clip(cos_angle, -1, 1))
    print("旋转角差（弧度）:", angle_diff)
    print("平移方向角差（弧度）:", angle_t)
    assert angle_diff < 1e-3, "恢复的旋转矩阵不正确"
    assert angle_t < 1e-3 or np.abs(angle_t - np.pi) < 1e-3, "恢复的平移方向不正确"
    print("✅ E_to_Rt 测试通过！")

# 运行测试
test_E_to_Rt()


