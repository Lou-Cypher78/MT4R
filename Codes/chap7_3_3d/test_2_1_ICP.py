import numpy as np
from code_2_icpdlt import vo_ICP

# 测试函数
def test_vo_ICP():
    np.random.seed(42)
    N = 30
    p1s = np.random.rand(N, 3)
    angle = np.pi / 6  # 30 degrees
    axis = np.array([0, 1.2, 1])  # 绕 z 轴旋转
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R_1_to_2 = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    t_1_to_2 = np.array([1.5, 1, 0]) # 1 to 2
    p2s = (R_1_to_2 @ p1s.T).T + t_1_to_2
    R_2_to_1, t_2_to_1 = R_1_to_2.T, - R_1_to_2.T @ t_1_to_2


    T_2_to_1_est = vo_ICP(N, p1s, p2s)

    R_est = T_2_to_1_est[:3, :3]
    t_est = T_2_to_1_est[:3, 3]
    print("True Rotation:\n", R_2_to_1)
    print("Estimated Rotation:\n", R_est)
    print("Rotation Error:\n", R_2_to_1 - R_est)
    print("\nTrue Translation:\n", t_2_to_1)
    print("Estimated Translation:\n", t_est)
    print("Translation Error:\n", t_2_to_1 - t_est)
    rot_error = np.linalg.norm(R_2_to_1 - R_est)
    trans_error = np.linalg.norm(t_2_to_1 - t_est)
    print(f"\nRotation Error (Frobenius norm): {rot_error:.6f}")
    print(f"Translation Error (L2 norm): {trans_error:.6f}") 

# 运行测试
test_vo_ICP()