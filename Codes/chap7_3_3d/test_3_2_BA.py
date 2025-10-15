import numpy as np
from code_3_2d3d import vo_BA_GN

np.random.seed(42)
fx, fy = 600, 500
cx, cy = 320, 240
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])
def test_vo_BA_GN():
    # 3D点
    N = 100 # must > 6
    p1s = np.random.uniform([-1, -1, 3], [1, 1, 6], (N, 3))  # Z应为正值
    # 真实位姿变换 T_2_to_1 (R, t)
    angle = np.deg2rad(34.7)
    R_gt = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    t_gt = np.array([0.9, -1.04, 0.2])
    T_2_to_1_gt = np.eye(4)
    T_2_to_1_gt[:3, :3] = R_gt
    T_2_to_1_gt[:3, 3] = t_gt
    # 相机2下的3D点
    T_1_to_2_gt = np.linalg.inv(T_2_to_1_gt)
    p1s_h = np.hstack([p1s, np.ones((N, 1))])  # 齐次坐标
    p2s_h = (T_1_to_2_gt @ p1s_h.T).T  # 在相机2下的坐标
    # 相机2下的2D点
    p2s_cam = p2s_h[:, :3]
    p2s_proj = (K @ p2s_cam.T).T
    nu_2s = p2s_proj / p2s_proj[:, [2]]  # 归一化
    nu_2s[:, 2] = 1.0  # 保证齐次形式

    # 6. 使用BA_GN估计
    # T_2_to_1_est = vo_BA_GD(N, p1s, nu_2s, K, alpha=5e-10)
    T_2_to_1_est = vo_BA_GN(N, p1s, nu_2s, K, alpha=1e-2)

    # 7. 评估误差
    def pose_error(T1, T2):
        R1, t1 = T1[:3, :3], T1[:3, 3]
        R2, t2 = T2[:3, :3], T2[:3, 3]
        rot_err = np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)) * 180 / np.pi
        trans_err = np.linalg.norm(t1 - t2)
        return rot_err, trans_err
    rot_err, trans_err = pose_error(T_2_to_1_gt, T_2_to_1_est)
    print("Ground Truth T_2_to_1:")
    print(T_2_to_1_gt)
    print("\nEstimated T_2_to_1:")
    print(T_2_to_1_est)
    print(f"\nRotation Error: {rot_err:.4f} degrees")
    print(f"Translation Error: {trans_err:.4f} units") # 有旋转时平移估计不准很正常

# 运行测试
test_vo_BA_GN()
