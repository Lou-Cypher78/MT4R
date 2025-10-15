import numpy as np
from code_1_fmat import calc_vo_Fmat_8p

np.random.seed(42)
# 相机内参矩阵 K
fx, fy = 600, 500
cx, cy = 320, 240
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])
K_inv = np.linalg.inv(K)

def skew(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])


def gen_data():
    # 生成随机的 3D 点，确保在第一个相机前方
    N = 50
    pts_3d = np.random.uniform(
        low=[-1, -1, 4], high=[1, 1, 8], size=(N, 3)
    ) 
    # 定义第二个相机的相对位姿（R, t）
    angle = np.deg2rad(30)
    R_true = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    t_true = np.array([0.5, 0, 0])  # 向右平移 0.5 米
    T_2_to_1 = np.eye(4)
    T_2_to_1[:3, :3] = R_true
    T_2_to_1[:3, 3] = t_true
    T_1_to_2 = np.linalg.inv(T_2_to_1)
    # 投影到2个相机
    pts_cam1 = pts_3d.T
    pts_img1_h = K @ pts_cam1
    pts_img1 = (pts_img1_h[:2] / pts_img1_h[2]).T  # 像素坐标
    pts_cam2 = T_1_to_2[:3, :3] @ pts_cam1 + T_1_to_2[:3, 3:]
    valid_idx = pts_cam2[2] > 0 # 过滤第二个相机后方的点
    pts_cam1 = pts_cam1[:, valid_idx]
    pts_cam2 = pts_cam2[:, valid_idx]
    pts_img1 = pts_img1[valid_idx.T]
    pts_img2_h = K @ pts_cam2
    pts_img2 = (pts_img2_h[:2] / pts_img2_h[2]).T
    # 转换为归一化平面上的齐次坐标
    pts_img1_h = np.hstack([pts_img1, np.ones((pts_img1.shape[0], 1))])
    pts_img2_h = np.hstack([pts_img2, np.ones((pts_img2.shape[0], 1))])
    E = skew(t_true) @ R_true
    return pts_img1_h, pts_img2_h, T_2_to_1



def calc_error(T_2_to_1_est, T_2_to_1):
    print("估计的相对位姿 T_2_to_1:")
    print(T_2_to_1_est)
    # 真实的相对位姿 T_2_to_1
    T_true = T_2_to_1
    R_true = T_true[:3, :3]
    t_true = T_true[:3, 3]
    print("\n真实的相对位姿 T_2_to_1:")
    print(T_true)
    # 你可以进一步比较 R 和 t 的误差
    R_est = T_2_to_1_est[:3, :3]
    t_est = T_2_to_1_est[:3, 3]
    # 计算旋转误差（角度）
    R_diff = R_est @ R_true.T
    angle_error = np.arccos((np.trace(R_diff) - 1) / 2)
    print(f"\n旋转误差（角度）: {np.rad2deg(angle_error):.2f} 度")
    # 平移误差（方向）
    t_true_unit = t_true.flatten() / np.linalg.norm(t_true)
    t_est_unit = t_est / np.linalg.norm(t_est)
    t_error_angle = np.arccos(np.clip(np.dot(t_true_unit, t_est_unit), -1.0, 1.0))
    print(f"平移方向误差: {np.rad2deg(t_error_angle):.2f} 度")

    

if __name__ == "__main__":
    nu_1s, nu_2s, T_2_to_1 = gen_data()
    T_2_to_1_est = calc_vo_Fmat_8p(nu_1s.shape[0], nu_1s, nu_2s, K)
    calc_error(T_2_to_1_est, T_2_to_1)
