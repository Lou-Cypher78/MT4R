import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
from code_1_fmat import triangulate_points

K = np.array([
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
])
np.random.seed(42)


########################## Components ##########################

def get_sim_data():
    # R and t
    theta = np.deg2rad(-30)  # 30度旋转
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    t = np.array([0.5, 0.1, 0])  # 基线
    T_1_to_2 = np.eye(4)
    T_1_to_2[:3, :3] = R
    T_1_to_2[:3, 3] = t
    T_2_to_1 = np.linalg.inv(T_1_to_2)
    # 3D points
    N = 50  # 点数
    points_3d = np.random.randn(N, 3) * 5
    points_3d[:, 2] = np.abs(points_3d[:, 2])  # 确保点在相机前方
    # 2D points
    nu_1s = np.zeros((N, 3))
    nu_2s = np.zeros((N, 3))
    for i in range(N):
        p_cam1 = points_3d[i]
        nu_1s[i] = K @ p_cam1 / p_cam1[2]
        p_cam2 = R @ p_cam1 + t
        nu_2s[i] = K @ p_cam2 / p_cam2[2]
    # noise
    noise_level = 1 # 像素噪声
    nu_1s_noisy = nu_1s + np.random.randn(*nu_1s.shape) * noise_level / K[0, 0]
    nu_2s_noisy = nu_2s + np.random.randn(*nu_2s.shape) * noise_level / K[0, 0]
    # nu_1s_noisy, nu_2s_noisy = nu_1s, nu_2s
    return N, nu_1s_noisy, nu_2s_noisy, T_2_to_1, points_3d


def calc_errors(N, p2s, p1s, nu_1s_noisy, nu_2s_noisy, points_3d):
    # 计算重投影误差
    errors1 = []
    errors2 = []
    for i in range(N):
        reproj1 = K @ p1s[i] / p1s[i, 2] # 重投影到相机1
        error1 = np.linalg.norm(reproj1[:2] - nu_1s_noisy[i][:2])
        reproj2 = K @ p2s[i] / p2s[i, 2] # 重投影到相机2
        error2 = np.linalg.norm(reproj2[:2] - nu_2s_noisy[i][:2])
        errors1.append(error1)
        errors2.append(error2)
    # 计算与真实3D点的误差
    position_errors = np.linalg.norm(p1s - points_3d, axis=1)
    # 打印统计信息
    print(f"平均重投影误差 (相机1): {np.mean(errors1):.4f} 像素")
    print(f"平均重投影误差 (相机2): {np.mean(errors2):.4f} 像素")
    print(f"平均3D位置误差: {np.mean(position_errors):.4f} 单位")
    print(f"最大3D位置误差: {np.max(position_errors):.4f} 单位")
    return position_errors



def visualzation(points_3d, p1s, position_errors):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(points_3d[:, 0], points_3d[:, 2], label='真实点')
    plt.scatter(p1s[:, 0], p1s[:, 2], label='重建点')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('X-Z平面视图')
    plt.legend()
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.hist(position_errors, bins=20)
    plt.xlabel('3D位置误差')
    plt.ylabel('点数')
    plt.title('3D重建误差分布')
    
    plt.tight_layout()
    plt.show()



########################## Test ##########################
    
    
if __name__ == "__main__":
    N, nu_1s_noisy, nu_2s_noisy, T_2_to_1, points_3d = get_sim_data()
    p1s, p2s = triangulate_points(N, nu_1s_noisy, nu_2s_noisy, T_2_to_1, K)
    position_errors = calc_errors(N, p2s, p1s, nu_1s_noisy, nu_2s_noisy, points_3d)
    visualzation(points_3d, p1s, position_errors)
