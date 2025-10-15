import numpy as np
import matplotlib.pyplot as plt

from code_kfs import filter_UKF
np.random.seed(42)

def generate_data(N_K):
    # 真实系统参数
    dt = 0.1
    x_0 = np.array([0.01, 0])  # 初始状态 [位置, 速度]
    P_0 = np.diag([0.1, 0.1])  # 初始协方差
    
    # 过程噪声和观测噪声
    Q = np.diag([0.01, 0.01])  # 过程噪声协方差
    R = np.diag([0.01])  # 观测噪声协方差
    
    # 输入控制 (这里设为0，模拟自由运动)
    us = [np.array([[0.0]]) for _ in range(N_K)]
    Qs = [Q for _ in range(N_K)]
    Rs = [R for _ in range(N_K)]
    
    # 生成真实轨迹和观测数据
    x_trues = [x_0]
    zs = []
    
    for k in range(N_K):
        # 非线性系统动态 (带有速度平方阻尼)
        x_next = np.array([
            [x_trues[-1][0] + x_trues[-1][1]*dt],
            [x_trues[-1][1] - 0.1*x_trues[-1][1]**2*dt]
        ]) + np.random.multivariate_normal([0,0], Q).reshape(2,1)
        x_trues.append(x_next[:, 0])
        # 非线性观测 (只观测位置，但带有非线性变换)
        z = np.array(np.sin(x_next[0])) + np.random.rand() * np.sqrt(R[0,0])
        zs.append(z)
    
    return x_0, P_0, us, Qs, Rs, x_trues, zs

def visualize(x_trues, xs_est, zs):
    # 提取真实和估计的位置、速度
    true_pos = [x[0] for x in x_trues]
    true_vel = [x[1] for x in x_trues]
    est_pos = [x[0] for x in xs_est]
    est_vel = [x[1] for x in xs_est]
    obs_pos = [np.arcsin(z[0]) for z in zs]  # 反变换观测值
    
    plt.figure(figsize=(12, 6))
    
    # 位置图
    plt.subplot(1, 2, 1)
    plt.plot(true_pos, label='True Position')
    plt.plot(est_pos, '--', label='Estimated Position')
    plt.plot(range(1, len(zs)+1), obs_pos, 'x', label='Observations')
    plt.xlabel('Time step')
    plt.ylabel('Position')
    plt.title('Position Estimation')
    plt.legend()
    
    # 速度图
    plt.subplot(1, 2, 2)
    plt.plot(true_vel, label='True Velocity')
    plt.plot(est_vel, '--', label='Estimated Velocity')
    plt.xlabel('Time step')
    plt.ylabel('Velocity')
    plt.title('Velocity Estimation')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N_K = 50
    
    # 定义非线性函数
    def f_func(x, u):
        dt = 0.1
        # 非线性动态模型 (带有速度平方阻尼)
        return np.array([
            x[0] + x[1]*dt,
            x[1] - 0.1*x[1]**2*dt
        ])
    def h_func(x):
        # 非线性观测模型 (sin函数)
        return np.array([np.sin(x[0])])
    
    # UKF参数
    alpha = 5
    kappa = 1.0
    beta = 8.0
    
    # 生成测试数据
    x_0, P_0, us, Qs, Rs, x_trues, zs = generate_data(N_K)
    
    # 运行UKF滤波器
    xs_est = filter_UKF(x_0, us, zs, f_func, h_func, Qs, Rs, P_0, alpha, kappa, beta, N_K)
    
    # 可视化结果
    visualize(x_trues, xs_est, zs)