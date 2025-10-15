import gym
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PIDController:
    def __init__(self, kp, ki, kd, dt=0.02):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.previous_error = 0.0
        self.integral = 0.0
        
    def update(self, error):
        pp = self.kp * error
        self.integral += error * self.dt
        ii = self.ki * self.integral
        derivative = (error - self.previous_error) / self.dt
        dd = self.kd * derivative
        control_output = pp + ii + dd
        self.previous_error = error
        return control_output


def run_cartpole_pid():
    """
    运行倒立摆PID控制仿真
    """
    try:
        env = gym.make('CartPole-v1', render_mode='human')  # 新版本gym的渲染方式
    except:
        env = gym.make('CartPole-v1')

    kp_angle = 15.0    # 角度比例增益
    ki_angle = 0.1     # 角度积分增益
    kd_angle = 1.0     # 角度微分增益
    pid_controller = PIDController(kp_angle, ki_angle, kd_angle)
    
    episodes = 5       # 运行5个episode
    max_steps = 500     # 每个episode最大步数
    all_rewards = []
    all_angles = []
    all_positions = []
    for episode in range(episodes):
        # 重置环境 - 兼容新旧版本gym
        try:
            state, info = env.reset()
        except ValueError:
            state = env.reset()
            info = {}
        episode_rewards = 0
        episode_angles = []
        episode_positions = []
        print(f"\n开始第 {episode + 1} 个episode")
        for step in range(max_steps):
            cart_position, cart_velocity, pole_angle, pole_velocity = state
            angle_error = -pole_angle
            pid_controller.integral = np.clip(pid_controller.integral, -1.0, 1.0)
            control_action = pid_controller.update(angle_error)
            action = 1 if control_action > 0 else 0
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except:
                next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_rewards += reward
            episode_angles.append(pole_angle)
            episode_positions.append(cart_position)
            if done:
                print(f"Episode {episode + 1} 在 {step + 1} 步后结束，总奖励: {episode_rewards:.2f}")
                break
            if step == max_steps - 1:
                print(f"Episode {episode + 1} 达到最大步数 {max_steps}，总奖励: {episode_rewards:.2f}")        
        all_rewards.append(episode_rewards)
        all_angles.append(episode_angles)
        all_positions.append(episode_positions)
    env.close()
    plot_results(all_rewards, all_angles, all_positions)



def plot_results(rewards, angles, positions):
    """
    绘制仿真结果
    """
    plt.figure(figsize=(15, 10))
    
    # 1. 奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(rewards, 'b-o', linewidth=2, markersize=4)
    plt.title('每个Episode的总奖励')
    plt.xlabel('Episode')
    plt.ylabel('总奖励')
    plt.grid(True)
    
    # 2. 最后一个episode的角度变化
    plt.subplot(2, 2, 2)
    if angles:
        last_angles = angles[-1]
        plt.plot(last_angles, 'r-', linewidth=2)
        plt.title('最后一个Episode的角度变化')
        plt.xlabel('时间步')
        plt.ylabel('角度 (弧度)')
        plt.grid(True)
    
    # 3. 最后一个episode的小车位置
    plt.subplot(2, 2, 3)
    if positions:
        last_positions = positions[-1]
        plt.plot(last_positions, 'g-', linewidth=2)
        plt.title('最后一个Episode的小车位置')
        plt.xlabel('时间步')
        plt.ylabel('位置')
        plt.grid(True)
    
    # 4. 所有episode的平均角度变化
    plt.subplot(2, 2, 4)
    if angles:
        max_length = max(len(angle) for angle in angles)
        avg_angles = np.zeros(max_length)
        count = np.zeros(max_length)
        
        for angle_list in angles:
            for i, angle in enumerate(angle_list):
                avg_angles[i] += angle
                count[i] += 1
        
        # 只绘制有数据的部分
        valid_indices = count > 0
        if np.any(valid_indices):
            avg_angles_valid = avg_angles[valid_indices] / count[valid_indices]
            plt.plot(avg_angles_valid, 'purple', linewidth=2)
            plt.title('所有Episode的平均角度变化')
            plt.xlabel('时间步')
            plt.ylabel('平均角度 (弧度)')
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()



def pid_parameter_tuning():
    """
    PID参数调优示例
    """
    parameter_sets = [
        {'kp': 10.0, 'ki': 0.05, 'kd': 0.5, 'name': '保守参数'},
        {'kp': 15.0, 'ki': 0.1, 'kd': 1.0, 'name': '中等参数'},
        {'kp': 25.0, 'ki': 0.2, 'kd': 2.0, 'name': '激进参数'}
    ]
    
    print("PID参数调优示例:")
    for params in parameter_sets:
        print(f"\n测试参数: {params['name']}")
        print(f"KP={params['kp']}, KI={params['ki']}, KD={params['kd']}")

# 如果还有numpy兼容性问题，添加这个修复
def fix_numpy_compatibility():
    """修复numpy兼容性问题"""
    try:
        import numpy as np
        # 为兼容性添加bool8别名
        if not hasattr(np, 'bool8'):
            np.bool8 = np.bool_
    except:
        pass

if __name__ == "__main__":
    # 修复兼容性问题
    fix_numpy_compatibility()
    
    # 运行主仿真
    run_cartpole_pid()
    
    # 显示参数调优示例
    pid_parameter_tuning()