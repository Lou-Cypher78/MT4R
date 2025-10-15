import numpy as np
from code_2_cons import opt_neqc_LSGD

ar = lambda x: np.array([x])
arr = lambda x: np.array([[x]])

def test_case1():
    # 目标函数: f(x) = (x-2)^2
    # 约束: x <= 1
    # 解析解: 无约束x=2，但被约束限制为 x=1
    f_func = lambda x: (x-2)**2
    df_func = lambda x: (2*(x-2))[None, :]
    h_func = lambda x: x - 1
    dh_func = lambda x: arr(1)
    x_0 = ar(0.5)  # 初始点在可行域外
    lambda_b = 100
    x_opt = opt_neqc_LSGD(f_func, df_func, h_func, dh_func, x_0, lambda_b, gamma=0.9)
    print(f"Test Case 1 - Optimal x: {x_opt[0]:.4f}")
    assert abs(x_opt - 1.0) < 1e-3



def test_case2():
    # 目标函数: f(x) = (x1-1)^2 + (x2-2)^2
    # 约束: x1 + x2 <= 2
    # 解析解: 在约束边界上    
    f_func = lambda x: (x[0]-1)**2 + (x[1]-2)**2
    df_func = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
    h_func = lambda x: ar(x[0] + x[1] - 2)  # h(x)<=0
    dh_func = lambda x: np.array([[1, 1]])
    x_0 = np.array([0.0, 0.0])
    lambda_b = 100
    x_opt = opt_neqc_LSGD(f_func, df_func, h_func, dh_func, x_0, lambda_b, gamma=0.8)
    print(f"Test Case 2 - Optimal x: {x_opt}")
    # 验证是否在约束边界上且接近解析解
    assert abs(x_opt[0] + x_opt[1] - 2) < 1e-3
    assert abs(x_opt[0] - 0.5) < 0.1
    assert abs(x_opt[1] - 1.5) < 0.1


def test_case3():
    # 目标函数: f(x) = x^2
    # 约束: x^2 - x <= 0 (即 0 <= x <= 1)
    # 解析解: x=0    
    f_func = lambda x: x**2
    df_func = lambda x: (2*x)[None, :]
    h_func = lambda x: x**2 - x
    dh_func = lambda x: (2*x - 1)[None, :]
    x_0 = ar(0.1)
    lambda_b = 100
    x_opt = opt_neqc_LSGD(f_func, df_func, h_func, dh_func, x_0, lambda_b, gamma=0.9, epsilon=1e-7)
    print(f"Test Case 3 - Optimal x: {x_opt}")
    assert abs(x_opt - 0.0) < 1e-3



def test_case4():
    # 目标函数: f(x) = (x1-3)^2 + (x2-4)^2
    # 约束: x1 >= 1, x2 >= 1, x1 + x2 <= 4
    # 解析解: (2,2)    
    f_func = lambda x: (x[0]-3)**2 + (x[1]-4)**2
    df_func = lambda x: np.array([2*(x[0]-3), 2*(x[1]-4)])
    # 多个不等式约束 h_i(x) <= 0
    h_func = lambda x: np.array([
        1 - x[0],  # h1(x) = 1-x1 <=0 → x1 >=1
        1 - x[1],  # h2(x) = 1-x2 <=0 → x2 >=1
        x[0] + x[1] - 4  # h3(x) = x1+x2-4 <=0 → x1+x2 <=4
    ])
    # 每个约束的导数
    dh_func = lambda x: np.array([
        [-1, 0],   # ∇h1
        [0, -1],   # ∇h2
        [1, 1]     # ∇h3
    ])
    x_0 = np.array([0.0, 0.0])
    lambda_b = 100
    x_opt = opt_neqc_LSGD(f_func, df_func, h_func, dh_func, x_0, lambda_b)
    print(f"Test Case 4 - Optimal x: {x_opt}")
    assert abs(x_opt[0] - 2.0) < 0.1
    assert abs(x_opt[1] - 2.0) < 0.1



if __name__ == "__main__":
    # test_case1()
    # test_case2()
    test_case3()
    # test_case4()