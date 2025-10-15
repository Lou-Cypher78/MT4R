import numpy as np
from numpy.linalg import norm
from code_4_qp import opt_qp_barrier

# 定义测试用例
test_cases = {
    'test1': {
        'H': np.array([[2]]),
        'g': np.array([-2]),
        'M_eq': np.zeros((0, 1)),  # 无等式约束
        'b_eq': np.zeros(0),
        'M': np.array([[1]]),
        'b': np.array([0.5]),
        'x0': np.array([0]),
        'expected': np.array([0.5]),
        'desc': "简单二次规划问题 (1D变量，1个不等式约束)"
        # min (x^2 - 2x) s.t. x ≤ 0.5 
    },
    'test2': {
        'H': np.array([[2., 0.], [0., 2.]]),
        'g': np.array([-2., -4.]),
        'M_eq': np.array([[1., 1.]]),
        'b_eq': np.array([1.]),
        'M': np.array([[1., 0.], [0., 1.]]),
        'b': np.array([0.6, 0.6]),
        'x0': np.array([0., 0.]),
        'expected': np.array([0.4, 0.6]),
        'desc': "中等复杂度问题 (2D变量，1个等式约束和2个不等式约束)"
        # min (x1^2 + x2^2 - 2x1 - 4x2), s.t. x1 + x2 = 1, x1 ≤ 0.5, x2 ≤ 0.5
    },
    'test3': {
        'H': np.array([[1., -2, 0], [-2, 5, -1], [0, -1, 2]]),
        'g': np.array([1., 1, 1]),
        'M_eq': np.array([[1., 1, 1]]),
        'b_eq': np.array([3.]),
        'M': np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1]]),
        'b': np.array([2., 2, 2]),
        'x0': np.array([1., 0., 0.]),
        'expected': np.array([1.0, 1.0, 1.0]),  # 近似解
        'desc': "高维复杂问题 (3D变量，1个等式约束和3个不等式约束)"
        #  x1 + x2 + x3 = 3, x1 ≤ 2, x2 ≤ 2, x3 ≤ 2
    }
}

#######################################################################

def run_qp_tests():
    print("=== 开始 QP 问题求解器测试 ===")
    print(f"共 {len(test_cases)} 个测试用例\n")
    for name in ['test2']: # 'test1' 'test3'
        test = test_cases[name]
        print(f"运行测试: {name} - {test['desc']}")
        x_opt = opt_qp_barrier(
            test['H'], test['g'], 
            test['M_eq'], test['b_eq'], 
            test['M'], test['b'], 
            test['x0'],
            lambda_b0=10, 
            gamma=0.95, 
            beta=0.9, 
            epsilon_1=0.5, 
            epsilon_2=1e-4
        )
        print(f"初始点: {test['x0']}")
        print(f"期望解: {test['expected']}")
        print(f"实际解: {x_opt}")
        # 验证约束条件
        if test['M_eq'].shape[0] > 0:
            eq_residual = norm(test['M_eq'] @ x_opt - test['b_eq'])
            print(f"等式约束残差: {eq_residual:.6f}")
        ineq_satisfied = np.all(test['M'] @ x_opt <= test['b'] + 1e-4)
        print(f"不等式约束满足: {ineq_satisfied}")
        # 验证与期望解的接近程度
        error = norm(x_opt - test['expected'])
        print(f"与期望解的误差: {error:.6f}")
        print("-" * 50)
    print("=== 测试完成 ===")


# 运行测试
if __name__ == "__main__":
    run_qp_tests()