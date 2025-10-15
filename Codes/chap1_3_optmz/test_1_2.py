import numpy as np
from code_1_nocons import * 

########################## Examples ##########################

# 1. 二次函数 (凸函数)
test1 = {
    'func': lambda x: x[0]**2 + 3*x[0] + 4,
    'dfunc': lambda x: np.array([2*x[0] + 3]),
    'ddfunc': lambda x: np.array([[2]]),
    'x0': np.array([0.0]),
    'expected': np.array([-1.5]),
    'desc': "Quadratic function f(x)=x^2+3x+4 (min at x=-1.5)"
}

# 2. 四次函数 (非凸函数)
test2 = {
    'func': lambda x: x[0]**4 - 3*x[0]**3 + 2,
    'dfunc': lambda x: np.array([4*x[0]**3 - 9*x[0]**2]),
    'ddfunc': lambda x: np.array([[12*x[0]**2 - 18*x[0]]]),
    'x0': np.array([4.0]),
    'expected': np.array([2.25]),  # 全局最小值
    'desc': "Quartic function f(x)=x^4-3x^3+2 (min at x=2.25)"
}

# 3. 指数函数
test3 = {
    'func': lambda x: np.exp(x[0]) + np.exp(-x[0]),
    'dfunc': lambda x: np.array([np.exp(x[0]) - np.exp(-x[0])]),
    'ddfunc': lambda x: np.array([[np.exp(x[0]) + np.exp(-x[0])]]),
    'x0': np.array([1.0]),
    'expected': np.array([0.0]),
    'desc': "Exponential function f(x)=e^x + e^-x (min at x=0)"
}

### 多维测试用例

# 4. 二维二次函数 (凸函数)
test4 = {
    'func': lambda x: x[0]**2 + x[1]**2 + x[0]*x[1] + x[0] + x[1],
    'dfunc': lambda x: np.array([2*x[0] + x[1] + 1, 2*x[1] + x[0] + 1]),
    'ddfunc': lambda x: np.array([[2, 1], [1, 2]]),
    'x0': np.array([1.0, 1.0]),
    'expected': np.array([-1/3, -1/3]),
    'desc': "2D quadratic function (min at [-1/3, -1/3])"
}

# 5. Rosenbrock函数 (经典测试函数)
test5 = {
    'func': lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2,
    'dfunc': lambda x: np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ]),
    'ddfunc': lambda x: np.array([
        [2 - 400*x[1] + 1200*x[0]**2, -400*x[0]],
        [-400*x[0], 200]
    ]),
    'x0': np.array([-1.5, 2.0]),
    'expected': np.array([1.0, 1.0]),
    'desc': "Rosenbrock function (min at [1,1])"
}

# 6. 三维二次函数
test6 = {
    'func': lambda x: x[0]**2 + 2*x[1]**2 + 3*x[2]**2 + x[0]*x[1] - x[0]*x[2] + x[0] + 2*x[1] - 3*x[2],
    'dfunc': lambda x: np.array([
        2*x[0] + x[1] - x[2] + 1,
        4*x[1] + x[0] + 2,
        6*x[2] - x[0] - 3
    ]),
    'ddfunc': lambda x: np.array([
        [2, 1, -1],
        [1, 4, 0],
        [-1, 0, 6]
    ]),
    'x0': np.array([0.0, 0.0, 0.0]),
    'expected': np.array([-1.5, -0.5, 0.25]),
    'desc': "3D quadratic function"
}


########################## Test function ##########################

def run_tests():
    # 收集所有测试用例
    test_cases = [test1, test2, test3, test4, test5, test6]
    
    # 定义算法列表
    algorithms = [
        # ("Gradient Descent with Line Search", opt_nc_LSGD),
        # ("Damped Newton's Method", opt_nc_damp_newton),
        ("Conjugate Gradient Method", opt_nc_conj_grad)
    ]
    
    # 设置容差
    tolerance = 1e-3
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i}: {test['desc']} ===")
        print(f"Initial point: {test['x0']}")
        print(f"Expected minimum: {test['expected']}")
        
        for name, algorithm in algorithms:
            try:
                if name == "Gradient Descent with Line Search":
                    result = algorithm(test['func'], test['dfunc'], test['x0'])
                elif name == "Damped Newton's Method":
                    result = algorithm(test['func'], test['dfunc'], test['ddfunc'], test['x0'])
                else:  # Conjugate Gradient
                    result = algorithm(test['func'], test['dfunc'], test['x0'])
                
                # 验证结果
                distance = np.linalg.norm(result - test['expected'])
                success = distance < tolerance
                
                print(f"\n{name}:")
                print(f"Result: {result}")
                print(f"Distance to expected: {distance:.6f}")
                print(f"Success: {'Yes' if success else 'No'}")
                
                # 验证函数值是否确实在结果点处更小
                f_expected = test['func'](test['expected'])
                f_result = test['func'](result)
                print(f"Function value at result: {f_result:.6f}")
                print(f"Function value at expected: {f_expected:.6f}")
                
            except Exception as e:
                print(f"\n{name} failed with error: {str(e)}")

if __name__ == "__main__":
    run_tests()