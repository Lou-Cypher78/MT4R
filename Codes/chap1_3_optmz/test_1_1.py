import numpy as np
from code_1_nocons import * 

########################## examp 1 ##########################

def f1(x):
    return x**2 + 3*x + 4

def df1(x):
    return 2*x + 3

def ddf1(x):
    return np.array([[2]])  # 注意返回二维数组

# 理论最小值在 x = -1.5
test1 = {
    'func': (df1, ddf1),
    'x0': np.array([0.0]),
    'expected': np.array([-1.5]),
    'desc': "Quadratic function f(x)=x^2+3x+4"
}

########################## examp 2 ##########################

def f2(x):
    return np.exp(x) + x**2

def df2(x):
    return np.exp(x) + 2*x

def ddf2(x):
    return np.array([[np.exp(x) + 2]])

# 理论最小值在 x ≈ -0.3517
test2 = {
    'func': (df2, ddf2),
    'x0': np.array([0.0]),
    'expected': np.array([-0.3517]),
    'desc': "Exponential function f(x)=exp(x)+x^2"
}

########################## examp 3 ##########################

def f3(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1] + x[0] + x[1]

def df3(x):
    return np.array([2*x[0] + x[1] + 1, 2*x[1] + x[0] + 1])

def ddf3(x):
    return np.array([[2, 1], [1, 2]])

# 理论最小值在 x = [-1/3, -1/3]
test3 = {
    'func': (df3, ddf3),
    'x0': np.array([0.0, 0.0]),
    'expected': np.array([-1/3, -1/3]),
    'desc': "Multivariate quadratic function"
}

########################## examp 4 ##########################

def f4(x):
    return x**4 - 3*x**3 + 2

def df4(x):
    return 4*x**3 - 9*x**2

def ddf4(x):
    return np.array([[12*x**2 - 18*x]])

# 局部最小值在 x = 2.25
test4 = {
    'func': (df4, ddf4),
    'x0': np.array([2.0]),
    'expected': np.array([2.25]),
    'desc': "Non-convex polynomial function"
}


########################## Test function ##########################

def run_tests():
    tests = [test1, test2, test3, test4]
    
    for i, test in enumerate(tests):
        print(f"\nTest {i+1}: {test['desc']}")
        
        # 梯度下降测试
        df_func, _ = test['func']
        gd_result = opt_nc_gd(df_func, test['x0'])
        print(f"Gradient Descent Result: {gd_result}")
        print(f"Expected: {test['expected']}")
        print(f"GD Error: {np.linalg.norm(gd_result - test['expected'])}")
        
        # 牛顿法测试（如果有二阶导数）
        if len(test['func']) == 2:
            df_func, ddf_func = test['func']
            newton_result = opt_nc_newton(df_func, ddf_func, test['x0'])
            print(f"Newton's Method Result: {newton_result}")
            print(f"Expected: {test['expected']}")
            print(f"Newton Error: {np.linalg.norm(newton_result - test['expected'])}")

if __name__ == "__main__":
    run_tests()