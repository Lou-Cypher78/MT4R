import numpy as np
from code_2_cons import opt_eqc_gd_line
import numpy as np

########################## Examples ##########################

test1 = {
    'desc': "Test 1: Minimize f(x) = x1^2 + x2^2 s.t. x1 + x2 - 1 = 0",
    'f_func': lambda x: x[0]**2 + x[1]**2,
    'df_func': lambda x: np.array([2*x[0], 2*x[1]]),
    'ddf_func': lambda x: 2*np.eye(2),
    'g_func': lambda x: np.array([x[0] + x[1] - 1]),
    'dg_func': lambda x: np.array([[1.0, 1.0]]),
    'ddg_func': lambda x: np.zeros((1, 2, 2)),
    'x0': np.array([0.0, 0.0]),
    'expected': np.array([0.5, 0.5])
}

test2 = {
    'desc': "Test 2: Minimize f(x) = x1^2 + x2^2 + x3^2 s.t. x1 + x2 = 1, x2 + x3 = 1",
    'f_func': lambda x: x[0]**2 + x[1]**2 + x[2]**2,
    'df_func': lambda x: 2 * x,
    'ddf_func': lambda x: 2 * np.eye(3),
    'g_func': lambda x: np.array([x[0] + x[1] - 1, x[1] + x[2] - 1]),
    'dg_func': lambda x: np.array([[1, 1, 0], [0, 1, 1]]),
    'ddg_func': lambda x: np.zeros((2, 3, 3)),
    'x0': np.array([0.0, 0.0, 0.0]),
    'expected': np.array([1/3, 2/3, 1/3])
}

test3 = {
    'desc': "Test 3: Minimize f(x) = x1^2 + (x2-1)^2  s.t. x1*x2 = 5",
    'f_func': lambda x: (x[0])**2 + (x[1]-1)**2,
    'df_func': lambda x: np.array([2*x[0], 2*(x[1]-1)]),
    'ddf_func': lambda x: 2 * np.eye(2),
    'g_func': lambda x: np.array([x[0] * x[1] - 5]),
    'dg_func': lambda x: np.array([[x[1], x[0]]]),
    'ddg_func': lambda x: np.array([[[0, 1], [1, 0]]]),
    'x0': np.array([1.0, 1.0]),
    'expected': "Not calculated yet",
}


########################## Test function ##########################
def run_test_case(test_case):
    print(f"Running: {test_case['desc']}")
    x_opt = opt_eqc_gd_line(
        df_func=test_case['df_func'],
        ddf_func=test_case['ddf_func'],
        g_func=test_case['g_func'],
        dg_func=test_case['dg_func'],
        ddg_func=test_case['ddg_func'],
        x_0=test_case['x0']
    )
    print(f"Optimal x: {x_opt}")
    print(f"Constraint g(x): {test_case['g_func'](x_opt)}")
    if isinstance(test_case['expected'], np.ndarray):
        print(f"Expected: {test_case['expected']}")
        print(f"Error norm: {np.linalg.norm(x_opt - test_case['expected'])}")
    else:
        print(f"Expected: {test_case['expected']}")
    print("-" * 50)

if __name__ == "__main__":
    tests = [
        test1, 
        test2, 
        test3,
    ]
    for test in tests:
        run_test_case(test)
