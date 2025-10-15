import numpy as np

from code_3_mineq import *

# -----------------------
# Case 1: 1D Linear Least Squares
# e(x) = Ax - b
A = np.array([[2.0]])
b = np.array([4.0])
e1 = lambda x: A @ x - b
J1 = lambda x: A
H1 = np.eye(1)
x0_1 = np.array([0.0])

# -----------------------
# Case 2: 1D Nonlinear Least Squares
# e(x) = sin(x) - y
y = np.array([0.5])
e2 = lambda x: np.sin(x) - y
J2 = lambda x: np.cos(x).reshape(1, 1)
H2 = np.eye(1)
x0_2 = np.array([1.0])

# -----------------------
# Case 3: 2D Linear Least Squares
# e(x) = A @ x - b
A3 = np.array([[1.0, 2.0],
               [3.0, 4.0]])
b3 = np.array([5.0, 11.0])
e3 = lambda x: A3 @ x - b3
J3 = lambda x: A3
H3 = np.eye(2)
x0_3 = np.array([0.0, 0.0])

# -----------------------
# Case 4: 2D Nonlinear Least Squares (Himmelblau residuals)
e4 = lambda x: np.array([
    x[0]**2 + x[1] - 11,
    x[0] + x[1]**2 - 7
])
J4 = lambda x: np.array([
    [2*x[0], 1],
    [1, 2*x[1]]
])
H4 = np.eye(2)
x0_4 = np.array([6.0, 6.0])

# -----------------------
# Run all algorithms on all test cases
cases = [
    ("Case 1 - 1D Linear", e1, J1, H1, x0_1),
    ("Case 2 - 1D Nonlinear", e2, J2, H2, x0_2),
    ("Case 3 - 2D Linear", e3, J3, H3, x0_3),
    ("Case 4 - 2D Nonlinear", e4, J4, H4, x0_4),
]

for name, e_func, J_func, H, x0 in cases:
    print(f"\n{name}")
    x_gd = opt_minls_gd(e_func, J_func, H, x0.copy())
    print("  Gradient Descent:", x_gd)
    x_gn = Gauss_Newton(e_func, J_func, H, x0.copy())
    print("  Gauss-Newton     :", x_gn)
    x_lm = Levenberg_Marquardt(e_func, J_func, H, x0.copy())
    print("  Levenberg-Marquardt:", x_lm)
