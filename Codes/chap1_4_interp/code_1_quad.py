import numpy as np

def find_insert_position(arr, x):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] <= x and (mid == len(arr) - 1 or x < arr[mid + 1]):
            return mid
        elif arr[mid] > x:
            right = mid - 1
        else:
            left = mid + 1
    return -1  # bad case

def quadratic_spline_interp(xs, ys, N):
    assert xs.shape == (N,) and ys.shape == (N,)
    P, p_ = np.zeros([3*N-3, 3*N-3]), np.zeros(3*N-3)
    for i in range(N-1):
        P_ii = np.array([
            [1, xs[i], xs[i]**2],
            [1, xs[i+1], xs[i+1]**2],
            [0, -1, -2*xs[i]],
        ])
        P_i1i = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 2*xs[i]],
        ])
        P[3*i:3*i+3, 3*i:3*i+3] = P_ii
        if i > 0:
            P[3*i:3*i+3, 3*i-3:3*i] = P_i1i
        p_[3*i:3*i+3] = np.array([ys[i], ys[i+1], 0])
    a_ = np.linalg.solve(P, p_)
    A = np.reshape(a_, [N-1, 3])
    def f_func(x):
        i = find_insert_position(xs, x)
        if i >= N-1:   
            a = A[N-2, :]
        else:
            a = A[i, :]
        return a[0] + a[1] * x + a[2] * x**2
    return f_func

def Quadratic_spline_sample(xs, ys, N, M):
    assert xs.shape == (N,)
    assert len(ys.shape) == 2 and ys.shape[0] == N
    d = ys.shape[1]
    a, b = np.min(xs), np.max(xs)
    new_xs = np.linspace(a, b, num=M, endpoint=True)
    new_ys = np.zeros([M, d])
    for dd in range(d):
        f_func = quadratic_spline_interp(xs, ys[:, dd], N)
        new_ys[:, dd] = np.array(list(map(f_func, new_xs)))
    return new_ys