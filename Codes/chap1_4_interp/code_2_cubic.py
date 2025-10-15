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


def cubic_spline_interp(xs, ys, N):
    assert xs.shape == (N,) and ys.shape == (N,)
    P, p_ = np.zeros([4*N-4, 4*N-4]), np.zeros(4*N-4)
    for i in range(N-1):
        P_ii = np.array([
            [1, xs[i], xs[i]**2, xs[i]**3],
            [1, xs[i+1], xs[i+1]**2, xs[i+1]**3],
            [0, -1, -2*xs[i], -3*xs[i]**2],
            [0, 0, -2, -6*xs[i]],
        ])
        P_i1i = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 2*xs[i], 3*xs[i]**2],
            [0, 0, 2, 6*xs[i]],
        ])
        if i == 0:
            P[:3, :4] = P_ii[:3, :] # first 3 lines
            P[3, -4:] = np.array([0, 1, 2*xs[-1], 3*xs[-1]**2])
        else:
            P[4*i:4*i+4, 4*i:4*i+4] = P_ii
            P[4*i:4*i+4, 4*i-4:4*i] = P_i1i
        p_[4*i:4*i+4] = np.array([ys[i], ys[i+1], 0, 0])
    a_ = np.linalg.solve(P, p_)
    A = np.reshape(a_, [N-1, 4])
    def f_func(x):
        i = find_insert_position(xs, x)
        if i >= N-1:   
            a = A[N-2, :]
        else:
            a = A[i, :]
        return a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3
    return f_func

def Cubic_spline_sample(xs, ys, N, M):
    assert xs.shape == (N,)
    assert len(ys.shape) == 2 and ys.shape[0] == N
    d = ys.shape[1]
    a, b = np.min(xs), np.max(xs)
    new_xs = np.linspace(a, b, num=M, endpoint=True)
    new_ys = np.zeros([M, d])
    for dd in range(d):
        f_func = cubic_spline_interp(xs, ys[:, dd], N)
        new_ys[:, dd] = np.array(list(map(f_func, new_xs)))
    return new_ys