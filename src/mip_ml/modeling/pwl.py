import numpy as np
from .function import AbstractFunction


def make_pwl_pts(function: AbstractFunction, tol=1e-2):
    f = function
    x_init = list(set([f.lb, *f.infrection_pts, f.ub]))  # 最初は境界と変曲点から探索を始める

    def split_pts(lb, ub):
        xs = []
        grad = (f(ub) - f(lb)) / (ub - lb)
        def f_pwl(x): return grad*(x-lb) + f(lb)
        c = f.df_inv(grad, (lb+ub)/2)
        err = np.abs(f(c) - f_pwl(c))

        if err > tol:
            xs.append(c)
            xs.extend(split_pts(lb, c))
            xs.extend(split_pts(c, ub))
        return xs

    inner_pts = []
    for i in range(len(x_init)-1):
        inner_pts.extend(split_pts(x_init[i], x_init[i+1]))  # tolを満たすまで分割する

    ret = np.array(sorted(x_init + inner_pts))

    return ret
