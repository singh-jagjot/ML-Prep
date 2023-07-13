import numpy as np

# Compute the cost


def compute_cost(w, b, x, y):
    cost = 0
    m = x.shape[0]
    for i in range(m):
        f_wb = w*x[i] + b
        cost = cost + (f_wb - y[i])**2
    tot = 1/(2*m) * cost
    return tot

# Compute Gradient


def comp_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w*x[i] + b
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

# Gradient Descent
# def gradient_descent():
