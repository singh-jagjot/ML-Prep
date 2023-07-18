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


def gradient_descent(w_in, b_in, x, y, alpha, num_of_itr):
    w = w_in
    b = b_in

    for i in range(num_of_itr):
        dj_dw, dj_db = comp_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b

# Initial Parameters
w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2

x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

w_final, b_final = gradient_descent(w_init, b_init, x_train ,y_train, tmp_alpha, iterations)

print(w_final, b_final)

print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")