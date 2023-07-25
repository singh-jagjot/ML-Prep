import numpy as np
import tensorflow as tf

X_train = np.array([[1.0], [2.0]], dtype=np.float32)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)

linear_layer = tf.keras.layers.Dense(units=1, activation="linear",)
print(linear_layer(X_train[0].reshape(1, 1)))

set_w = np.array([[200]])
set_b = np.array([100])
linear_layer.set_weights([set_w, set_b])
w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}")
print(linear_layer(X_train))

X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

pos = Y_train == 1
neg = Y_train == 0

model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_dim = 1, activation="sigmoid", name = "L1")])
model.summary()
logistic_layer = model.get_layer("L1")

print(logistic_layer.get_weights())
set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())
print(logistic_layer(X_train))

