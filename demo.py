import sys
import numpy as np
# import tensorflow as tf

# cifar = tf.keras.datasets.cifar100
# (x_train, y_train), (x_test, y_test) = cifar.load_data()
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=100,)

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=5, batch_size=64)

# hw = tf.config.get_visible_devices()
# tf.config.set_visible_devices(hw[0])

import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.linear_model import LinearRegression, Ridge
# # from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import make_blobs
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.activations import relu,linear
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.optimizers import Adam
# hw = tf.config.get_visible_devices()
# tf.config.set_visible_devices(hw[0])

# def gen_blobs():
#     classes = 6
#     m = 800
#     std = 0.4
#     centers = np.array([[-1, 0], [1, 0], [0, 1], [0, -1],  [-2,1],[-2,-1]])
#     X, y = make_blobs(n_samples=m, centers=centers, cluster_std=std, random_state=2, n_features=2)
#     return (X, y, centers, classes, std)

# X, y, centers, classes, std = gen_blobs()
# print("X.shape", X.shape, "y.shape", y.shape)
# X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)

# tf.random.set_seed(1234)
# model = Sequential(
#     [
#         ### START CODE HERE ### 
#         Dense(120, activation='relu', name="l1"),
#         Dense(40, activation='relu', name="l2"),
#         Dense(6, activation='linear', name="l3")
#         ### END CODE HERE ### 

#     ], name="Complex"
# )
# model.compile(
#     ### START CODE HERE ### 
#     loss=SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=Adam(0.01),
#     ### END CODE HERE ### 
# )

# model.fit(
#     X_train, y_train,
#     epochs=1000
# )

x = np.arange(1,5).reshape((2,2))
# print(np.linalg.norm(x[0]))
# print(np.linalg.norm(x[1]))
# print(np.linalg.norm(x[0]) - np.linalg.norm(x[1]))
# print(np.linalg.norm(x[0] - x[1]))
b = np.array([2,3])
print(x)
aa = np.array([])
print(np.append(aa,np.average(x[:,0])))
aa = np.array([1,-1])
print(np.sum(aa)/np.sum(aa))
print('a')
