"""A super simple neural network"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

print(tf.__version__)
l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer="sgd", loss="mean_squared_error")

# simple data set that appears to be y = 2x - 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
# [[18.984734]], which is close to 19
print(f'Heres what I learned: {l0.get_weights()}')
