"""CNN for classification - predicting clothing types found in images"""

import tensorflow as tf
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (testing_images, testing_labels) = data.load_data()

# reshape to something that Conv2D will accept
training_images = training_images.reshape(60000, 28, 28, 1)
testing_images = testing_images.reshape(10000, 28, 28, 1)
# convert to binary values instead of 0-255
training_images = training_images / 255.0
testing_images = testing_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
model.fit(training_images, training_labels, epochs=10)
model.evaluate(testing_images, testing_labels)

classifications = model.predict(testing_images)
print(classifications[0])
print(testing_labels[0])

# [1.5337873e-11 4.6833150e-15 6.8722241e-12 2.3794763e-11 1.6816113e-11
# 4.2387438e-08 1.0751111e-11 3.7891855e-08 2.3104289e-14 9.9999988e-01]
# 9
