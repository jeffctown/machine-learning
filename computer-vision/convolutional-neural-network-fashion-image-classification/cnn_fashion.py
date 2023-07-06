"""Tensorflow to bring in all of the juicy ML goodness"""
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
