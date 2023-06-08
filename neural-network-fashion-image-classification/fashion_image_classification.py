"""A simple neural network to categorize fashion images"""
import tensorflow as tf

class AccuracyCompletionCallback(tf.keras.callbacks.Callback):
    """A training callback class to stop model training at a specific accuracy"""
    def on_epoch_end(self, epoch, logs={}):
        print(f'Checking accuracy callback...accuracy is {logs.get("accuracy")}')
        __accuracy = 0.95
        if logs.get('accuracy') > __accuracy:
            print(f'Reached {__accuracy} accuracy so training is complete!')
            self.model.stop_training = True

callback = AccuracyCompletionCallback()

# get TF's sample fashion image data
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# convert grayscale values to binary (0 or 1)
training_images = training_images / 255.0
test_images = test_images / 255.0

# setup NN
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# setup ML algorithm and train
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(training_images,
          training_labels,
          callbacks=[callback],
          epochs=50)

# evaluate
model.evaluate(test_images, test_labels)

# explore the data / predictions deeper
classifications = model.predict(test_images)

# this one it got wrong
# i = 17
# print(f'{classifications[i]}')
# print(f'{test_labels[i]}')

# Reached 0.95 accuracy so training is complete!
# 1875/1875 [==============================] - 1s 795us/step - loss: 0.1329 - accuracy: 0.9505
# 313/313 [==============================] - 0s 365us/step - loss: 0.4381 - accuracy: 0.8830
# Looks to be overfitting
