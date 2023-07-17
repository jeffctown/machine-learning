"""CNN for multiclass classification - identifying rock paper or scissors in an image"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import collections.abc
#tensorflow_datasets needs the following alias to be done manually.te
collections.Iterable = collections.abc.Iterable
import tensorflow_datasets as tfds

load_saved_model = False
saved_model_location = "saved_model.h5"
EPOCHS = 25

def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/255)
    image = tf.image.random_flip_left_right(image)
    return image, label

if load_saved_model:
    model = tf.keras.models.load_model(saved_model_location)
else:
    TFDS_MODEL = 'rock_paper_scissors'
    data = tfds.load(TFDS_MODEL,
                 split='train',
                 as_supervised=True)
    val_data = tfds.load(TFDS_MODEL,
                     split='test',
                     as_supervised=True)

    train = data.map(augmentimages)
    train_batches = train.shuffle(100).batch(32)
    validation_batches = val_data.batch(32)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

    model.compile(loss = 'binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches, validation_steps=1, verbose=1)
    model.save(saved_model_location)

TEST_DIR = "test-images/"
test_image_names = ['rock1.jpg',
                    'rock2.jpg',
                    'paper1.jpg',
                    'paper2.jpg',
                    'scissors1.jpg',
                    'scissors2.jpg']

for image_name in test_image_names:
    PATH = TEST_DIR + image_name
    img = image.load_img(PATH, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(image_name)
    classes = classes[0]
    print(classes)

    if classes[0] > classes[1] and classes[0] > classes[2]:
        print(f'Prediction: paper! {classes[0]*100}%')
    if classes[1] > classes[0] and classes[1] > classes[2]:
        print(f'Prediction: rock! {classes[1]*100}%')
    if classes[2] > classes[1] and classes[2] > classes[0]:
        print(f'Prediction: scissors! {classes[2]*100}%')
