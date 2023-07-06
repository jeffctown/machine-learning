"""CNN for classification - predicting whether an image contains a human or horse with transfer learning (based on Google Inception V3)"""

import urllib.request
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop

WEIGHTS_URL = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
WEIGHTS_FILE = "inception_v3.h5"
TRAINING_DIR = "../convolutional-neural-network-human-or-horse-classification/images/horse-or-human/"
VALIDATION_DIR = "../convolutional-neural-network-human-or-horse-classification/images/validation-horse-or-human"

urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)

pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
pre_trained_model.load_weights(WEIGHTS_FILE)

# whoa
# pre_trained_model.summary()

for layer in pre_trained_model.layers:
    layer.trainable = False

# get training data
train_datagen = ImageDataGenerator(
    rescale=1/255,
    # image augmentation to improve the training data
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
# Found 1027 images belonging to 2 classes.

# get validation data
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(150, 150),
                                                              class_mode='binary')
# # Found 256 images belonging to 2 classes.


last_layer = pre_trained_model.get_layer('mixed7')
print(f'last output shape: {last_layer.output_shape}')
# last output shape: (None, 7, 7, 768) - so 7x7 images
last_output = last_layer.output

# flatten to 1 dimension
x = layers.Flatten()(last_output)
# hidden layer of 1024 units
x = layers.Dense(1024, activation='relu')(x)
# add final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc'])

EPOCHS = 8
model.fit_generator(train_generator,
                    epochs=EPOCHS,
                    validation_data=validation_generator)
# Epoch 8/8
# 11/11 [==============================] - 10s 909ms/step - loss: 0.0112 - acc: 0.9961 - val_loss: 0.0328 - val_acc: 0.9883
