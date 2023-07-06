"""CNN for classification - predicting whether an image contains a human or horse"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.optimizers.experimental import RMSprop
import numpy as np

TRAINING_DIR = "images/horse-or-human/"
VALIDATION_DIR = "images/validation-horse-or-human"

# rescale to 1./255
train_datagen = ImageDataGenerator(
    rescale=1/255,
    # after initial results, added image augmentation to improve the training data
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'

)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(300, 300),
    class_mode='binary'
)
# Found 1027 images belonging to 2 classes.

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 298, 298, 16)      448
#
#  max_pooling2d (MaxPooling2D  (None, 149, 149, 16)     0
#  )
#
#  conv2d_1 (Conv2D)           (None, 147, 147, 32)      4640
#
#  max_pooling2d_1 (MaxPooling  (None, 73, 73, 32)       0
#  2D)
#
#  conv2d_2 (Conv2D)           (None, 71, 71, 64)        18496
#
#  max_pooling2d_2 (MaxPooling  (None, 35, 35, 64)       0
#  2D)
#
#  conv2d_3 (Conv2D)           (None, 33, 33, 64)        36928
#
#  max_pooling2d_3 (MaxPooling  (None, 16, 16, 64)       0
#  2D)
#
#  conv2d_4 (Conv2D)           (None, 14, 14, 64)        36928
#
#  max_pooling2d_4 (MaxPooling  (None, 7, 7, 64)         0
#  2D)
#
#  flatten (Flatten)           (None, 3136)              0
#
#  dense (Dense)               (None, 512)               1606144
#
#  dense_1 (Dense)             (None, 2)                 1026
#
# =================================================================
# Total params: 1,704,610
# Trainable params: 1,704,610
# Non-trainable params: 0
# _________________________________________________________________

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# get validation data
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(300, 300),
                                                              class_mode='binary')
# Found 256 images belonging to 2 classes.

model.fit_generator(train_generator,
                    epochs=15,
                    validation_data=validation_generator)

# Epoch 15/15
# 33/33 [==============================] - 15s 444ms/step -
# loss: 5.4388e-05 - accuracy: 1.0000 - val_loss: 4.9753 - val_accuracy: 0.8047
# looks to be overfitting

# test it!
TEST_DIR = "images/testing-horse-or-human/"
test_image_names = ['human-1.jpg',
                    'human-2.jpg',
                    'human-3.jpg',
                    'horse-1.jpg',
                    'horse-2.jpg',
                    'horse-3.jpg']

for image_name in test_image_names:
    PATH = TEST_DIR + image_name
    img = image.load_img(PATH, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes[0])

    if classes[0] > 0.5:
        print(f'{image_name} is a human!')
    else:
        print(f'{image_name} is a horse!')

# 1/1 [==============================] - 0s 53ms/step
# [1.]
# human-1.jpg is a human!
# 1/1 [==============================] - 0s 12ms/step
# [1.]
# human-2.jpg is a human!
# 1/1 [==============================] - 0s 11ms/step
# [1.]
# human-3.jpg is a human!
# 1/1 [==============================] - 0s 11ms/step
# [0.]
# horse-1.jpg is a horse!
# 1/1 [==============================] - 0s 11ms/step
# [1.]
# horse-2.jpg is a human!
# 1/1 [==============================] - 0s 13ms/step
# [0.]
# horse-3.jpg is a horse!
