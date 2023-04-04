'''
This is a basic outline of how the convolutional
neural network to detect the gravitational lenses
from COSMOS2020 catalog will look.

Ezra Huscher
April 2023
'''

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import tensorflow as tf

# Parameters of the neural network

data_dir = '/home/dobby/LensingProject/training_set/'
batch_size = 20
img_height = 100
img_width = 100
num_classes = 2   # Is a gravitational lense or does not (binary)
epochs = 10

# What if we make the third dimension 2 and feed in both blue and red COSMOS2020 filters?
input_shape = (img_width, img_height, 1)

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


# Plot initial training set
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Load the data and split it between train and test sets
# We have 20 for sure lenses, and 47 likely lenses.
# Maybe train with 15 sures and 35 likelies?
# Then, test with the rest.
(x_train, y_train), (x_test, y_test) = train_ds, val_ds

class_names = ['Lense', 'No lense']


# Remember to batch, shuffle, and configure the training, validation, and test sets for performance:
#train_ds = configure_for_performance(train_ds)
#val_ds = configure_for_performance(val_ds)
#test_ds = configure_for_performance(test_ds)


# Scale images to the [0, 1] range (normalize)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Output some things
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ----------------------------------
# What layers to use?
# Dropout = to prevent overfitting...
# Dense = fully connected layers...
# Pooling = downsamples by dividing the input into rectangular regions, then computing the max of each region

# Build the model.
# kernel size:  height and width of the 2D convolution window .. was kernel_size=(3, 3)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# Show some parameters (weights and biases) in each layer and also the total parameters
model.summary()

# Compile using an optimization algorithm
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# How did we do?
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])