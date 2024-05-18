import cv2 as cv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()

training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Define your neural network architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='leaky_relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='leaky_relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='leaky_relu'),
    layers.BatchNormalization(),

    # Add a residual block (experiment with stacking these)
    layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='leaky_relu'),
    # layers.add([models.layers[0].output, models.layers[-1].output]),  # Add residual connectio

    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='leaky_relu'),
    layers.Dropout(0.2),
    # Adjust for your number of classe
    layers.Dense(100, activation='softmax')

    # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.Flatten(),
    # layers.Dense(64, activation='relu'),
    # layers.Dense(10, activation='softmax')

])

# Compile the model
model.compile(optimizer= RMSprop(learning_rate = 0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=30, validation_data=(testing_images, testing_labels))

# Now, you can use the trained model to make predictions
img = cv.imread('dog.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32)) / 255.0  # Resize and normalize the image

plt.imshow(img, cmap=plt.cm.binary)
plt.show()

prediction = model.predict(np.array([img]))
index = np.argmax(prediction)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trucks']
# print(f'Prediction: {class_names[index]}')
# index = np.argmax(prediction)
print(f'prediction is {class_names[index]}')

model.save('img_pred.keras')
print("file saved successfully")
