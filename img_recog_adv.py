# Import libraries
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator # For data augmentation

# Load CIFAR-100 dataset (modify if using a different dataset)
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()

# Data normalization (typical for image data)
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Data augmentation (optional, but recommended)
datagen = ImageDataGenerator(
    shear_range=0.2,  # Randomly shear images
    zoom_range=0.2,   # Randomly zoom images
    horizontal_flip=True)  # Randomly flip images horizontally

training_generator = datagen.flow(training_images, training_labels, batch_size=32)
validation_generator = datagen.flow(testing_images, testing_labels, batch_size=32)

# Define a more complex neural network architecture
model = models.Sequential([
    # Convolutional layers with ReLU activation
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),  # Increased number of filters

    # Dropout layer to prevent overfitting (optional, but common)
    layers.Dropout(0.2),  # Randomly drop 20% of neurons during training

    # Flatten layer to prepare for dense layers
    layers.Flatten(),

    # Dense layers with ReLU activation
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),  # Dropout again

    # Output layer with softmax activation for 100 CIFAR-100 classes
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
model.fit(training_generator, epochs=50, validation_data=validation_generator)  # Use validation generator

# Now you can use the trained model to make predictions
img = cv.imread('dog.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32)) / 255.0  # Resize and normalize the image

plt.imshow(img, cmap=plt.cm.binary)
plt.show()

prediction = model.predict(np.array([img]))
index = np.argmax(prediction)

# Assuming you have loaded the CIFAR-100 class names (replace with your method)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trucks']

print(f'Predicted class: {class_names[index]}')
