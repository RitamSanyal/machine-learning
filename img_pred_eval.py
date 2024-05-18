import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

model = load_model('img_pred.keras')

img = cv.imread('plane_2.png')
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