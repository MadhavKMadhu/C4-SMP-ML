# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))

class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']

#plt.figure()
#plt.imshow(X_train[1])
#plt.colorbar()
#plt.grid(False)
#plt.show()

''' Data (both test and train) being Pre-Processed before being trained '''
X_train = X_train / 255.0
X_test = X_test / 255.0

''' Displaying first 25 images from the dataset '''
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(X_train[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[Y_train[i]])
# plt.show()

''' Setting up the layers '''
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation='relu'),     # Rectified Linear Unit Activaiton Fn
    keras.layers.Dense(10)
])

# Model compilation
model.compile(optimizer='adam',
loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
metrics=['accuracy']
)

# Train the Model
model.fit(X_train, Y_train, epochs=10)

# Evaluate Accuracy
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose = 2)
print('\nTest Accuracy: ', test_acc)

# Make Predicitions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X_test)

# print(np.argmax(predictions[0]))

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color = color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Verifying predictions of 12th image
i = 10
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], Y_test, X_test)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], Y_test)
plt.show()

# Plotting several images alongwith their predictions
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2*i + 1)
    plot_image(i, predictions[i], Y_test, X_test)
    plt.subplot(num_rows, 2*num_cols, 2*i + 2)
    plot_value_array(i, predictions[i], Y_test)
plt.tight_layout()
plt.show()

# Importing an image that isn't part of the dataset and using it to test the model
import requests
from PIL import Image

url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream = True)
img_new = Image.open(response.raw)
plt.imshow(img_new)
plt.show()

# Resizing and converting to grayscale
import cv2

img_array = np.asarray(img_new)
resized = cv2.resize(img_array, (28, 28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray_scale)

plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()
print(image)

# Feeding it to model
image = image / 255
image = image.reshape(1, 784)
prediction = model.predict_classes(image)
print("Predicted Digit:", str(prediction))