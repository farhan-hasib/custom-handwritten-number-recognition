# Author: M Farhan Hasib

# Modules
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

# Loads the model
model = tf.keras.models.load_model('trained_model.model')

# Loads the images from the digits folder
for images in glob.glob('digits/*'):
    # Loads the image using OpenCV
    image = cv2.imread(images)[:,:,0]
    # Resizes the image to 28x28 pixels
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    # Inverts the image
    image = np.invert(np.array(image))
    # Makes prediction for the image
    prediction = model.predict(image.reshape(1, 28, 28))
    print("Digit Prediction: ", np.argmax(prediction))
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()
