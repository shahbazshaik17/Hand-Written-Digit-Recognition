import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('handwritten_digits.keras')
# Load custom images and predict them
image_number = 9
while os.path.isfile('digit/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digit/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
    finally:    
        image_number += 1