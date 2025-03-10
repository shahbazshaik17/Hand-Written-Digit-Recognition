# Handwritten Digit Recognition using TensorFlow

## Overview

This project trains a neural network using the MNIST dataset to recognize handwritten digits. The trained model is then used to predict custom handwritten digits from image files.

## Execution Steps

### Train the Model

The script first loads the MNIST dataset, normalizes it, and trains a neural network.

To train the model, run:

python Train.py

This will:

Train a neural network with two hidden layers.

Evaluate the model on test data.

Save the trained model as handwritten_digits.keras.

### Predict Custom Images

Ensure you have custom images named in the format digit/digitX.png (where X is a number) before running the prediction script.

To use the trained model for predictions, execute:

python Predict.py

This will:

Load the trained model.

Read images from the digit/ directory.

Process and predict the handwritten digit.

Display the image along with the prediction.

## Troubleshooting

Ensure the image files exist in the specified digit/ folder.

Images should be grayscale and properly formatted for better recognition.

If the script fails to read an image, it will print an error and move to the next one.
