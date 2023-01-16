# Author: M Farhan Hasib

# Modules
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import hyperparameters

# Loads the MNIST dataset
mnist = tf.keras.datasets.mnist

# Splits the dataset into training and testing
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Concatenates the training and testing data
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

# Splits the data further into training, validation and testing
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1-hyperparameters.train_ratio))
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=((hyperparameters.test_ratio)/(hyperparameters.validatin_ratio+hyperparameters.test_ratio)))

# Normalizes the data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Reshapes the data
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# Creates the model with 1 input layer, 3 hidden layers and 1 output layer with custom hyperparameters
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=hyperparameters.hidden_layer_1, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=hyperparameters.hidden_layer_2, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=hyperparameters.hidden_layer_3, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=hyperparameters.OUTPUT_LAYER, activation=tf.nn.softmax))

# Compiles the model using Adam optimizer and sparse categorical crossentropy loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trains the model with custom epochs
model.fit(X_train, y_train, epochs=hyperparameters.epochs)

# Evaluates the model
val_loss, val_acc = model.evaluate(X_test, y_test)

print("Loss: ", val_loss)
print("Accuracy: ", val_acc)
    
# Saves the model
model.save('trained_model.model')