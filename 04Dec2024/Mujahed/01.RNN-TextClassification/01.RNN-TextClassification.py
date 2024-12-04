# 01. Load All Packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import numpy as np


# 02.Load Dataset with X and Y
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Preprocess the data: Pad sequences to ensure uniform input size
X_train = pad_sequences(X_train, maxlen=500)  # Pad sequences to length of 500
X_test = pad_sequences(X_test, maxlen=500)

# Build the RNN model
model = Sequential()

# Embedding layer to represent words as dense vectors
model.add(Embedding(input_dim=10000, output_dim=128, input_length=500))

# Add a simple RNN layer
model.add( SimpleRNN(128, return_sequences=False))

# Add a Dropout layer to prevent overfitting
model.add(Dropout(0.5))

# Add a Dense output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))


# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Optionally, plot the training history (accuracy and loss)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
