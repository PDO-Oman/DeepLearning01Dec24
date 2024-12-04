import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load and preprocess the IMDB dataset
max_features = 10000  # Vocabulary size (top 10,000 most frequent words)
maxlen = 200  # Maximum length of a sequence (pad/truncate to this length)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure all inputs are the same length
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# 2. Build the RNN model
model = models.Sequential([
    layers.Embedding(input_dim=max_features, output_dim=128, input_length=maxlen),  # Embedding layer
    layers.SimpleRNN(64, activation='tanh'),  # RNN layer
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train the model
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2  # Use 20% of the training data for validation
)

# 4. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# 5. Predict on new data
sample_review = x_test[0].reshape(1, -1)  # Take a single test review
prediction = model.predict(sample_review)
print(f"Predicted Sentiment: {'Positive' if prediction > 0.5 else 'Negative'}")
