# 01. Import TF Package
import tensorflow as tf
from tensorflow.keras import layers, models

# 02. Load Sample Dataset
(X_train,Y_train),(X_test,Y_test) =tf.keras.datasets.mnist.load_data()

# 03. Normalize the data(scale pixel values to 0-1 range)
X_train = X_train / 255.0 
X_test = X_test / 255.0 

# 04. Reshape data to add channel dimension(28X28X1 for grayscale image)
X_train = X_train.reshape(-1, 28,28,1) 
X_test = X_test.reshape(-1, 28,28,1) 

# 05. Build the NN Model
model = models.Sequential([
    ########### HIDDEN LAYERS ##############
    # 05.1 Create  Conv2D Layer
    layers.Conv2D(
        32, (3, 3), 
        activation='relu',
        input_shape=(28, 28, 1)),
    
    # 05.2 Create MaxPooling2D Layer 
    layers.MaxPooling2D((2,2)),

    # 05.3 
    layers.Conv2D(
        64, (3, 3), 
        activation='relu'),

    # 05.4 Create Max Pooling 2D Layer
    layers.MaxPooling2D((2, 2)),
    
    # 05.5 Create Flatten Layer
    layers.Flatten(),

    # 05.6 Create Dense with ReLU Activation
    layers.Dense(64, activation='relu'),

    # 05.7 Create Dense with Softmax Actication
    layers.Dense(10, activation='softmax')  #Output Layer for 10 classes
])


# 06. Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 07. Train the model
model.fit(
    X_train,
    Y_train,
    epochs=5, 
    batch_size=32)

# 08. Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# 09. Make Prediction
predictions  = model.predict(X_test[:5])

# 10. Validate
print("Predicted Labels: ", tf.argmax(predictions,axis=1).numpy())
print("True Labels: ", Y_test[:5])