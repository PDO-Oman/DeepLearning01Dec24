import tensorflow as tf

from tensorflow.keras import layers,models
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train / 255.0  # Normalize to [0, 1]
x_test = x_test / 255.0

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics= ['accuracy']
)

model.fit(x_train,y_train,epochs=10,batch_size=64, validation_data=(x_test, y_test))

test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)
print (f'Test accurecy : {test_acc:.2f}')

prediction=model.predict(x_test[:10])

print("Predicted Lable : ", tf.argmax(prediction,axis=1).numpy())
print("true Lable : ", y_test[:10])