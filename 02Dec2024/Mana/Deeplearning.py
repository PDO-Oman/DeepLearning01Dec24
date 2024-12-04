import tensorflow as tf
from tensorflow.keras import layers,models


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Normailize
x_train=x_train /255.0
x_test=x_test /255.0

#reshape
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

#Build Neural Netrork model 
model=models.Sequential([
    layers.Conv2D(
        32,(3,3),
        activation='relu',
        input_shape=(28,28,1)),
    
    layers.MaxPooling2D((2,2,)),

    layers.Conv2D(
        64,(3,3),
        activation='relu'),

    layers.MaxPooling2D((2,2,)),
    layers.Flatten(),

    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics= ['accuracy']
)

model.fit(x_train,y_train,epochs=5,batch_size=32)

test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)
print (f'Test accurecy : {test_acc:.2f}')

prediction=model.predict(x_test[:5])

print("Predicted Lable : ", tf.argmax(prediction,axis=1).numpy())
print("true Lable : ", y_test[:5])