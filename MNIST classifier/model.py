import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
train_size = int(len(x_train)* 0.7)
train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).batch(20)

model = Sequential()
model.add(Flatten(input_shape = (28,28)))#2차원을 1차원으로 바꾸어줌
model.add(Dense(20, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
hist = model.fit(train_ds, validation_data = val_ds, epochs = 10)

model.evaluate(x_test, y_test)

model.summary()

model.save('mnist_model.h5')
