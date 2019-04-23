import tensorflow as tf
from keras.layers import *
from keras.models import Sequential


def build_model(input_shape, output_len):
    print("Input shape", input_shape)
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(output_len, activation=tf.nn.softmax))

    return model


def train(x, y, model, epochs=10):
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=x, y=y, epochs=epochs)

    return model


def evaluate(model, x, y):
    loss, acc = model.evaluate(x, y)

    return loss, acc


def predict(model, x):
    prediction = model.predict(x)
    # predictions = prediction.argmax(axis=1)
    # print((predictions == y).mean())
    return prediction.argmax(axis=1)


if __name__ == '__main__':
    pass
