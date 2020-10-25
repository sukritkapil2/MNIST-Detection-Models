import matplotlib.pyplot as plt

import numpy as np
from build_dataset import build_dataset
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical

x_train, y_train, x_test, y_test = build_dataset()
y_train = to_categorical(y_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2)

# Dense Layers - 1, Convolutional Layers - 0, Activation - softmax
def model_dense_1():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    return model


# Dense Layers - 2, Convolutional Layers - 0, Activation - relu, softmax
def model_dense_2():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

# Dense Layers - 3, Convolutional Layers - 0, Activation - relu, softmax
def model_dense_3():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

# Dense Layers - 3, Convolutional Layers - 0, Activation - relu, tanh, softmax
def model_dense_4():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    return model

# Dense Layers - 2, Convolutional Layers - 1, Activation - relu, softmax
def cnn_model_5():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='softmax')) 

    return model

# Dense Layers - 2, Convolutional Layers - 1, Activation - relu, tanh, softmax
def cnn_model_6():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(10, activation='softmax')) 

    return model

# Dense Layers - 2, Convolutional Layers - 2, Activation - (relu, softmax), MaxPooling2D
def cnn_model_7():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax')) 

    return model

# Dense Layers - 3, Convolutional Layers - 2, Activation - (relu, softmax), MaxPooling2D
def cnn_model_8():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax')) 

    return model

# Dense Layers - 3, Convolutional Layers - 2, Activation - (tanh, softmax), MaxPooling2D
def cnn_model_9():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='tanh'))
    model.add(Conv2D(64, (3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(10, activation='softmax')) 

    return model

# Dense Layers - 3, Convolutional Layers - 3, Activation - (relu, softmax), Dropout
def cnn_model_10():

    model = Sequential()
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', strides=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    return model

# Dense Layers - 3, Convolutional Layers - 4, Activation - (relu, softmax), MaxPooling2D, Dropout
def cnn_model_11():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax')) 

    return model

# Dense Layers - 2, Convolutional Layers - 6, Activation - (relu, softmax), MaxPooling2D, Dropout
def cnn_model_12():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def test_model(model):
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    acc = np.sum(y_pred == y_test) / np.size(y_pred)
    return acc


def compile_and_train_model(model):

    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])
    datagen = ImageDataGenerator(zoom_range=0.1, height_shift_range=0.1, width_shift_range=0.1, rotation_range=10)
    
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16), steps_per_epoch=500, epochs=15,
        verbose=2, validation_data=(x_val, y_val), callbacks=[annealer])
    
    train_loss, train_acc = model.evaluate(x_val, y_val, verbose=0)

    plt.figure()
    plt.plot(hist.history['loss'], color='b', label = 'Loss')
    plt.plot(hist.history['val_loss'], color='r', label = 'Validation Loss')
    plt.legend()
    plt.close()
    
    plt.figure()
    plt.plot(hist.history['accuracy'], color='b', label = 'Accuracy')
    plt.plot(hist.history['val_accuracy'], color='r', label = 'Validation Accuracy')
    plt.legend()
    plt.close()

    test_acc = test_model(model)

    return {'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc}
