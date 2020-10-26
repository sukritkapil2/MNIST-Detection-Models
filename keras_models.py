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

# Dense Layers - 1, Activation - softmax
def model_1():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    return model

# Dense Layers - 1, Activation - sigmoid
def model_2():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid'))

    return model

# Dense Layers - 2, Activation - relu, sigmoid
def model_3():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))

    return model

# Dense Layers - 2, Activation - tanh, sigmoid
def model_4():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(10, activation='sigmoid'))

    return model

# Dense Layers - 3, Activation - tanh, tanh, sigmoid
def model_5():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(15, activation='tanh'))
    model.add(Dense(10, activation='sigmoid'))

    return model

# Dense Layers - 3, Activation - tanh, tanh, softmax
def model_6():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    return model

# Dense Layers - 3, Activation - relu, relu, softmax
def model_7():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

# Dense Layers - 3, Activation - relu, relu, softmax
def model_8():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

# Dense Layers - 3, Activation - relu, relu, softmax
def model_9():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model
# Dense Layers - 3, Activation - relu, relu, softmax
def model_10():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

    # Dense Layers - 3, Activation - relu, relu, softmax
def model_11():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(10, activation='softmax'))                 

    return model

    # Dense Layers - 3, Activation - relu, relu, softmax
def model_12():

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model


def test_model(model):
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    acc = np.sum(y_pred == y_test) / np.size(y_pred)
    return acc


def compile_and_train_model(model, num):

    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])
    datagen = ImageDataGenerator(zoom_range=0.1, height_shift_range=0.1, width_shift_range=0.1, rotation_range=10)

    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16), steps_per_epoch=500, epochs=15,
        verbose=2, validation_data=(x_val, y_val), callbacks=[annealer])

    train_loss, train_acc = model.evaluate(x_val, y_val, verbose=0)

    plt.figure()
    plt.plot(hist.history['loss'], color='b', label = 'Loss')
    plt.plot(hist.history['val_loss'], color='r', label = 'Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'model_loss_{num}.png')
    plt.close()

    plt.figure()
    plt.plot(hist.history['accuracy'], color='b', label = 'Accuracy')
    plt.plot(hist.history['val_accuracy'], color='r', label = 'Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'model_acc_{num}.png')
    plt.close()

    test_acc = test_model(model)

    return {'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc}
