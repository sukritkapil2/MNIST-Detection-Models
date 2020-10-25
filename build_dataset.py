# Import Libraries
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical


# Preprocesses the images
def int2float_grey(x):
    x = x / 255
    return x

# Builds the dataset locally
def build_dataset():
    # Any results you write to the current directory are saved as output.
    with open("./Dataset/mnist_train.csv") as file:
        data_train = pd.read_csv(file)

    # y_train has the label, x_train has the image data
    y_train = np.array(data_train.iloc[:, 0])
    x_train = np.array(data_train.iloc[:, 1:])

    with open("./Dataset/mnist_test.csv") as file:
        data_test = pd.read_csv(file)

    # y_test has the label, x_test has the image data
    y_test = np.array(data_test.iloc[:, 0])
    x_test = np.array(data_test.iloc[:, 1:])

    x_train = int2float_grey(x_train)
    x_test = int2float_grey(x_test)

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    return x_train, y_train, x_test, y_test

# Builds the dataset on Google Colab
def colab_fetch_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = int2float_grey(x_train)
    x_test = int2float_grey(x_test)

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    y_train = to_categorical(y_train)

    return x_train, y_train, x_test, y_test
