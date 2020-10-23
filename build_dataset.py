# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.datasets import mnist

size_img = 28
threshold_color = 100 / 255


def int2float_grey(x):
    x = x / 255
    return x


def build_dataset():
    # Any results you write to the current directory are saved as output.
    with open("./Dataset/mnist_train.csv") as file:
        data_train = pd.read_csv(file)

        # y_train has the label, x_train has the image data
        y_train = np.array(data_train.iloc[:, 0])
        x_train = np.array(data_train.iloc[:, 1:])

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          train_size=0.8, test_size=0.2,
                                                          random_state=1)

    with open("./Dataset/mnist_test.csv") as file:
        data_test = pd.read_csv(file)

        # y_test has the label, x_test has the image data
        y_test = np.array(data_test.iloc[:, 0])
        x_test = np.array(data_test.iloc[:, 1:])

        n_features_train = x_train.shape[1]
        n_samples_train = x_train.shape[0]
        n_samples_test = x_test.shape[0]

        x_train = int2float_grey(x_train)
        x_test = int2float_grey(x_test)

        x_train = np.reshape(x_train, (-1, 28, 28, 1))
        x_test = np.reshape(x_test, (-1, 28, 28, 1))
        x_val = np.reshape(x_val, (-1, 28, 28, 1))

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_val = to_categorical(y_val)

        print(f"Features : {n_features_train}")
        print(f"Number of training samples : {n_samples_train}")
        print(f"Number of Testing Samples : {n_samples_test}")
        print(f"x_train shape : {x_train.shape}")
        print(f"y_train shape : {y_train.shape}")
        print(f"x_test shape : {x_test.shape}")
        print(f"y_test shape : {y_test.shape}")

    return x_train, y_train, x_test, y_test, x_val, y_val


def colab_fetch_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=1)

    n_features_train = x_train.shape[1]
    n_samples_train = x_train.shape[0]
    n_samples_test = x_test.shape[0]

    x_train = int2float_grey(x_train)
    x_test = int2float_grey(x_test)

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    x_val = np.reshape(x_val, (-1, 28, 28, 1))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    print(f"Features : {n_features_train}")
    print(f"Number of training samples : {n_samples_train}")
    print(f"Number of Testing Samples : {n_samples_test}")
    print(f"x_train shape : {x_train.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"x_test shape : {x_test.shape}")
    print(f"y_test shape : {y_test.shape}")

    return x_train, y_train, x_test, y_test, x_val, y_val
