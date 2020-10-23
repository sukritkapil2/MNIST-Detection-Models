# Import Keras and Tensorflow
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.python.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from build_dataset import build_dataset

# Import Training, Testing and Validation
x_train, y_train, x_test, y_test, x_val, y_val = build_dataset()


# Define different model architectures
def model1():
    model = Sequential()
    model.add(Conv2D(20, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    # model.add(BatchNormalization())
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', strides=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    return model


def compile_and_train_model(model, number):
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])
    datagen = ImageDataGenerator(zoom_range=0.1,
                                 height_shift_range=0.1,
                                 width_shift_range=0.1,
                                 rotation_range=10)

    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                               steps_per_epoch=500,
                               epochs=30,
                               verbose=2,
                               validation_data=(x_val, y_val),
                               callbacks=[annealer,
                                          ModelCheckpoint(f'model{number}.h5', save_best_only=True)])

    final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
    print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.show()
    plt.plot(hist.history['acc'], color='b')
    plt.plot(hist.history['val_acc'], color='r')
    plt.show()


model = model1()
compile_and_train_model(model, 1)
