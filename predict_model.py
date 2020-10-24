from pandas import np
from tensorflow.keras.models import load_model
from build_dataset import build_dataset

model = load_model('model1.h5')

x_train, y_train, x_test, y_test, x_val, y_val = build_dataset()

print("Making Predictions")

y_pred = np.argmax(model.predict(x_test), axis=-1)
acc = np.sum(y_pred == y_test) / np.size(y_pred)
print("[RESULT] TEST ACCURACY = {}".format(acc))
