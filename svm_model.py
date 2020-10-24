from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from sklearn import metrics

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  train_size=0.3,
                                                  random_state=1)
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
print(x_train.shape)
print(x_test.shape)
# for gaussian kernel
model_SVC = SVC(kernel='rbf')
# for polynomial kernel
model_SVC = SVC(kernel='poly')
model_SVC.fit(x_train, y_train)

y_pred = model_SVC.predict(x_test)
# accuracy
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
