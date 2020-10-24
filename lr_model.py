from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          train_size=0.1,
                                                          random_state=1)

x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
print(x_train.shape)
print(x_test.shape)

model_LR = LogisticRegression(max_iter=500000)
model_LR.fit(x_train, y_train)
print('Training Complete')
score = model_LR.score(x_test, y_test)
print(score)

