from build_dataset import build_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

x_train, y_train, x_test, y_test = build_dataset()
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# SVM Model 1
def model_poly():
    X_train, _, Y_train, _ = train_test_split(x_train, y_train, train_size=0.3)
    model_SVC = SVC(kernel='poly')
    model_SVC.fit(X_train, Y_train)
    return model_SVC

# SVM Model 2
def model_rbf():
    X_train, _, Y_train, _ = train_test_split(x_train, y_train, train_size=0.3)
    model_SVC = SVC(kernel='rbf')
    model_SVC.fit(X_train, Y_train)
    return model_SVC

# Decision Tree Model
def model_DT():
    model_DT = DecisionTreeClassifier()
    model_DT.fit(x_train, y_train)
    return model_DT

# Random Forest Classifier
def model_RF():
    model_RF = RandomForestClassifier()
    model_RF.fit(x_train, y_train)
    return model_RF

def model_score(model):
    score = model.score(x_test, y_test)
    return score
