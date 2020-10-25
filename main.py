import keras_models
import trad_models
import numpy as np
import matplotlib.pyplot as plt

print('Welcome to Selective Topics Assignment')

model_info = {}

# # Building 12 models
models = [keras_models.model_dense_1, keras_models.model_dense_2, keras_models.model_dense_3, keras_models.model_dense_4,
    keras_models.cnn_model_5, keras_models.cnn_model_6, keras_models.cnn_model_7, keras_models.cnn_model_8,
    keras_models.cnn_model_9, keras_models.cnn_model_10, keras_models.cnn_model_11, keras_models.cnn_model_12]

for func in models:
    model = func()
    name = str(func.__name__)
    print(f'TRAINING MODEL : {name}')
    model_info[name] = keras_models.compile_and_train_model(model)['test_acc']

# Building Traditional Models
models = [trad_models.model_poly_SVM, trad_models.model_rbf_SVM, trad_models.model_DT, trad_models.model_RF]

for func in models:
    model = func()
    name = str(func.__name__)
    print(f'TRAINING MODEL : {name}')
    model_info[name] = trad_models.model_score(model)

labels = list(model_info.keys())
values = list(model_info.values())

plt.figure()
plt.bar(np.arange(len(labels)), values, align='center')
plt.xticks(np.arange(len(labels)), labels, rotation='vertical')
plt.yticks(np.linspace(0, 1, 21))
plt.savefig('model_comparison.png')
plt.close()
