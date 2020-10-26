import keras_models
import trad_models
import numpy as np
import matplotlib.pyplot as plt

print('Welcome to Selective Topics Assignment')

model_info = {}

# Building 12 models

models = [keras_models.model_1, keras_models.model_2, keras_models.model_3, keras_models.model_4,
    keras_models.model_5, keras_models.model_6, keras_models.model_7, keras_models.model_8,
    keras_models.model_9, keras_models.model_10, keras_models.model_11, keras_models.model_12]

for count, func in enumerate(models):
    model = func()
    name = str(func.__name__).split('_')[-1]
    print(f'TRAINING MODEL : {name}')
    model_info[name] = keras_models.compile_and_train_model(model, count+1)['test_acc']

# Building Traditional Models
models = [trad_models.model_poly, trad_models.model_rbf, trad_models.model_DT, trad_models.model_RF]

for func in models:
    model = func()
    name = str(func.__name__).split('_')[-1]
    print(f'TRAINING MODEL : {name}')
    model_info[name] = trad_models.model_score(model)

labels = np.array(list(model_info.keys()))
values = np.array(list(model_info.values()))

sort_index = np.argsort(values)
labels = labels[sort_index]
values = values[sort_index]

plt.figure()
plt.bar(np.arange(len(labels)), values, align='center')
plt.xticks(np.arange(len(labels)), labels, rotation=40)
plt.yticks(np.linspace(0, 1, 21))
plt.savefig('model_comparison.png')
plt.close()
