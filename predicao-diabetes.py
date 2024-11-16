import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Preparando dataset

dataset = np.loadtxt('data_diabetes.csv', delimiter=',', skiprows=1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
dataset[dataset[:, 0] == 2, 0] = 1

# Dividindo dataset em treinamento e validação

index_20percent = int(0.2 * len(dataset[:, 0]))

XVALIDATION = dataset[:index_20percent, 1:]
YVALIDATION = dataset[:index_20percent, 0]

XTRAIN = dataset[index_20percent:, 1:]
YTRAIN = dataset[index_20percent:, 0]

# Normalizando os dados

mean = XTRAIN.mean(axis=0)
XTRAIN -= mean
std = XTRAIN.std(axis=0)
XTRAIN /= std

XVALIDATION -= mean
XVALIDATION /= std

# Criando um modelo de rede neural

model = Sequential()
model.add(Dense(8, input_dim=len(XTRAIN[0, :]), activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Treinando o modelo

callback_a = ModelCheckpoint(filepath='modelo_diabetes.hdf5', monitor='val_loss', save_best_only=True,
                             save_weights_only=True)
callback_b = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)

history = model.fit(XTRAIN, YTRAIN, validation_data=(XVALIDATION, YVALIDATION), epochs=256, batch_size=400,
                    callbacks=[callback_a, callback_b])

# Verificando as curvas de aprendizado

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Acurácia')
plt.xlabel('Epoca')
plt.legend(['Dados de treinamento', 'Dados de validação'], loc='lower right')
plt.show()

model.load_weights('modelo_diabetes.hdf5')

# Avaliando o modelo de treinamento

scores = model.evaluate(XTRAIN, YTRAIN)
print(model.metrics_names)
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Avaliando o modelo de validação

scores = model.evaluate(XVALIDATION, YVALIDATION)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Comparação das previsões com os valores reais

print(XVALIDATION[0:10])
print(YVALIDATION[0:10])

prediction = model.predict(XVALIDATION)

print(prediction[0:10])

print(prediction[0:10].round())

accuracy = accuracy_score(YVALIDATION, prediction.round())
precision = precision_score(YVALIDATION, prediction.round())
recall = recall_score(YVALIDATION, prediction.round())
f1score = f1_score(YVALIDATION, prediction.round())
print("Accuracy: %.2f%%" % (accuracy * 100.0))
model.save('modelo_diabetes_salvo.hdf5')
