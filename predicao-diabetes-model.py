import numpy as np
from keras.models import load_model

dataset = np.loadtxt('data_diabetes.csv', delimiter=',', skiprows=1)

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

dataset[dataset[:, 0] == 2, 0] = 1

index_20percent = int(0.2 * len(dataset[:, 0]))
print(index_20percent)

XTRAIN = dataset[index_20percent:, 1:]

mean = XTRAIN.mean(axis=0)
XTRAIN -= mean
std = XTRAIN.std(axis=0)
XTRAIN /= std

model = load_model('modelo_diabetes_salvo.hdf5')

TESTEARRAY = np.array(
    [0.0,0.0,0.0,25.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,3.0,0.0,0.0,0.0,0.0,7.0,6.0,1.0])
TESTEARRAY -= mean
TESTEARRAY /= std
prediction = model.predict(TESTEARRAY[None])
print(prediction.round())
