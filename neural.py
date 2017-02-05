import csv
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

from sklearn.ensemble import RandomForestClassifier

def getData(dtype="train"):
    filename = "train_2008.csv"
    if dtype == "test":
        filename = "test_2008.csv"
    outputLabel = "PES1"
    inputLabels = ["HUFINAL", "HUSPNISH", "HETENURE", "HETELHHD", "HUFAMINC", "HRNUMHOU", "HRHTYPE", "HUBUS", "GEREG", "GESTCEN", "GTCBSAST", "GTCBSASZ", "PEAGE", "PEMARITL", "PESEX", "PEAFEVER", "PEEDUCA", "PTDTRACE", "PEHSPNON", "PRPERTYP", "PRCITSHP", "PRINUSYR", "PEMJNUM", "PEHRUSL1", "PRDTIND1"]
    headers = {}
    with open(filename, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            if i == 0:
                for j, s in enumerate(line):
                    headers[s] = j
            else:
                break
    data = np.loadtxt(filename, delimiter=",", skiprows=1, dtype=int)
    X = data[:, [headers[h] for h in inputLabels]]
    if dtype == "train":
        y = data[:, headers[outputLabel]]
        return X, y
    else:
        return X

def putData(y):
    with open('output.csv', 'w') as f:
        f.write('id,PES1\n')
        for i, val in enumerate(y):
            f.write(str(i) + "," + str(val) + "\n")

X, y = getData(dtype="train")
y -= 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

y_train = np_utils.to_categorical(y_train, nb_classes=2)
y_test = np_utils.to_categorical(y_test, nb_classes=2)

model = Sequential()

model.add(Dense(100, input_dim = len(X_train[0])))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Activation('softmax'))


model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

fit = model.fit(X_train, y_train, batch_size=256, nb_epoch=100, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

X = getData(dtype="test")
y = model.predict_on_batch(X)
y += 1
y = [np.argmax(a) + 1 for a in y]
putData(y)
