import csv
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

from sklearn.ensemble import RandomForestClassifier

def getData(dtype="train"):
    filename = "train_2008.csv"
    if dtype == "test":
        filename = "test_2008.csv"
    outputLabel = "PES1"
    inputLabels = ["PXRACE1", "PTDTRACE", "PELKFTO", "PEDISOUT", "PEMOMTYP", "PELKLL1O", "PEIO1ICD", "PXMS123", "PECOHAB", "PRMARSTA", "PRDTIND1", "PUDIS", "PEAGE", "PRCOWPG", "PEDISREM", "PUDWCK4", "PEHRAVL", "PULAY", "PULINENO", "PELKAVL", "PXLKLL2O", "PUBUSCK3", "PXLKFTO", "PELNDAD", "PEMNTVTY", "PRFAMNUM", "PUWK", "PRIMIND1", "HETELAVL", "PEMLR", "PXDWLKO", "PRDTOCC1", "PEGR6COR", "PULK", "PXSCHLVL", "PRMJOCC1", "PEIO1COW", "PEDIPGED", "PXHRAVL", "PRMJIND1", "PRCOW1", "PXCOHAB", "PEDWWNTO", "PXLKAVL", "PXLKLL1O", "HUFAMINC", "PERRP", "PXLKM1", "PEHSPNON", "HRHHID2", "HETELHHD", "PRMJOCGR", "PXLKLWO", "PEMARITL", "PUNLFCK1", "PXPARENT", "PXLKDUR", "PRDTCOW1", "PXLNDAD", "PEPARENT", "PXGRPROF", "PEDISDRS", "PRFAMREL", "PREXPLF", "PUDIS1", "PEEDUCA", "PUCHINHH", "PEFNTVTY", "PXSCHENR", "PXGR6COR", "HEHOUSUT", "OCCURNUM", "PELKLWO", "PEMS123", "HRHTYPE", "PRWNTJOB", "PXHGCOMP", "PELNMOM", "PXMOMTYP", "PXSCHFT", "PRFAMTYP", "HETENURE", "PREMPNOT", "PEGRPROF", "PRDTHSP", "PESCHENR", "PXLNMOM", "PRNLFSCH", "PXRET1", "PXDADTYP", "PRWKSTAT", "PRWKSCH"]
    #inputLabels = ["HUFINAL", "HUSPNISH", "HETENURE", "HETELHHD", "HUFAMINC", "HRNUMHOU", "HRHTYPE", "HUBUS", "GEREG", "GESTCEN", "GTCBSAST", "GTCBSASZ", "PEAGE", "PEMARITL", "PESEX", "PEAFEVER", "PEEDUCA", "PTDTRACE", "PEHSPNON", "PRPERTYP", "PRCITSHP", "PRINUSYR", "PEMJNUM", "PEHRUSL1", "PRDTIND1", "PEIO1COW", "PRDTOCC1", "PRMJIND1", "PRMJOCC1", "PRMJOCGR", "PRNAGPWS", "PEERNUOT", "PUERNH1C", "PEERNLAB", "PENLFJH", "PENLFACT"]
    headers = {}
    with open(filename, "r") as f:
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
    with open('output_neural.csv', 'w') as f:
        f.write('id,PES1\n')
        for i, val in enumerate(y):
            f.write(str(i) + "," + str(val) + "\n")

X, y = getData(dtype="train")
y -= 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

y_train = np_utils.to_categorical(y_train, nb_classes=2)
y_test = np_utils.to_categorical(y_test, nb_classes=2)

model = Sequential()

model.add(Dense(50, input_dim = len(X_train[0])))
model.add(Activation('sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation('softmax'))


model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

fit = model.fit(X_train, y_train, batch_size=200, nb_epoch=100, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

X = getData(dtype="test")
y = model.predict_on_batch(X)
y += 1
y = [np.argmax(a) + 1 for a in y]
putData(y)
