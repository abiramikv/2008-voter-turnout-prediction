import csv
import keras
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def getData(dtype="train"):
    filename = "train_2008.csv"
    if dtype == "test":
        filename = "test_2008.csv"
    outputLabel = "PES1"
    inputLabels = ["HUFINAL", "HUSPNISH", "HETENURE", "HETELHHD", "HUFAMINC", "HRNUMHOU", "HRHTYPE", "HUBUS", "GEREG", "GESTCEN", "GTCBSAST", "GTCBSASZ", "PEAGE", "PEMARITL", "PESEX", "PEAFEVER", "PEEDUCA", "PTDTRACE", "PEHSPNON", "PRPERTYP", "PRCITSHP", "PRINUSYR", "PEMJNUM", "PEHRUSL1", "PRDTIND1", "PEIO1COW", "PRDTOCC1", "PRMJIND1", "PRMJOCC1", "PRMJOCGR", "PRNAGPWS", "PEERNUOT", "PUERNH1C", "PEERNLAB", "PENLFJH", "PENLFACT"]
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
    with open('output_sklearn.csv', 'w') as f:
        f.write('id,PES1\n')
        for i, val in enumerate(y):
            f.write(str(i) + "," + str(val) + "\n")

X, y = getData(dtype="train")

clf = RandomForestClassifier(n_estimators=20000, verbose=1)
#clf = AdaBoostClassifier(n_estimators=1000)
scores = cross_val_score(clf, X, y, cv=5, verbose=1)
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
clf.fit(X, y)

X = getData(dtype="test")
y = clf.predict(X)
putData(y)
