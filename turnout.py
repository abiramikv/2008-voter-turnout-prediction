import csv
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def getData(dtype="train"):
    filename = "train_2008.csv"
    if dtype == "test":
        filename = "test_2008.csv"
    outputLabel = "pes1"
    inputLabels = ["pespouse", "pesex", "peeduca", "hufaminc", "huspnish", "hetenure", "hehousut", "hrnumhou", "hrhtype", "gtmetsta", "peage"]
    headers = {}
    with open(filename, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            if i == 0:
                for j, s in enumerate(line):
                    headers[s.lower()] = j
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
#X = getData(dtype="test")

clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=5, verbose=1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf.fit(X, y)

print(clf.score(X, y))

X = getData(dtype="test")
y = clf.predict(X)
putData(y)

#test_raw  = getData("test_2008.csv")
