import csv
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

from sklearn.preprocessing import OneHotEncoder

# one-hot-encoding for categorical variables
def categorize(x):
    enc = OneHotEncoder()
    return enc.fit_transform(x).toarray()[:,1:]

# assumes input of single feature column, outputs
def splitFeature(x1, group_type):
    x2 = np.zeros(len(x1))
    for i,x in enumerate(x1):
        if x < 0:
            x2[i] = abs(x)
            x1[i] = 0

    x1 = np.vstack(x1)
    x2 = np.vstack(x2)

    if group_type == 'categorical':
        x1 = categorize(x1)

    if np.any(x2):
        x2 = categorize(x2)
        return np.hstack((x1, x2))


    return x1

cat = ["HUFINAL", "HETENURE", "HETELHHD", "HRHTYPE", "HUBUS", "GEREG", "GESTCEN", "GTCBSAST", "PEMARITL", "PESEX", "PEEDUCA", "PTDTRACE", "PRPERTYP", "PRCITSHP", "PRDTIND1", "PRCHLD"]
num = ["HUFAMINC", "HRNUMHOU", "GTCBSASZ", "PEAGE", "PEAFEVER", "PEHSPNON", "PRINUSYR", "PEMJNUM", "PEHRUSL1", "PESCHFT", "PESCHLVL", "PRNMCHLD", "PEDIPGED", "PECYC", "PEGRPROF"]
inputLabels = cat + num

def getData(dtype="train"):
    filename = "train_2008.csv"
    if dtype == "test":
        filename = "test_2008.csv"
    outputLabel = "PES1"
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

X_train, y = getData(dtype="train")
X_test = getData(dtype="test")

n = len(X_train)
X = np.vstack((X_train, X_test))

X_t = np.array([]).reshape(np.shape(X)[0], 0)
for i in range(np.shape(X)[1]):
    if inputLabels[i] in cat:
        group = 'categorical'
    else:
        group = 'numeric'

    X_t = np.hstack((X_t, splitFeature(X[:, i], group)))

X = X_t[:n, :]
#clf = RandomForestClassifier(n_estimators=2000, verbose=1)
#clf = AdaBoostClassifier(n_estimators=100)
clf = Perceptron(n_iter=50)
scores = cross_val_score(clf, X, y, cv=4, verbose=1)
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
clf.fit(X, y)

X = X_t[n:, :]
y = clf.predict(X)
putData(y)
