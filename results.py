
#import sk_learn
import csv
import numpy as np
import matplotlib.pyplot as plt

def plotError(X_train, y_train, X_test, y_test, clfs):
    Ein = []
    Eout = []

    for clf in clfs:
        clf.fit(X_train, y_train)
        Ein.append(1 - clf.score(X_train))
        Eout.append(1 - clf.score(X_test))

    plt.xlabel('Error')
    plt.title('Error vs. Model')
    plt.plot(Ein, 'o-', label='Training Error')
    plt.plot(Eout, 'o-', label='Test Error')


def getInputData(filename):
    reader = csv.DictReader(open("train_2008.csv"))
    outputLabel = "pes1"
    inputLabels = reader.fieldnames
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
    y = data[:, headers[outputLabel]]
    X = data[:, [headers[h] for h in inputLabels]]
    print(X)
    return X, y

X, y = getInputData("train_2008.csv")
#test_raw  = getData("test_2008.csv")
