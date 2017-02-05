#import sk_learn
import csv
import numpy as np

def getInputData(filename):
    outputLabel = "pes1"
    inputLabels = ["id"]
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
