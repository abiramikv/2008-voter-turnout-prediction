import csv
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

filename = "train_2008.csv"
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


y = data[:, headers[outputLabel]]
y_r = y - 2
ratio = float(np.count_nonzero(y_r)) / float(len(y_r))
print(ratio)

res = []
N = float(len(headers))
for i, h in enumerate(headers):
    col = data[:, [headers[h]]]

    clf = RandomForestClassifier(n_estimators=10)
    scores = cross_val_score(clf, col, y, cv=5, verbose=1)
    print h, scores.mean(), scores.std(), float(i)/N
    res.append((scores.mean(), h))
res.sort(reverse=True)
print res
selected = []
for (r, x) in res:
    if r > ratio:
        selected.append(x)
print '", "'.join(selected)
