
#import sk_learn
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


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

# one-hot-encoding for categorical variables
def categorize(x):
    enc = OneHotEncoder()
    return enc.fit_transform(x).toarray()[:,1:]

X = np.array([[-1, 0, 3],
     [-3, 1, -1],
     [2, 3, 4],
     [1, 2, 4]])
X_t = np.array([]).reshape(np.shape(X)[0],0)

for i in range(np.shape(X)[1]):
    X_t = np.hstack((X_t, splitFeature(X[:, i], 'categorical')))
    print(X_t)
