import csv
import random
import sys
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as pl


# # one-hot-encoding for categorical variables
# def categorize(x):
#     enc = OneHotEncoder()
#     return enc.fit_transform(x).toarray()[:,1:]
#
# # assumes input of single feature column, outputs
# def splitFeature(x1, group_type):
#     x2 = np.zeros(len(x1))
#     for i,x in enumerate(x1):
#         if x < 0:
#             x2[i] = abs(x)
#             x1[i] = 0
#
#     x1 = np.vstack(x1)
#     x2 = np.vstack(x2)
#
#     if group_type == 'categorical':
#         x1 = categorize(x1)
#
#     if np.any(x2):
#         x2 = categorize(x2)
#         return np.hstack((x1, x2))
#
#     return x1
#
# cat = ["HUFINAL", "HETENURE", "HETELHHD", "HRHTYPE", "HUBUS", "GEREG", "GESTCEN", "GTCBSAST", "PEMARITL", "PESEX", "PEEDUCA", "PTDTRACE", "PRPERTYP", "PRCITSHP", "PRDTIND1", "PRCHLD"]
# num = ["HUFAMINC", "HRNUMHOU", "GTCBSASZ", "PEAGE", "PEAFEVER", "PEHSPNON", "PRINUSYR", "PEMJNUM", "PEHRUSL1", "PESCHFT", "PESCHLVL", "PRNMCHLD", "PEDIPGED", "PECYC", "PEGRPROF"]
# inputLabels = cat + num
#
# def getData(dtype="train"):
#     filename = "train_2008.csv"
#     if dtype == "test":
#         filename = "test_2008.csv"
#     outputLabel = "PES1"
#     headers = {}
#     with open(filename, "r") as f:
#         reader = csv.reader(f, delimiter=",")
#         for i, line in enumerate(reader):
#             if i == 0:
#                 for j, s in enumerate(line):
#                     headers[s] = j
#             else:
#                 break
#     data = np.loadtxt(filename, delimiter=",", skiprows=1, dtype=int)
#     X = data[:, [headers[h] for h in inputLabels]]
#
#
#     if dtype == "train":
#         y = data[:, headers[outputLabel]]
#         return X, y
#     else:
#         return X
#
# def putData(y):
#     with open('output_sklearn.csv', 'w') as f:
#         f.write('id,PES1\n')
#         for i, val in enumerate(y):
#             f.write(str(i) + "," + str(val) + "\n")
#
# X_train, y = getData(dtype="train")
# X_test = getData(dtype="test")
#
# n = len(X_train)
# X = np.vstack((X_train, X_test))
#
# X_t = np.array([]).reshape(np.shape(X)[0], 0)
# for i in range(np.shape(X)[1]):
#     if inputLabels[i] in cat:
#         group = 'categorical'
#     else:
#         group = 'numeric'
#
#     X_t = np.hstack((X_t, splitFeature(X[:, i], group)))
#
# X = X_t[:n, :]
#



models = [
    ("Neural Network", None),
    ("AdaBoost Classifier", AdaBoostClassifier()),
    ("Bagging Classifier", BaggingClassifier()),
    ("Extra Trees Classifier", ExtraTreesClassifier()),
    ("Gradient Boosting Classifier", GradientBoostingClassifier()),
    ("Random Forest Classifier", RandomForestClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Ridge Classifier", RidgeClassifier()),
    ("SGD Classifier", SGDClassifier()),
    ("Passive Aggressive Classifier", PassiveAggressiveClassifier())
]


means = [0.7651, 0.77308715284220941, 0.76244201150491742, 0.7634564235788952, 0.77476959237953857, 0.78349972165522362, 0.70296282550875244, 0.77579018989299187, 0.67457165831632337, 0.65715964619286193]
std =  [0.0023, 0.002613778195753735, 0.0037886208108983514, 0.0037157763125882917, 0.0024353749169541698, 0.0028210283393290841, 0.00177184502269097, 0.0039221539565891918, 0.11669656665495121, 0.12417093749602467]

# for name, clf in models:
#     print "Training", name
#     accuracies = []
#     for _ in range(10):
#         if name == "Neural Network":
#             continue
#         X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0, 2**31))
#         clf.fit(X_train, y_train)
#         score = clf.score(X_test, y_test)
#         accuracies.append(score)
#     means.append(np.mean(accuracies))
#     std.append(np.std(accuracies))
#
#
# print means, std


objects = [x for x, _ in models]

index = np.arange(len(objects))
bar_width = 0.35

error_config = {'ecolor': '0.3'}

pl.bar(index, means, bar_width, yerr=std, error_kw=error_config)

pl.ylim([0.6, 0.8])
pl.xlabel('Model Used')
pl.ylabel('Testing Accuracy')
pl.title('Accurancy of Various Classifiers for Voting Data')
pl.xticks(index, objects, rotation='vertical')

pl.tight_layout()
pl.savefig("accuracies.png")
pl.clf()
