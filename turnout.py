#import sk_learn
import csv
import numpy as np

def getData(filename):
    headers = []
    with open(filename, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            if i == 0:
                headers = line
    print headers
    #         print 'line[{}] = {}'.format(i, line)
    # data = np.loadtxt(filename, delimiter=",", skiprows=1, dtype=int)
    # print(data, data[:][-1])
    # return data

train_raw = getData("train_2008.csv")
test_raw  = getData("test_2008.csv")
