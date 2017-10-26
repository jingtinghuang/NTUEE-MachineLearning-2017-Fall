import numpy as np
import pandas as pd
import math
import csv
import sys


def main(argv):
    train = pd.read_csv(argv[2])
    train = pd.DataFrame.as_matrix(train)
    label = pd.read_csv(argv[3])
    label = pd.DataFrame.as_matrix(label)
    test = pd.read_csv(argv[4], encoding='big5')
    test = pd.DataFrame.as_matrix(test)
    test = test.astype(float)
    label = np.squeeze(label, axis=1)

    mu0 = np.zeros(106)
    mu1 = np.zeros(106)
    zeros = []
    ones = []
    for i in range(train.shape[0]):
        if label[i]:
            mu1 += train[i]
            ones.append(i)
        else:
            mu0 += train[i]
            zeros.append(i)

    mu1 /= len(ones)
    mu0 /= len(zeros)

    cov1 = np.cov(train[ones], rowvar=False)
    cov0 = np.cov(train[zeros], rowvar=False)

    p1 = len(ones) / train.shape[0]
    p0 = 1 - p1
    cov = p1 * cov1 + p0 * cov0

    inverse = np.linalg.pinv(cov)
    deter = np.linalg.det(cov)
    F = 1/((2 * math.pi) ** 53) * (deter ** (1/2))

    f = open(argv[5], 'w')
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    for n in range(test.shape[0]):
        x = test[n]
        f0 = F * np.exp((-1. / 2.) * np.dot(np.dot(x - mu0, inverse), x.T - mu0.T))
        f1 = F * np.exp((-1. / 2.) * np.dot(np.dot(x - mu1, inverse), x.T - mu1.T))
        P1 = (f1 * p1) / (f1 * p1 + f0 * p0)
        predict = 1 if P1 > 0.5 else 0
        w.writerow([str(n + 1), predict])
    f.close()


if __name__ == '__main__':
    main(sys.argv[1:])
