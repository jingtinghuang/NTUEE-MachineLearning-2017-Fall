import numpy as np
import pandas as pd
import csv
import sys
import math

def sigmoid(scores):
    #scores = scores * 1e-9 + 1
    return 1. / (1. + np.exp(-scores))

norm_index = [1, 2, 4, 5, 6]

def main(argv):
    global label,x,y


    test = pd.read_csv(argv[4],encoding='big5')


    test.replace("NR", 0, inplace=True)
    weight = np.loadtxt("./log_weight.txt")
    weight = weight.astype(float)
    test = pd.DataFrame.as_matrix(test)
    test = test.astype(float)
    test = np.concatenate((np.ones((test.shape[0], 1)), test), axis=1)
    for i in norm_index:
        mu = np.mean(test[:, i], axis=0)
        sig = np.std(test[:, i], axis=0)
        test[:, i] = (test[:, i] - mu) / sig
    f = open(argv[5],'w')
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    
    

    for n in range(test.shape[0]):
        x = test[n]
        hypo = np.dot(x, weight)
        hypo = sigmoid(hypo)
        result = np.around(hypo)
        w.writerow([str(n+1),int(result)])
    

    f.close()








if __name__ == '__main__':
    main(sys.argv[1:])
