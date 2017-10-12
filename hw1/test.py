import numpy as np
import pandas as pd
import csv
import sys
import math



unselected = []
size = 18 - len(unselected)

def main(argv):
    global label,x,y


    test = pd.read_csv(argv[0],encoding='big5',header=None)


    test.replace("NR", 0, inplace=True)
    weight = np.loadtxt("./weight.txt")
    weight = weight.astype(float)
    test = pd.DataFrame.as_matrix(test)
    test = np.delete(test,[0,1],1)
    test = test.astype(float)
    f = open(argv[1],'w')
    w = csv.writer(f)
    w.writerow(['id', 'value'])
    
    

    for n in range(240):
        current = test[18*n:18*(n+1),:]
        current = np.delete(current,unselected,0)
        current = np.array(current)
        current[9][current[9]<0] = 0
        current = current.reshape((size * 9,))
        current = np.concatenate((current,current**2), axis=0)
        current = np.append([1],current)
        score = np.sum(current * weight)
        w.writerow(["id_"+str(n),score])
    

    f.close()








if __name__ == '__main__':
    main(sys.argv[1:])
