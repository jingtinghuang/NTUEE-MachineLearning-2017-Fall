import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import pickle
import csv
import sys

sentence_length = 30




def main(argv):
    
    
    
    test = pd.read_csv(argv[0],delimiter = ",",engine='python')
    test = pd.DataFrame.as_matrix(test)
    users = test[:,1]
    movies = test[:,2]
    users -= 1
    movies -= 1 

    model = load_model("./best.h5")

    
    
    predictions = model.predict([users,movies])
    f = open(argv[1], 'w')
    w = csv.writer(f)
    w.writerow(['TestDataID','Rating'])
    for n in range(predictions.shape[0]):
        w.writerow([str(n+1), float(predictions[n])])
    f.close()
    
    


if __name__ == '__main__':
    main(sys.argv[1:])
