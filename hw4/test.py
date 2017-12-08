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
import sys
import csv

sentence_length = 30




def main(argv):
    
    
    
    test = pd.read_csv(argv[0],delimiter = "\+\+\+\$\+\+\+",engine='python',header=None)
    
    y = list(test[0])
    y = y[1:]

    
    
    
    with open('./tokenizer_semi.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    test = []
    
    for n in range(len(y)):
        test.append(tokenizer.texts_to_sequences([y[n][len(str(n))+1:]])[0])
    
    test = sequence.pad_sequences(test, maxlen= sentence_length)
    

    model = load_model("./LSTM.h5")
    
    
    predictions = model.predict(test)
    predictions = np.argmax(predictions,axis=1)
    f = open(argv[1], 'w')
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    for n in range(predictions.shape[0]):
        w.writerow([str(n), int(predictions[n])])
    f.close()
    
    


if __name__ == '__main__':
    main(sys.argv[1:])
