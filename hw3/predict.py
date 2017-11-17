import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import csv
from keras.models import load_model
import sys

input_shape = (48,48,1)
output_dim = 7
batch_size = 128
epochs = 300





def main(argv):
    
    
    test = pd.read_csv(argv[0])
    test = pd.DataFrame.as_matrix(test)
    data = np.ones((test.shape[0],2304))
    for n in range(test.shape[0]):
        data[n] = np.array(test[n,1].split(' '),int)
    test = data

    mean = np.loadtxt("./mean.txt")
    sig = np.loadtxt("./sig.txt")
    mean.astype(float)
    sig.astype(float)
    test -= mean
    test /= sig    
    test = np.reshape(test,(-1,48,48,1))
    
   
    model = load_model("./model.h5")
    model.summary()
    
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
