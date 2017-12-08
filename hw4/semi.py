import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM, Conv1D, MaxPooling1D, Bidirectional, GRU
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam
from keras import regularizers
import sys
import pickle 

sentence_length = 40
batch_size = 256
output_dim = 2
max_features = 20000
filters = 128

def build_model():
    
    model = Sequential()

    model.add(Embedding(max_features, 128))
    model.add(Dropout(0.5))
    
    model.add(Bidirectional(GRU(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
    model.add(GRU(128, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim,activation="softmax"))
    return model

def main(argv):
    
    data = pd.read_csv(argv[0],sep='\+\+\+\$\+\+\+ ', header=None,engine='python')
    train_no_label = pd.read_csv(argv[1], header=None,delimiter = "\+\+\+\$\+\+\+",engine='python')
    hue = list(data[1])
    label = np.array(data[0])
    xn = list(train_no_label[0])
    

    
    
    
    val_label = label[:1000]
    label = label[1000:]
    xx = np.loadtxt("nolabel_result.txt")
    top_zeros = []
    top_ones = []
    
    for i,x in enumerate(xx):
        if x[0] >= 0.99:
            top_zeros.append(i)
        elif x[1] >= 0.99:
            top_ones.append(i)

            
    
    extra_zeros = [xn[i] for i in top_zeros]
    extra_ones = [xn[i] for i in top_ones]
    
   
    #label = np.concatenate(([0]*50000,[1]*50000,label)) 

    label = np_utils.to_categorical(label, output_dim)
    val_label = np_utils.to_categorical(val_label, output_dim)

    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(hue)
    tokenizer.fit_on_texts(extra_zeros)
    tokenizer.fit_on_texts(extra_ones)

    
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    zeros = []
    ones = []
    train = []

    for n in range(len(hue)):
        train.append(tokenizer.texts_to_sequences([hue[n]])[0])
    for n in range(len(extra_ones)):
        ones.append(tokenizer.texts_to_sequences([extra_ones[n]])[0])
    for n in range(len(extra_zeros)):
        zeros.append(tokenizer.texts_to_sequences([extra_zeros[n]])[0])
    
    train = sequence.pad_sequences(train, maxlen= sentence_length)
    ones = sequence.pad_sequences(ones, maxlen = sentence_length)
    zeros = sequence.pad_sequences(zeros, maxlen = sentence_length)
    
    val = train[:1000]
    train = train[1000:]
    #train = np.concatenate((train_no_label,train))
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model = build_model()
    # model = load_model('LSTM.h5')
    model.summary()     
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    epoch_count = 25
    for i in range(epoch_count):
    

        # np.random.shuffle(zeros)
        # np.random.shuffle(ones)
        extended_train_X = np.vstack((train, zeros, ones))
        extended_train_y = np.vstack((label, np_utils.to_categorical(np.vstack((np.zeros((len(zeros),1)), np.ones((len(ones),1)))))))
        # extended_train_X = np.vstack((train, zeros[:200000], ones[:200000]))
        # extended_train_y = np.vstack((label, np_utils.to_categorical(np.vstack((np.zeros((200000,1)), np.ones((200000,1)))))))

        

        model.fit(
            extended_train_X,
            extended_train_y,
            batch_size=batch_size,
            epochs=1,
            validation_data=(val, val_label),
            
        )
        print(i)
        # model.save("./lstms/LSTM_semi"+str(i)+".h5")
    
    
        
    
    
    
    model.save("./LSTM_semi.h5")
    
    


    

    
    
if __name__ == '__main__':
    main(sys.argv[1:])
