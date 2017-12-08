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
import matplotlib.pyplot as plt
import pickle 

sentence_length = 30
batch_size = 128
output_dim = 2
max_features = 20000
filters = 128
kernel_size = 7
epochs = 10

def build_model():
    
    model = Sequential()

    model.add(Embedding(max_features, 128))
    model.add(Dropout(0.5))
    
    model.add(Bidirectional(GRU(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.25))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim,activation="softmax"))
    return model


def main():
    
    data = pd.read_csv("./training_label.txt",sep='\+\+\+\$\+\+\+ ', header=None,engine='python')
    
    hue = list(data[1])
    label = np.array(data[0])
    
    
    val_label = label[:10000]
    label = label[10000:] 

    
    label = np_utils.to_categorical(label, output_dim)
    val_label = np_utils.to_categorical(val_label, output_dim)

    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(hue)
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    train = []

    for n in range(len(hue)):
        train.append(tokenizer.texts_to_sequences([hue[n]])[0])
    
    
    train = sequence.pad_sequences(train, maxlen= sentence_length)
    val = train[:10000]
    train = train[10000:]
    
    model = build_model()
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)
    model.summary()
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    '''
    change_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,min_lr=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto') 
    history = model.fit(train, label,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(val, val_label),
          callbacks=[change_lr,es])
    '''
    
    history = model.fit(
            train,
            label,
            batch_size=1024,
            epochs=25,
            validation_data=(val, val_label),
            
    )

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./acc_rnn.png')
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_rnn.png')
        
    
    model.save("./LSTM.h5")
    
    


    

    
    
if __name__ == '__main__':
    main()
