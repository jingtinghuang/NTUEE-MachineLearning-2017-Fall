import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Dot, Add, Concatenate, Dense
from keras.layers import LSTM, Conv1D, MaxPooling1D, Bidirectional, GRU
from keras.layers import Flatten, Input, Dropout
from keras.utils import np_utils
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
import pickle 

max_features = 0
batch_size = 128
factors_num = 200
epochs = 200

filters = 128

def build_model(users, movies, f=factors_num):
    uin = Input(shape = (None,))
    
    U = Embedding(users, f, input_length=1)(uin)
    UB = Embedding(users,1,input_length=1)(uin)
    UB = Flatten()(UB)
    U = Flatten()(U)
    '''beyond strong baseline'''
    U = Dropout(0.5)(U)

    m_in = Input(shape = (None,))
    M = Embedding(movies, f, input_length=1)(m_in)
    MB = Embedding(movies,1,input_length=1)(m_in)
    MB = Flatten()(MB)
    M = Flatten()(M)
    '''beyond strong baseline'''
    M = Dropout(0.5)(M)
    UM = Concatenate()([U,M])
    UM = Dense(128,activation="relu")(UM)
    UM = Dense(52, activation='relu')(UM)

    out = Dense(1)(UM)
    '''strong baseline''' 
    #out = Add()([out,UB])
    #out = Add()([out,MB])
    model = Model([uin,m_in],out)

    # model = Sequential()
    # model.add(Merge([U, M], mode='dot', dot_axes=1))
    
    return model



def main():
    
    train = pd.read_csv("train.csv",sep=',',engine='python',)
    test = pd.read_csv("test.csv",delimiter = ",",engine='python')
    movies = pd.read_csv("movies.csv",delimiter = "::",engine='python')
    users = pd.read_csv("users.csv",delimiter = "::",engine='python')
    users.replace("M", 1, inplace=True)
    users.replace("F", 0, inplace=True)
    train = pd.DataFrame.as_matrix(train)
    test = pd.DataFrame.as_matrix(test)
    movies = pd.DataFrame.as_matrix(movies)
    users = pd.DataFrame.as_matrix(users)
    
    
    # users = users.astype(float)
    
    label = train[:,3]
    train = train[:,:3]
    

    num_movies = np.max(train[:,2])
    num_users = np.max(train[:,1])

    
    
    Users = train[:,1]
    Users -= 1
    np.random.seed(0)
    np.random.shuffle(Users)
    users_val = Users[:80000]
    Users = Users[80000:]
    

    Movies = train[:,2]
    Movies -= 1 
    np.random.seed(0)
    np.random.shuffle(Movies)
    movies_val = Movies[:80000]
    Movies = Movies[80000:]

    '''
    mu = np.mean(label)
    sig = np.std(label)
    label = (label - mu) / sig

    np.savetxt('./mu.txt',[mu])
    np.savetxt('./sig.txt',[sig])
    '''

    np.random.seed(0)
    np.random.shuffle(label)
    val_label = label[:80000]
    label = label[80000:]

    

    
    

    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    #model = CFModel(num_users, num_movies, factors_num)
    model = build_model(num_users, num_movies)
    model.summary()
    model.compile(loss='mse', optimizer='adamax')
    callbacks = [EarlyStopping('val_loss', patience=3), 
             ModelCheckpoint("./MF"+str(factors_num)+".h5", save_best_only=True),
             ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,min_lr=0)]
    history = model.fit([Users, Movies], label, epochs=epochs,
        batch_size=batch_size, validation_data=([users_val, movies_val],val_label),callbacks=callbacks)

    
    # model.save("./MF"+str(factors_num)+".h5")

    

    
    
    
    
    
    
if __name__ == '__main__':
    main()
