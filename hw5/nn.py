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
    # g = Input(shape = (None,))
    


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
    movies = movies['Genres']
    movies = movies.str.split('|',1,expand = True)
    train = pd.DataFrame.as_matrix(train)
    num_movies = np.max(train[:,2])
    num_users = np.max(train[:,1])
    movies = pd.DataFrame.as_matrix(movies)
    y = np.zeros((num_movies,))
    #y = np.zeros((movie.shape[0],))
    lul = {
        'Action':0,
	    'Adventure' :1,
	    'Animation':2,
	    "Children's":3,
	    'Comedy':4,
        'Crime':5,
	    'Documentary':6,
	    'Drama':7,
	    'Fantasy':8,
	    'Film-Noir':9,
	    'Horror':10,
	    'Musical':11,
	    'Mystery':12,
	    'Romance':13,
	    'Sci-Fi':14,
	    'Thriller':15,
	    'War':16,
	    'Western':17
    }
    
    i = 0
    for a in movies:
        y[i] = lul[a[0]]
        i += 1
    print(y[:10])
    test = pd.DataFrame.as_matrix(test)
    users = pd.DataFrame.as_matrix(users)
    
    
    # users = users.astype(float)
    
    label = train[:,3]
    train = train[:,:3]
    

    num_movies = np.max(train[:,2])
    num_users = np.max(train[:,1])

    
    
    Users = train[:,1]
    Users -= 1

    genre = []

    np.random.seed(0)
    np.random.shuffle(Users)
    
    
    print(len(y))
    Movies = train[:,2]
    Movies -= 1 
    np.random.seed(0)
    np.random.shuffle(Movies)
    gender = []
    for ggg in Users:
        gender.append(1)
    for x in Movies:
        genre.append(y[x])
    genre = np.array(genre)
    genre_val = genre[:80000]
    genre = genre[80000:] 
    users_val = Users[:80000]
    Users = Users[80000:]
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
    '''
    history = model.fit([Users, Movies,genre,gender], label, epochs=epochs,
        batch_size=batch_size, validation_data=([users_val, movies_val],val_label),callbacks=callbacks)
    '''

    
    # model.save("./MF"+str(factors_num)+".h5")

    

    
    
    
    
    
    
if __name__ == '__main__':
    main()
