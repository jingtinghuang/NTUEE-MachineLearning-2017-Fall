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
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE  

sentence_length = 30




def main():


    
    
    movie = pd.read_csv("movies.csv",delimiter = "::",engine='python')
    
    test = pd.read_csv("test.csv",delimiter = ",",engine='python')
    test = pd.DataFrame.as_matrix(test)

    movie = movie['Genres']
    movie = movie.str.split('|',1,expand = True)
    users = test[:,1]
    movies = test[:,2]
    users -= 1
    movies -= 1 
    movie = pd.DataFrame.as_matrix(movie)
    y = np.zeros((movie.shape[0],8))
    #y = np.zeros((movie.shape[0],))
    lul = {
        'Action':0,
	    'Adventure' :0,
	    'Animation':1,
	    "Children's":1,
	    'Comedy':2,
        'Crime':3,
	    'Documentary':4,
	    'Drama':2,
	    'Fantasy':5,
	    'Film-Noir':3,
	    'Horror':6,
	    'Musical':2,
	    'Mystery':5,
	    'Romance':2,
	    'Sci-Fi':5,
	    'Thriller':6,
	    'War':7,
	    'Western':0
    }
    print(movie[1])
    print(movie[-1])
    i = 0
    for a in movie:
        y[i,lul[a[0]]] = 1
        # y[i] = lul[a[0]]
        i += 1
    print(y[-1])
    model = load_model("./final/best.h5")
    u = np.array(model.layers[2].get_weights()).squeeze()
    m = np.array(model.layers[3].get_weights()).squeeze()
    m = np.array(m,dtype=np.float64)
    vis_data = TSNE(n_components=2).fit_transform(m)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]

    cm = plt.cm.get_cmap('Dark2')
    sc = plt.scatter(vis_x,vis_y, c = y,cmap=cm)
    # plt.colorbar(sc)
    plt.show()





   

    
    '''
    predictions = model.predict([users,movies])
   
    f = open("./result.csv", 'w')
    w = csv.writer(f)
    w.writerow(['TestDataID','Rating'])
    for n in range(predictions.shape[0]):
        w.writerow([str(n+1), float(predictions[n])])
    f.close()
    '''
    


if __name__ == '__main__':
    main()
