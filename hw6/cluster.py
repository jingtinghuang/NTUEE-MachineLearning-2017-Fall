import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import csv
from numpy.random import RandomState
import sklearn.cluster
import pickle
import sys

input_shape = (48,48,1)
output_dim = 7
batch_size = 128
epochs = 300





def main(argv):
    train = np.load(argv[0])
    

    test = pd.read_csv(argv[1])
    im1 = test["image1_index"]
    #print(im1.head(5))

    im1 = pd.DataFrame.as_matrix(im1)
    #print(im1[:5])
    im2 = test["image2_index"]
    #print(im2.head(5))
    im2 = pd.DataFrame.as_matrix(im2)
    #print(im2[:5])
    '''
    kk = [29,500,6000,11600,540000,780000,1110000,1330000,1560000,1870000]
    print(train.shape)
    train = train.reshape((train.shape[0],28,28))
    print(train[0].shape)
    print(train[0])
    for n in kk:
        plt.imshow(train[im1[n]], interpolation='nearest')
        plt.show()
        plt.imshow(train[im2[n]], interpolation='nearest')
        plt.show()
    
    exit()
    '''
    
    with open('pca.pickle', 'rb') as handle:
        pca = pickle.load(handle)
    '''
    pca = PCA(n_components=300,whiten=True,svd_solver = 'randomized').fit(train)
    '''
    train_pca = pca.transform(train)
    '''
    with open('pca.pickle', 'wb') as handle:
        pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ''' 

    

    '''
    rng = RandomState(8888)
    kmeans = KMeans(n_clusters=2,random_state=rng).fit(train_pca)
    '''

    
    with open('kmeans.pickle', 'rb') as handle:
        kmeans = pickle.load(handle)
    
    '''
    with open('kmeans.pickle', 'wb') as handle:
        pickle.dump(kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    kmeans = kmeans.predict(train_pca)
    #np.round(kmeans.cluster_centers_,decimals=2)
    


    re = []
    f = open(argv[2], 'w')
    w = csv.writer(f)
    w.writerow(['ID','Ans'])
    print(im1.shape)
    print(im2.shape)
    for n in range(im1.shape[0]):
        l1 = kmeans[im1[n]]
        l2 = kmeans[im2[n]]
        
        if l1 == l2:
            w.writerow([str(n), 1])
            re.append(1)
        else:
            w.writerow([str(n), 0])
            re.append(0)
    print(np.sum(re))
    print(re[:11])
    f.close()


    
    


if __name__ == '__main__':
    main(sys.argv[1:])
