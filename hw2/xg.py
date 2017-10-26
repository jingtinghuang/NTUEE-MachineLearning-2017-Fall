import numpy as np
import pandas as pd
import numpy as np
import sys
import csv
import xgboost as xg 

norm_index = [0,1,3,4,5]
batch_size = 5000
nb_classes = 2
nb_epoch = 200
input_dim = 106


def main(argv):
    train = pd.read_csv(argv[2])
    train = pd.DataFrame.as_matrix(train)
    test = pd.read_csv(argv[4])
    test = pd.DataFrame.as_matrix(test)
   
    label = pd.read_csv(argv[3])
    label = pd.DataFrame.as_matrix(label)
    label = np.squeeze(label, axis=1)
    
    # train = np.concatenate((np.ones((train.shape[0], 1)), train), axis=1)
   

    '''
    for i in norm_index:
        mu = np.mean(train[:, i], axis=0)
        sig = np.std(train[:, i], axis=0)
        train[:, i] = (train[:, i] - mu) / sig

    for i in norm_index:
        mu = np.mean(test[:, i], axis=0)
        sig = np.std(test[:, i], axis=0)
        test[:, i] = (test[:, i] - mu) / sig
    '''

    val = train[-3000:]
    #train = train[:-3000]
    val_label = label[-3000:]
    #label = label[:-3000]       

    #train = np.reshape(train, (train.shape[0],train.shape[1],1))
    #test = np.reshape(test, (test.shape[0],test.shape[1],1))
    gbm = xg.XGBClassifier(max_depth=3,
        n_estimators=1000,learning_rate=0.07).fit(train, label)
    validations = gbm.predict(val)
    '''
    count = 0
    for i in range(val.shape[0]):
        if validations[i] == val_label[i]:
            count += 1
    print(count/validations.shape[0])
    '''
    predictions = gbm.predict(test)
    '''
    count = 0
    for i in range(test.shape[0]):
        if predictions[i] == val_label[i]:
            count += 1
    print(count/predictions.shape[0])
    '''
    f = open(argv[5], 'w')
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    for n in range(predictions.shape[0]):
        w.writerow([str(n+1), int(predictions[n])])
    f.close()



if __name__ == '__main__':
    main(sys.argv[1:])
