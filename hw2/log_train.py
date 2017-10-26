import numpy as np
import pandas as pd
import math



learning_rate = 1
weight = np.zeros(107)
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
m = np.zeros(weight.shape)
v = np.zeros(weight.shape)
cache = np.zeros(weight.shape)
x = []
y = []
iterations = 5000
reg = 0.00
norm_index = [1, 2, 4, 5, 6]

def sigmoid(scores):
    #scores = scores * 1e-9 + 1
    return 1. / (1. + np.exp(-scores))

def sigdef(d):
    return sigmoid(d) * (1 - sigmoid(d))



def gd(x,y):
    global weight, bias, learning_rate, m, v, cache


    for i in range(iterations):

        hypo = np.dot(x, weight)
        hypo = sigmoid(hypo)
        compare = np.zeros(hypo.shape)
        result = np.around(hypo)
        loss = hypo - y
        #cost = - np.sum(np.multiply(y, np.log(hypo)) + np.multiply((1 - y), np.log(1 - hypo)))

        #cost /= x.shape[0]
        compare[result == y] = 1
        #cost = np.sum(loss ** 2) / x.shape[0]
        #cost += reg * np.sum(weight * weight)
        #cost_a = math.sqrt(cost)
        #loss = sigdef(loss)
        gra = np.dot(x.T, loss)
        # gra += weight * reg * 2
        '''
        s_gra += gra ** 2
        ada = np.sqrt(s_gra)
        weight = weight - learning_rate * gra / ada
        '''
        m = beta1 * m + (1 - beta1) * gra
        mt = m / (1 - beta1 ** (i + 1))
        v = beta2 * v + (1 - beta2) * (gra ** 2)
        vt = v / (1 - beta2 ** (i + 1))
        weight += - learning_rate * mt / (np.sqrt(vt) + eps)
        
        if i % 50 == 0:
            print('iteration: %d | Cost: %f  ' % (i, np.sum(compare)/y.shape[0]))
            #print(cost)






def main():
    train = pd.read_csv("./X_train")
    train = pd.DataFrame.as_matrix(train)
    label = pd.read_csv("./Y_train")
    label = pd.DataFrame.as_matrix(label)
    label = np.squeeze(label, axis=1)
    train = np.concatenate((np.ones((train.shape[0], 1)), train), axis=1)
    print(train[0])
    for i in norm_index:
        mu = np.mean(train[:, i], axis=0)
        sig = np.std(train[:, i], axis=0)
        train[:, i] = (train[:, i] - mu) / sig
    print(train[0])
    print(label.shape)
    print(label[:10])







    gd(train,label)

    np.savetxt("./weight.txt", weight)






if __name__ == '__main__':
    main()
