import numpy as np
import pandas as pd
import math


unselected = []
features = range(18)
hours = range(9)
features_num = len(features)
hours_num = len(hours)
learning_rate = 1e-2
weight = np.zeros(((features_num-len(unselected))*hours_num)*2+1)
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
m = np.zeros(weight.shape)
v = np.zeros(weight.shape)
cache = np.zeros(weight.shape)
x = []
y = []
iterations = 60000
reg = 0.01



def gd(x):
    global weight, bias, learning_rate, m, v, cache

    x_t = x.T
    s_gra = np.zeros(len(x[0]))

    for i in range(iterations):
        hypo = np.dot(x, weight)
        loss = hypo - y
        cost = np.sum(loss ** 2) / x.shape[0]
        cost += reg * np.sum(weight * weight)
        cost_a = math.sqrt(cost)
        
        if cost_a < 5.8:
            np.savetxt("./weight.txt", weight)
        
        gra = np.dot(x_t, loss)
        gra += weight * reg * 2
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
        
        if i % 1000 == 0:
            print('iteration: %d | Cost: %f  ' % (i, cost_a))




    # my code
    '''
    weight = np.loadtxt("./weight.txt")
    weight = weight.astype(float)
    bias = np.loadtxt("./bias.txt")
    bias = float(bias)
    

    for n in range(50000):
        dX = np.zeros([features_num, hours_num])
        db = 0.0
        errors = []
        for goal in range(5750):

            current = train[:, goal:goal+hours_num]
            yH = label[goal+hours_num]

            dX += ((-2) * (yH - (np.sum(weight * current) + bias)) * current) / 5750
            db += ((-2) * (yH - (np.sum(weight * current) + bias))) / 5750

            error = (yH - (np.sum(weight * current) + bias))**2
            errors.append(error)

        
        # adam
        m = beta1 * m + (1 - beta1) * dX
        mt = m / (1 - beta1 ** (i + 1))
        v = beta2 * v + (1 - beta2) * (dX ** 2)
        vt = v / (1 - beta2 ** (i + 1))
        weight += - learning_rate * mt / (np.sqrt(vt) + eps)
        
        #ada
        cache += dX ** 2
        weight = weight - learning_rate * dX / np.sqrt(cache + eps)
        


        weight = weight - learning_rate * dX
        bias = bias - learning_rate * db
        error = np.sqrt(np.mean(errors))

        if i % 10 == 0:
            print([i, np.sqrt(np.mean(errors))])

            # learning_rate *= 0.99
        i += 1
    '''

def main():
    global label,x,y
    train = pd.read_csv("./train.csv", encoding='big5')
    test = pd.read_csv("./test.csv", encoding='big5')
    train.replace("NR", 0, inplace=True)
    test.replace("NR", 0, inplace=True)
    train = pd.DataFrame.as_matrix(train)
    train = np.delete(train,range(3), 1)
    train = train.reshape([12, 20, 18, 24])
    train = train.astype(float)
    by_month = []
    for t in train:
        gg = np.concatenate(t, 1)
        by_month.append(gg)
    by_month = np.array(by_month)

    for month in by_month:
        for j in range(471):
            x.append(month[:,j:j+hours_num])
            y.append(month[9,j+hours_num])

    x = np.array(x)
    train = []
    for n in x:
        n[9][n[9] < 0] = 0
        train.append(np.delete(n,unselected,0))
    train = np.array(train)
    train = train.reshape((-1,(features_num-len(unselected))*hours_num))
    
    train = np.concatenate((train,train**2), axis=1)
    train = np.concatenate((np.ones((train.shape[0], 1)), train), axis=1)
    y = np.array(y)


    # train_set -= np.mean(train_set)
    # train_set /= np.std(train_set,axis = 0)
    gd(train)

    np.savetxt("./weight.txt", weight)





if __name__ == '__main__':
    main()
