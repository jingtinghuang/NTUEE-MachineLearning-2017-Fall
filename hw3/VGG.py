import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import keras.backend as K
import csv
import sys

input_shape = (48,48,1)
output_dim = 7
batch_size = 128
epochs = 300




def build_model():
    
    model = Sequential()
    model.add(Conv2D(64,kernel_size = (3,3),
        input_shape = input_shape,activation="relu",
        padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64,kernel_size = (3,3),
        activation="relu",padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(128,kernel_size = (3,3),
        activation="relu",padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128,kernel_size = (3,3),
        activation="relu",padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    

    model.add(Conv2D(192,kernel_size = (3,3),
        activation="relu",padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(192,kernel_size = (3,3),
        activation="relu",padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(256,kernel_size = (3,3),
        activation="relu",padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256,kernel_size = (3,3),
        activation="relu",padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    model.add(Flatten())

    model.add(Dense(1024,activation="relu"))
    model.add(Dropout(0.25))
    
    model.add(Dense(output_dim,activation="softmax"))
    return model

def main(argv):
    
    train = pd.read_csv(argv[0])
    train = pd.DataFrame.as_matrix(train)
    data = np.ones((train.shape[0],2304))
    label = np.ones((train.shape[0],))
    for n in range(train.shape[0]):
        data[n] = np.array(train[n,1].split(' '),int)
        label[n] = train[n,0]
    train = data 
    
    mean = np.mean(train, axis = 0)
    sig = np.std(train, axis = 0)

    

    train -= mean
    train /= sig

      
    

    train = np.reshape(train,(-1,48,48,1))
    

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    
    
    val = train[-1000:]
    train = train[:-1000]
    val_label = label[-1000:]
    label = label[:-1000]  



         

    #train = np.reshape(train, (train.shape[0],train.shape[1],1))
    #test = np.reshape(test, (test.shape[0],test.shape[1],1))
    label = np_utils.to_categorical(label, output_dim)
    val_label = np_utils.to_categorical(val_label, output_dim)
    model = build_model()
    # nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, 
    #     beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.summary()
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    change_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=0)
    
    history = model.fit_generator(datagen.flow(train, label,batch_size=batch_size), 
        steps_per_epoch = train.shape[0]/batch_size, epochs=epochs,validation_data=(val, val_label),callbacks=[change_lr])
   
    
    model.save("./model.h5")
    """
    predictions = model.predict(test)
    predictions = np.argmax(predictions,axis=1)
    f = open("./result.csv", 'w')
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    for n in range(predictions.shape[0]):
        w.writerow([str(n), int(predictions[n])])
    f.close()
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./acc_vgg.png')
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_vgg.png')
    
    


if __name__ == '__main__':
    main(sys.argv[1:])
