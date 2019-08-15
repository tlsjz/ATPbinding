#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)    
import numpy as np
import os
import pickle
import tensorflow as tf
import keras
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import Model, load_model
from keras.layers import Concatenate
from keras import regularizers
import keras.layers.core as core
from keras.layers import Dense,Activation,Convolution2D, Convolution1D, MaxPool2D, Flatten, BatchNormalization, Dropout, Input, Bidirectional, MaxPool1D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from keras.callbacks import ModelCheckpoint
import math
import lightgbm as lgb
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class roc_callback(keras.callbacks.Callback):
    def __init__(self,training_data, validation_data):

        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):        
        y_pred = self.model.predict(self.x)
        roc = metrics.roc_auc_score(self.y, y_pred)      

        y_pred_val = self.model.predict(self.x_val)
        roc_val = metrics.roc_auc_score(self.y_val, y_pred_val)      

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return   

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    x = Convolution2D(filters, (num_row, num_col),
        kernel_initializer= 'glorot_normal',
        strides=strides,
        padding=padding,
        data_format='channels_first',
        use_bias=False)(x)
    x = BatchNormalization(axis=1, scale=False)(x)
    x = Activation('relu')(x)
    return x

def inception():
    #traindatapickle = open('/home/songjiazhi/atpbinding/atp227/feature15.pickle','rb')
    traindatapickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/5/fulltrain.pickle','rb')
    traindata = pickle.load(traindatapickle)
    feature_train = traindata[1]
    label_train = traindata[0]
    
    #testdatapickle = open('/home/songjiazhi/atpbinding/independent/feature15.pickle','rb')
    testdatapickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/5/test.pickle','rb')
    testdata = pickle.load(testdatapickle)
    feature_test = testdata[1]
    label_test = testdata[0]
    
    feature_train = np.array(feature_train)
    feature_test = np.array(feature_test)
    feature_train = feature_train.reshape(-1,1,15,31)
    feature_test = feature_test.reshape(-1,1,15,31)
    label_train_one = np_utils.to_categorical(label_train, num_classes=2)
    label_test_one = np_utils.to_categorical(label_test, num_classes=2)
    
    inputfile = Input((1,15,31))
    
    #15*31*256
    branch1x1 = conv2d_bn(inputfile, 64, 1, 1)
    
    branch5x5 = conv2d_bn(inputfile, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    
    branch3x3dbl = conv2d_bn(inputfile, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    
    branch_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(inputfile)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    
    incep_1_out = Concatenate(axis=1)([branch1x1, branch5x5, branch3x3dbl, branch_pool])
    
    #15*31*288
    branch1x1 = conv2d_bn(incep_1_out, 64, 1, 1)
    
    branch5x5 = conv2d_bn(incep_1_out, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    
    branch3x3dbl = conv2d_bn(incep_1_out, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    
    branch_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(incep_1_out)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    
    incep_2_out = Concatenate(axis=1)([branch1x1, branch5x5, branch3x3dbl, branch_pool])
    
    #15*31*288
    branch1x1 = conv2d_bn(incep_2_out, 64, 1, 1)
    
    branch5x5 = conv2d_bn(incep_2_out, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    
    branch3x3dbl = conv2d_bn(incep_2_out, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    
    branch_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(incep_2_out)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    
    incep_3_out = Concatenate(axis=1)([branch1x1, branch5x5, branch3x3dbl, branch_pool])
    
    ##7*15*768
    #branch3x3 = conv2d_bn(incep_3_out, 384, 3, 3, strides=(2,2), padding='valid')
    
    #branch3x3dbl = conv2d_bn(incep_3_out, 64, 1, 1)
    #branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    #branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2,2), padding='valid')
    
    #branch_pool = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid', data_format='channels_first')(incep_3_out)
    
    #incep_4_out = Concatenate(axis=1)([branch3x3, branch3x3dbl, branch_pool])
    
    ##7*15*768
    #branch1x1 = conv2d_bn(incep_4_out, 192, 1, 1)
    
    #branch7x7 = conv2d_bn(incep_4_out, 128, 1, 1)
    #branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    #branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    
    #branch7x7dbl = conv2d_bn(incep_4_out, 128, 1, 1)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    
    #branch_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(incep_4_out)
    #branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    
    #incep_5_out = Concatenate(axis=1)([branch1x1, branch7x7, branch7x7dbl, branch_pool])
    
    ##7*15*768
    #branch1x1 = conv2d_bn(incep_5_out, 192, 1, 1)
    
    #branch7x7 = conv2d_bn(incep_5_out, 160, 1, 1)
    #branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
    #branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    
    #branch7x7dbl = conv2d_bn(incep_5_out, 160, 1, 1)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    
    #branch_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(incep_5_out)
    #branch_pool = conv2d_bn(branch_pool, 192, 1, 1)    
    
    #incep_6_out = Concatenate(axis=1)([branch1x1, branch7x7, branch7x7dbl, branch_pool])
    
    ##7*15*768
    #branch1x1 = conv2d_bn(incep_6_out, 192, 1, 1)
    
    #branch7x7 = conv2d_bn(incep_6_out, 192, 1, 1)
    #branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    #branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    
    #branch7x7dbl = conv2d_bn(incep_6_out, 192, 1, 1)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    #branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    
    #branch_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(incep_6_out)
    #branch_pool = conv2d_bn(branch_pool, 192, 1, 1)    
    
    #incep_7_out = Concatenate(axis=1)([branch1x1, branch7x7, branch7x7dbl, branch_pool])    
    
    
    out = Flatten()(incep_3_out)
    
    #out = Dense(1024, activation='relu')(out)
    #out = Dropout(0.5)(out)
    #out = Dense(256, activation='relu', name = 'featurelayer')(out)
    #out = Dense(128, activation='relu')(out)
    #out = Dropout(0.4)(out)
    #out = Dense(64, activation='relu')(out)
    #out = GlobalAveragePooling2D(name = 'featurelayer')(incep_3_out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(0.4)(out)
    out = Dense(64, activation = 'relu')(out)
    prediction = Dense(2, activation='softmax')(out)
    
    incepmodel = Model(inputfile, prediction)
    adam = Adam(lr=0.0001,epsilon=1e-08)
    incepmodel.compile(optimizer=adam, loss='binary_crossentropy',metrics=['binary_accuracy'])   
    #incepmodel.summary()
    
    print('Training------')
    filepath = '/home/songjiazhi/atpbinding/atp227/fivefolddeepmodel/independent2/weights-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath, save_best_only=False, save_weights_only=False) 
    class_weight = {0:0.5211,1:12.3510}
    incepmodel.fit(feature_train, label_train_one, epochs=15, batch_size=256, class_weight = class_weight, shuffle=True, callbacks=[roc_callback(training_data=(feature_train, label_train_one), validation_data=(feature_test, label_test_one)), checkpoint])  
    #incepprediction = incepmodel.predict(feature_test)
    #inceppredictionpickle = open('/home/songjiazhi/atpbinding/atp227/incpeprediction/fivefold/5/incepprediction.pickle','wb')
    #pickle.dump(incepprediction, inceppredictionpickle)
    
    
    
if __name__=="__main__":   
    inception()
    
    