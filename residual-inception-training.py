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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    #x = Dropout(0.3)(x)
    return x

def ResidualInception(part):
    #traindatapickle = open('/home/songjiazhi/atpbinding/atp227/feature15.pickle','rb')
    #traindatapickle = open('/home/songjiazhi/atpbinding/atp388/fivefold/'+str(part)+'/train.pickle','rb')
    #traindatapickle = open('/home/songjiazhi/atpbinding/atp227/featureimportance/pssmssasa/fivefold/'+str(part)+'/fulltrain.pickle','rb')
    traindatapickle = open('/home/songjiazhi/atpbinding/traindata/fulldata.pickle','rb')
    traindata = pickle.load(traindatapickle)
    feature_train = traindata[1]
    label_train = traindata[0]
    
    #testdatapickle = open('/home/songjiazhi/atpbinding/independent/feature15.pickle','rb')
    #testdatapickle = open('/home/songjiazhi/atpbinding/atp388/fivefold/'+str(part)+'/test.pickle','rb')
    #testdatapickle = open('/home/songjiazhi/atpbinding/atp227/featureimportance/pssmssasa/fivefold/'+str(part)+'/test.pickle','rb')
    testdatapickle = open('/home/songjiazhi/atpbinding/atp41/newfeature15.pickle','rb')
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
    
    branch1_1 = conv2d_bn(inputfile, 64, 3, 3)
    
    branch1_2 = conv2d_bn(inputfile, 64,3, 3)
    branch1_2 = conv2d_bn(branch1_2, 64, 3, 3)
    
    branch1_3 = conv2d_bn(inputfile, 64, 3, 3)
    branch1_3 = conv2d_bn(branch1_3, 64, 3, 3)
    branch1_3 = conv2d_bn(branch1_3, 64, 3, 3)
    branch1_3 = conv2d_bn(branch1_3, 64, 3, 3)
    
    shortcut_1 = conv2d_bn(inputfile, 64, 1, 1)
    
    incep_1_out = Concatenate(axis=1)([branch1_1, branch1_2, branch1_3])
    
    branch2_1 = conv2d_bn(incep_1_out, 96, 3, 3)
    
    branch2_2 = conv2d_bn(incep_1_out, 96, 3, 3)
    branch2_2 = conv2d_bn(branch2_2, 96, 3, 3)
    
    incep_2_out = Concatenate(axis=1)([branch2_1, branch2_2, shortcut_1])
    
    #branch3_1 = conv2d_bn(incep_2_out, 200, 1, 1, strides=(2,2), padding='valid')
    
    #branch3_2 = conv2d_bn(incep_2_out, 100, 1, 1, strides=(2,2), padding='valid')
    #branch3_2 = conv2d_bn(branch3_2, 200, 3, 3)
    
    #branch3_3 = conv2d_bn(incep_2_out, 100, 1, 1, strides=(2,2), padding='valid')
    #branch3_3 = conv2d_bn(branch3_3, 100, 3, 3)
    #branch3_3 = conv2d_bn(branch3_3, 100, 3, 3)
    #branch3_3 = conv2d_bn(branch3_3, 200, 3, 3)
    
    #incep_3_out = Concatenate(axis=1)([branch3_1, branch3_2, branch3_3])
    
    #branch4_1 = conv2d_bn(incep_3_out, 200, 3, 3)
    
    #branch4_2 = conv2d_bn(incep_3_out, 200, 3, 3)
    #branch4_2 = conv2d_bn(branch4_2, 200, 3, 3)
    
    #shortcut2 = conv2d_bn(incep_2_out, 200, 1, 1, strides=(2,2), padding='valid')
    
    #incep_4_out = Concatenate(axis=1)([branch4_1, branch4_2, shortcut2])
    
    incepout = conv2d_bn(incep_2_out, 100, 11, 11)
    incepout = Flatten()(incepout)
    incepout = Dense(256, activation='relu')(incepout)
    incepout = Dropout(0.4)(incepout)
    incepout = Dense(64, activation='relu')(incepout)
    #incepout = GlobalAveragePooling2D()(incepout)
    prediction = Dense(2, activation='softmax')(incepout)
    
    IncepResidualModel = Model(inputfile, prediction)
    adam = Adam(lr=0.0001,epsilon=1e-08)
    IncepResidualModel.compile(optimizer=adam, loss='binary_crossentropy',metrics=['binary_accuracy'])  
    #IncepResidualModel.summary()
    
    print('Training------')
    #filepath = '/home/songjiazhi/atpbinding/atp388/fivefoldmodel/'+str(part)+'/reincepmodel/weights-{epoch:02d}.hdf5'
    filepath = '/home/songjiazhi/atpbinding/models/reincepmodel/weights-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath, save_best_only=False, save_weights_only=False) 
    class_weight = {0:0.5205,1:12.7113}
    IncepResidualModel.fit(feature_train, label_train_one, epochs=60, batch_size=256, class_weight = class_weight, shuffle=True, callbacks=[roc_callback(training_data=(feature_train, label_train_one), validation_data=(feature_test, label_test_one)),checkpoint])      
    
if __name__=="__main__":  
    ResidualInception('1')
    #for i in range(1,6):
        #print(i)
        #ResidualInception(str(i))
