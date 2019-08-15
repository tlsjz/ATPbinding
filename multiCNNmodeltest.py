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
    
def multimodel(part):
    traindatapickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/'+part+'/train.pickle','rb')
    #traindatapickle = open('/home/songjiazhi/atpbinding/atp227/feature15.pickle','rb')
    traindata = pickle.load(traindatapickle)
    label_train = traindata[0]
    pssmfeature_train_pickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/'+part+'/pssmfeature.pickle','rb')
    #pssmfeature_train_pickle = open('/home/songjiazhi/atpbinding/evaluate/pssmfeature.pickle','rb')
    pssmfeature_train = pickle.load(pssmfeature_train_pickle)
    psipredfeature_train_pickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/'+part+'/psipredfeature.pickle','rb')
    #psipredfeature_train_pickle = open('/home/songjiazhi/atpbinding/evaluate/psipredfeature.pickle','rb')
    psipredfeature_train = pickle.load(psipredfeature_train_pickle)
    chemicalfeature_train_pickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/'+part+'/chemicalfeature.pickle','rb')
    #chemicalfeature_train_pickle = open('/home/songjiazhi/atpbinding/evaluate/chemicalfeature.pickle','rb')
    chemicalfeature_train = pickle.load(chemicalfeature_train_pickle)
    
    testdatapickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/'+part+'/test.pickle','rb')
    #testdatapickle = open('/home/songjiazhi/atpbinding/independent/feature15.pickle','rb')
    testdata = pickle.load(testdatapickle)
    label_test = testdata[0]
    pssmfeature_test_pickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/'+part+'/pssmfeature_test.pickle','rb')
    #pssmfeature_test_pickle = open('/home/songjiazhi/atpbinding/evaluate/pssmfeature_test.pickle','rb')
    pssmfeature_test = pickle.load(pssmfeature_test_pickle)
    psipredfeature_test_pickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/'+part+'/psipredfeature_test.pickle','rb')
    #psipredfeature_test_pickle = open('/home/songjiazhi/atpbinding/evaluate/psipredfeature_test.pickle','rb')
    psipredfeature_test = pickle.load(psipredfeature_test_pickle)
    chemicalfeature_test_pickle = open('/home/songjiazhi/atpbinding/atp227/fivefold/'+part+'/chemicalfeature_test.pickle','rb')
    #chemicalfeature_test_pickle = open('/home/songjiazhi/atpbinding/evaluate/chemicalfeature_test.pickle','rb')
    chemicalfeature_test = pickle.load(chemicalfeature_test_pickle)
    
    pssmfeature_train = np.array(pssmfeature_train)
    pssmfeature_train = pssmfeature_train.reshape(-1,1,15,20)
    psipredfeature_train = np.array(psipredfeature_train)
    psipredfeature_train = psipredfeature_train.reshape(-1,1,15,3)
    chemicalfeature_train = np.array(chemicalfeature_train)
    chemicalfeature_train = chemicalfeature_train.reshape(-1,1,15,7)
    label_train_one = np_utils.to_categorical(label_train, num_classes=2)
    
    pssmfeature_test = np.array(pssmfeature_test)
    pssmfeature_test = pssmfeature_test.reshape(-1,1,15,20)
    psipredfeature_test = np.array(psipredfeature_test)
    psipredfeature_test = psipredfeature_test.reshape(-1,1,15,3)
    chemicalfeature_test = np.array(chemicalfeature_test)
    chemicalfeature_test = chemicalfeature_test.reshape(-1,1,15,7)
    label_test_one = np_utils.to_categorical(label_test, num_classes=2)
    
    pssminput = Input((1,15,20))
    psipredinput = Input((1,15,3))
    chemicalinput = Input((1,15,7))
    
    #chemical model
    chemical_x = Convolution2D(64, (1,1), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(chemicalinput)
    chemical_x = BatchNormalization(axis=1, scale=False)(chemical_x)
    chemical_x = Activation('relu')(chemical_x)
    
    chemical_x = Convolution2D(64, (3,3), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(chemical_x)
    chemical_x = BatchNormalization(axis=1, scale=False)(chemical_x)
    chemical_x = Activation('relu')(chemical_x)
    
    chemical_x = Convolution2D(96, (3,3), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(chemical_x)
    chemical_x = BatchNormalization(axis=1, scale=False)(chemical_x)
    chemical_x = Activation('relu')(chemical_x)  
    
    chemical_x = Flatten()(chemical_x)
    chemical_x = BatchNormalization(axis=1, scale=False)(chemical_x)
    chemical_x = Dense(256, activation='relu')(chemical_x)
    chemical_x = Dropout(0.5)(chemical_x)
    chemical_x = Dense(128, activation='relu')(chemical_x)
    
    #psipred model
    psipred_x = Flatten()(psipredinput)
    psipred_x = BatchNormalization(axis=1, scale=False)(psipred_x)
    psipred_x = Dense(64, activation='relu')(psipred_x)
    
    #pssm model
    
    pssm_x = Convolution2D(64, (3,3), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(pssminput)
    pssm_x = BatchNormalization(axis=1, scale=False)(pssm_x)
    pssm_x = Activation('relu')(pssm_x)
    
    pssm_x = Convolution2D(64, (3,3), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(pssm_x)
    pssm_x = BatchNormalization(axis=1, scale=False)(pssm_x)
    pssm_x = Activation('relu')(pssm_x)
    
    pssm_x = Convolution2D(96, (5,5), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(pssm_x)
    pssm_x = BatchNormalization(axis=1, scale=False)(pssm_x)
    pssm_x = Activation('relu')(pssm_x)
    
    pssminput_reshape = core.Reshape((1,20,15))(pssminput)
    pssm_reshape_x = Convolution2D(64, (3,3), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(pssminput_reshape)
    pssm_reshape_x = BatchNormalization(axis=1, scale=False)(pssm_reshape_x)
    pssm_reshape_x = Activation('relu')(pssm_reshape_x)
    
    pssm_reshape_x = Convolution2D(64, (3,3), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(pssm_reshape_x)
    pssm_reshape_x = BatchNormalization(axis=1, scale=False)(pssm_reshape_x)
    pssm_reshape_x = Activation('relu')(pssm_reshape_x)
    
    pssm_reshape_x = Convolution2D(96, (5,5), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(pssm_reshape_x)
    pssm_reshape_x = BatchNormalization(axis=1, scale=False)(pssm_reshape_x)
    pssm_reshape_x = Activation('relu')(pssm_reshape_x)    
    
    pssm_x = Flatten()(pssm_x)
    pssm_reshape_x = Flatten()(pssm_reshape_x)
    
    pssm = Concatenate(axis=1)([pssm_x, pssm_reshape_x])
    pssm = BatchNormalization(axis=1, scale=False)(pssm)
    pssm = Dense(256, activation='relu')(pssm)
    pssm = Dropout(0.5)(pssm)
    pssm = Dense(128, activation='relu')(pssm)
    
    concatenate
    output = Concatenate(axis=1)([chemical_x, psipred_x, pssm])
    #output = Concatenate(axis=1)([psipred_x, pssm])    
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(2,activation='softmax')(output)
    
    testmodel = Model([pssminput, psipredinput, chemicalinput], output)
    #testmodel = Model([pssminput, psipredinput],output)
    adam = Adam(lr=0.0001,epsilon=1e-08)
    testmodel.compile(optimizer=adam, loss='binary_crossentropy',metrics=['binary_accuracy'])   
    #testmodel.summary()
    
    print('Training------')
    #filepath = '/home/songjiazhi/atpbinding/atp227/multideepmodel/5/weights-{epoch:02d}.hdf5'
    filepath = '/home/songjiazhi/atpbinding/evaluate/multimodels/weights-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath, save_best_only=False, save_weights_only=False)     
    class_weight = {0:0.5211,1:12.3510}
    #testmodel.fit([pssmfeature_train, psipredfeature_train, chemicalfeature_train], label_train_one, epochs=15, batch_size=256, class_weight=class_weight, shuffle=True, callbacks=[roc_callback(training_data=([pssmfeature_train, psipredfeature_train, chemicalfeature_train], label_train_one), validation_data=([pssmfeature_test, psipredfeature_test, chemicalfeature_test], label_test_one))])    
    testmodel.fit([pssmfeature_train, psipredfeature_train], label_train_one, epochs=15, batch_size=256, class_weight=class_weight, shuffle=True, callbacks=[roc_callback(training_data=([pssmfeature_train, psipredfeature_train], label_train_one), validation_data=([pssmfeature_test, psipredfeature_test], label_test_one)), checkpoint])    
    #multiprediction = testmodel.predict([pssmfeature_test, psipredfeature_test])
    #multipredictionpickle = open('/home/songjiazhi/atpbinding/atp227/featureimportance/pssmss/fivefold/5/multiprediction.pickle','wb')
    #pickle.dump(multiprediction, multipredictionpickle)
multimodel('5')
    
    
    
    
    
    
    