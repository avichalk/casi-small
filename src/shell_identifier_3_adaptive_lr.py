import datetime
import json
import sys
from pathlib import Path
import os
from resource import *
import tracemalloc
import traceback

import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler,CSVLogger, ModelCheckpoint,EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

import network_architectures as arch
from preprocessing_log_binary2 import co_preprocessing, density_preprocessing


def main():

    tracemalloc.start()

    current, peak = tracemalloc.get_traced_memory()
    print(current / 10**6) 

    if len(sys.argv) == 2:
        name = sys.argv[1]
    else:
        name = 'test'
# make directories
    os.makedirs('../data/ModelOutputs/',exist_ok=True)
    os.makedirs('../data/temp_co/',exist_ok=True)
    os.makedirs('../logs/',exist_ok=True)

    with open('hypers_3.json', 'r') as f:
        params = json.load(f)
    
    current, peak = tracemalloc.get_traced_memory()
    #print(current / 10**6) 

    model_hypers = params['model_hypers']
    train_hypers = params['train_hypers']
    data_hypers = params['data_hypers']
    hypers = {**model_hypers, **train_hypers, **data_hypers}

    current, peak = tracemalloc.get_traced_memory()
    #print(current / 10**6)

    x, y = co_preprocessing(data_path=data_hypers['data_path'])
    #print(x.shape)
    np.save("../logs/before_training_x.npy", x)
    np.save("../logs/before_training_y.npy", y)

    current, peak = tracemalloc.get_traced_memory()
    #print(current / 10**6)
    current, peak = tracemalloc.get_traced_memory()
    #print(current / 10**6)
    #print(traceback.print_stack())
    model = ShellIdentifier(name, model_hypers=model_hypers)
    #current, peak = tracemalloc.get_traced_memory()
    #print(current / 10**6)
    #print('model returned')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    model.fit(x_train, y_train, **train_hypers)

    error = model.evaluate(x_test,
                           y_test,
                           batch_size=hypers['batch_size'])

    hypers["error"] = error
    log_hypers('../models/hypers_3.csv', hypers)

    print(f'Test error of trained model: {error}\n\n')

    pred = model.predict(x,batch_size=hypers['batch_size'])

    #error = model.evaluate(y, pred)

    #print(f'Total error of final model: {error}\n\n')

    np.savez_compressed(f'../data/ModelOutputs/{name}_outputs',
                        X=x,
                        Y=y,
                        P=pred)

lr = 0.001    


class ShellIdentifier:
    def __init__(self,
                 name,
                 model_hypers=None,
                 load=False):
        self.name = name
        self.gpu_count = get_gpu_count()
        self.model = None
        self.multi_gpu_model = None

        if load and (model_hypers is None):
            self.load_init(name)
        else:
            self.new_init(model_hypers)

    def new_init(self, model_hypers): 
        if self.gpu_count > 1:
            with tf.device('/cpu:0'):
                self.model = self.build_model(**model_hypers)

            self.multi_gpu_model = multi_gpu_model(self.model,
                                                   gpus=self.gpu_count)
            self.multi_gpu_model.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='mse')
        else:
            self.model = self.build_model(**model_hypers)
            print('here?')
            current, peak = tracemalloc.get_traced_memory()
            print(current / 10**6)
            self.model.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='mse')

    def load_init(self, name):
        if self.gpu_count > 1:
            with tf.device('/cpu:0'):
                self.model = load_model(name)

            self.multi_gpu_model = multi_gpu_model(self.model,
                                                   gpus=self.gpu_count)
            self.multi_gpu_model.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='mse')
        else:
            self.model = load_model(name)
            print('before model.compile cpu ver')
            current, peak = tracemalloc.get_traced_memory()
            print(current / 10**6)
            self.model.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='mse')

    def fit(self, x, y, epochs=1, batch_size=64, verbose=1):
        if self.gpu_count > 1:
            model = self.multi_gpu_model
            batch_size = batch_size * self.gpu_count
        else:
            model = self.model

        x_train, x_val, y_train, y_val = train_test_split(x,
                                                          y,
                                                          test_size=0.1)
#        gen = ImageDataGenerator(rotation_range=5,
#                                 width_shift_range=0.1,
#                                 height_shift_range=0.1,
#                                 horizontal_flip=True,
#                                 vertical_flip=True,
#                                 fill_mode='constant',
#                                 cval=0.)

        csv_logger = CSVLogger(f'../logs/{self.name}_training.csv',
                               append=True)
        checkpoint = ModelCheckpoint(f'../models/{self.name}.h5',
                                     save_weights_only=True,
                                     save_best_only=True)

#        model.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size),
#                            steps_per_epoch=x_train.shape[0] / batch_size,
#                            epochs=epochs,
#                            verbose=verbose,
#                            callbacks=[csv_logger, checkpoint],
#                            validation_data=(x_val, y_val))

        def scheduler(epoch):
            lrate=K.get_value(model.optimizer.lr)
            if epoch % 19==0:
                K.set_value(model.optimizer.lr, lrate/2.0)
            if epoch % 31==0:
                K.set_value(model.optimizer.lr, lrate*1.5)
            return K.get_value(model.optimizer.lr) 
       
        change_lr = LearningRateScheduler(scheduler)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=100, mode='auto') 
        
        model.fit(x=x_train,y=y_train,
                  batch_size=batch_size,          
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[csv_logger, checkpoint,change_lr,early_stop],
                            validation_data=(x_val, y_val))

        model.load_weights(f'../models/{self.name}.h5')

        self.save()

        return self

    def predict(self, x, batch_size=64):
        if self.gpu_count > 1:
            model = self.multi_gpu_model
            batch_size = batch_size * self.gpu_count
        else:
            model = self.model

        preds = []
        for chunk in pred_generator(x):
            preds.append(model.predict(chunk, batch_size=batch_size))

        return np.concatenate(preds)

    def evaluate(self, x, y, batch_size=64):
        if self.gpu_count > 1:
            model = self.multi_gpu_model
            batch_size = batch_size * self.gpu_count
        else:
            model = self.model

        return model.evaluate(x, y, batch_size=batch_size)

    def build_model(self,
                    filters=16,
                    noise_std=0.1,
                    activation='selu',
                    last_activation='selu'):
        print('building model')
        traceback.print_stack()
        current, peak = tracemalloc.get_traced_memory()
        print(current / 10**6)
        model = arch.restrict_net_residual_block(filters=filters,
                                   noise_std=noise_std,
                                   activation=activation,
                                   final_activation=last_activation,
                                   )
        print('done with model')
        return model
#        return arch.dilated_res_net(filters=filters,
#                                   noise_std=noise_std,
#                                   activation=activation,
#                                   final_activation=last_activation
#                                   )
            

    def save(self):
        with open(f'../models/{self.name}.json', 'w') as f:
            f.write(self.model.to_json())

        self.model.save_weights(f'../models/{self.name}.h5')


def log_hypers(log_path, hypers):
    if Path(log_path).is_file():
        hyper_log = pd.read_csv(log_path)
    else:
        hyper_log = pd.DataFrame()

    hypers["timestamp"] = datetime.datetime.now()

    hyper_log = pd.concat([hyper_log, pd.DataFrame(hypers, index=[0])])
    hyper_log = hyper_log[hyper_log['epochs'] >= 10]

    hyper_log.to_csv(log_path, index=False)


def load_model(name, optimizer='nadam', loss='mse'):
    with open(f'../models/{name}.json', 'r') as f:
        model = model_from_json(f.read())

    model.compile(optimizer=optimizer,
                  loss=loss)

    model.load_weights(f'../models/{name}.h5')

    return model


def pred_generator(x, chunk_size=256):
    for i in range(0, x.shape[0], chunk_size):
        yield x[i:i + chunk_size]


def get_gpu_count():
    devices = device_lib.list_local_devices()
    return len([x.name for x in devices if x.device_type == 'GPU'])


if __name__ == '__main__':
    main()
