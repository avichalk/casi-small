import datetime
import json
import sys
from pathlib import Path
import os

import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler,CSVLogger, ModelCheckpoint,EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from shell_identifier_3_adaptive_lr import ShellIdentifier
import network_architectures as arch
from preprocessing_log_binary2 import co_preprocessing, density_preprocessing


if len(sys.argv) == 2:
    name = sys.argv[1]
else:
    name = 'test'

with open('hypers_3.json', 'r') as f:
    params = json.load(f)

model_hypers = params['model_hypers']
train_hypers = params['train_hypers']
data_hypers = params['data_hypers']
hypers = {**model_hypers, **train_hypers, **data_hypers}

model = ShellIdentifier(name, load=True)

x, y = co_preprocessing(data_path=data_hypers['data_path'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=2)

pred = model.predict(x_test,batch_size=hypers['batch_size'])

np.savez_compressed(f'../data/ModelOutputs/{name}_outputs',
                        X=x_test,
                        Y=y_test,
                        P=pred)
