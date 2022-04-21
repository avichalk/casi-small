"""
Contains methods which prepare data so that it may be fed into predictive
models, as well as multiple helper functions which handle loading the data and
other similar tasks.
"""


import re
from pathlib import Path
import random

import numpy as np
from astropy.io import fits
import os


def density_preprocessing(x, y):
    x = normalize(x)
    y = normalize(y)

    x = np.sign(x) * np.log10(1 + np.abs(x))
    x /= np.std(x)
    y = np.sign(y) * np.log10(1 + np.abs(y))
    y /= np.std(y)

    x = slice_density(x)
    y = slice_density(y)

    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    return x, y

def co_preprocessing(data_path='/groups/yshirley/cnntrain512'):
    if os.path.isfile('../data/temp_co/dust_filaments_0312.npy'):
        x,y=np.load('../data/temp_co/dust_filaments_0312.npy')
    else:


        tracer_files = get_co_tracer_files(data_path)
        co_files = get_co_files(data_path)
        
        #print(tracer_files)
        #print(co_files)
        #mask_files = get_co_mask_files(data_path)

        
        co, labels_co = load_fits_co(co_files)
        labels_no, tracer = load_fits_no(tracer_files)

        #print(co[0].shape)
        #print(labels_co[0].shape)
        #print(labels_no[0].shape)
        #print(tracer[0].shape)
        #tracer = load_fits(tracer_files)
        dataset = co + tracer
        labels = np.concatenate((labels_co, labels_no), axis=0)
        #print(dataset.shape)
        #labels = [1]*140+[0]*60+[1]*60+[0]*140 # len(co) len(tracer)
        #labels = [1]*len(co)+[0]*len(tracer)
        #labels = load_fits(mask_files)


        x = np.asarray(dataset)
        #print(x.shape)
        y = np.asarray(labels) # true or false for filament
        print(y.shape)
        #z = np.abs(np.asarray(true_labels))

        min_data = np.min(x)
        x = np.log(x - min_data + 1.)

        mean_data = np.mean(x)
        std_data = np.std(x)
        if std_data == 0:
            std_data = 1

        x = x - mean_data
        x = x / std_data
        
        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        #z = np.expand_dims(z, axis=-1)
        #print(x.shape)

        np.save('../data/temp_co/dust_filaments_0312.npy',[x,y])

    return x, y


def get_co_files(data_path):
    return [str(x) for x in Path(f'{data_path}/yes').glob('*.fits') if 'merge' in str(x)]

def get_co_tracer_files(data_path):
    return [str(x) for x in Path(f'{data_path}/no').glob('*.fits') if 'mask' not in str(x)]


#def get_co_mask_files(data_path):
#    return [str(x) for x in Path(f'{data_path}/yes').glob('*.fits') if 'mask' not in str(x)]

def load_fits_co(files):
    co_data = []
    mask_data = []
    print(files[0])
    for file in [files[0]]:
        with fits.open(file) as fits_data:
            co_data.append(fits_data[0].data)
        with fits.open(file.replace('image', 'mask')) as fits_data:
            mask_data.append(fits_data[0].data)

    return co_data, mask_data


def load_fits_no(files):
    co_data = []
    mask_data = []
    print(files[0])
    for file in [files[0]]:
        with fits.open(file) as fits_data: # mask data
            co_data.append(fits_data[0].data)
        with fits.open(file.replace('noimage', 'noimagemask')) as fits_data: # regular data
            mask_data.append(fits_data[0].data)

    return mask_data, co_data



def prediction_to_fits(pred, ref_files=None, outpath='../models/prediction.fits'):
    with fits.open(ref_files[0]) as fits_ref:
        ref_shape = fits_ref[0].data.shape
        fits_ref[0].data[:] = np.squeeze(pred)[:ref_shape[0], :ref_shape[1], :ref_shape[2]]
        fits_ref.writeto(outpath, overwrite=True)


#def pad_data(data, value=0.):
#    return np.pad(data,
#                  ((0, 0), (0, 1), (0, 1)),
#                  'constant',
#                  constant_values=value)


def slice_density(data):
    slices = np.empty((np.sum(data.shape), data.shape[1], data.shape[2]))
    slice_count = 0

    for i in range(data.shape[0]):
        slices[slice_count, :] = data[i, :, :]
        slice_count += 1

    for j in range(data.shape[1]):
        slices[slice_count, :] = data[:, j, :]
        slice_count += 1

    for k in range(data.shape[2]):
        slices[slice_count, :] = data[:, :, k]
        slice_count += 1

    return slices


def normalize(data):
    data -= np.mean(data)
    if np.std(data) ==0:
        data=data
    else:
        data /= np.std(data)

    return data


co_preprocessing()
