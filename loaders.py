#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:51:16 2021

@author: yanni
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os
from os.path import join, isfile
from os import listdir
import sigpy as sp
from scipy import signal

        # path = sys.path[0] + '/data/train-val-data/'
        # path = path + data + '-' + 'experiment-' + str(experiment) + '.txt'
        # self.filenames = np.loadtxt(path, dtype='str')
        # data = []

        # # ~850  files
        # for ii in range(len(self.filenames)):
        #     with np.load(self.filenames[ii], allow_pickle=True) as f:
        #         data.append(f['data'])
        # reshaped_data = np.array(data, dtype=np.complex64).transpose(0, 3, 1, 2)
        # self.channels = np.reshape(reshaped_data, (-1, reshaped_data.shape[-2], reshaped_data.shape[-1]))
        
        # thr = 0.4
        # for i in range(len(self.channels)):
        #     z = np.max(np.abs(self.channels[i]))
        #     mask = np.abs(self.channels[i]) > thr * z
        #     self.channels[i] = self.channels[i] * mask
        #     self.channels[i] = sp.fft(self.channels[i], axes=(-1,-2))

        # X = []
        # for i in range(0, 10000):
        #     x0 = np.zeros((128*128), dtype=complex)
        #     Npoints = np.random.randint(1,10)
        #     idx = np.random.randint(128*128, size=Npoints)

        #     x0[idx] = np.random.randn(*idx.shape) + 1j * np.random.randn(*idx.shape)

        #     x0 = x0.reshape((128, 128))
        #     N = 1
        #     filt = np.outer(np.hamming(N), np.hamming(N))
        #     x0 = signal.convolve(x0, filt, mode='same')
        #     y0 = sp.fft(x0, axes=(-1,-2))
        #     y0 = y0 / np.max(np.abs(y0))
        #     X.append(y0)
        
        # path = '/home/asad/score-based-channels/data/marius-data/'
        # filenames = [path + f for f in listdir(path)]
        # self.channels  = []

        # i = 0
        # for filename in filenames:# Preload file and serialize
        #     if (i == 0):
        #         contents = hdf5storage.loadmat(filenames[0])
        #         channels = np.asarray(contents['output_h'], dtype=np.complex64)
        #         i += 1
            
        #     contents = hdf5storage.loadmat(filename)
        #     channels = np.vstack((channels, np.asarray(contents['output_h'], dtype=np.complex64)))
        
        # self.channels.append(channels[:, 0])

        # # Normalization
        # self.mean = np.mean(self.channels, axis=0)
        # self.std  = np.std(self.channels, axis=0)

        # # Generate random QPSK pilots
        # self.pilots = 1/np.sqrt(2) * (2 * np.random.binomial(1, 0.5, size=(
        #     self.channels.shape[0], config.data.image_size[1], config.data.num_pilots)) - 1 + \
        #         1j * (2 * np.random.binomial(1, 0.5, size=(
        #     self.channels.shape[0], config.data.image_size[1], config.data.num_pilots)) - 1))
            
        # # Complex noise power
        # self.noise_power = 1/np.sqrt(2) * config.data.noise_std

        # Normalization
        # self.channels_unnorm = self.channels.copy()
        # self.channels = self.channels / np.max(np.abs(self.channels), axis=0)


        # # Normalize
        # H_cplx = self.channels_unnorm[idx]
        # H_cplx_norm = (H_cplx - self.mean) / self.std
        
        # # Convert to reals
        # H_real_norm = \
        #     np.stack((np.real(H_cplx_norm), np.imag(H_cplx_norm)), axis=0)
        

        # # Also get Hermitian H, real-viewed
        # H_herm_norm = np.conj(np.transpose(H_cplx_norm))
        # H_real_herm_norm = \
        #     np.stack((np.real(H_herm_norm), np.imag(H_herm_norm)), axis=0)
            
        # sample = {'X': x.astype(np.float32),
        #           'H': H_real_norm.astype(np.float32),
        #           'H_herm': H_real_herm_norm.astype(np.float32),
        #           'P': self.pilots[idx].astype(np.complex64),
        #           'Y': Y.astype(np.complex64),
        #           'sigma_n': self.noise_power.astype(np.float32)}

class Channels(Dataset):

    def __init__(self, config, data, experiment):
        self.channels = torch.load(sys.path[0] + '/data/mri-data/knee-tea2.pt')['X']

        # Convert to array
        self.channels = np.asarray(self.channels)
        self.channels = np.reshape(self.channels,
               (-1, self.channels.shape[-2], self.channels.shape[-1]))
        
        # Complex noise power
        self.noise_power = 1/np.sqrt(2) * config.data.noise_std
        
    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = self.channels[idx]
        N = self.noise_power * (np.random.normal(size=x.shape) + \
            1j * np.random.normal(size=x.shape))
        Y = x + N

        x = np.stack((np.real(x), np.imag(x)), axis=0)

        sample = {'X': x.astype(np.float32),
                  'Y': Y.astype(np.complex64)}

        return sample