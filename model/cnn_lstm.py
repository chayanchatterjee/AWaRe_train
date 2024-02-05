# -*- coding: utf-8 -*-
"""CNN-LSTM model"""

''' 
 * Copyright (C) 2021 Chayan Chatterjee <chayan.chatterjee@research.uwa.edu.au>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
'''

####################################################### IMPORTS #################################################################

# internal
from .base_model import BaseModel
from SampleFileTools1 import SampleFile
from dataloader.dataloader import DataLoader

# external
from matplotlib import pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns

#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from numpy import array

from dataloader.dataloader import DataLoader

import tensorflow as tf

#tf.config.threading.set_inter_op_parallelism_threads() 
#tf.config.threading.set_intra_op_parallelism_threads()
#tf.config.set_soft_device_placement(enabled)

device_type = 'GPU'
n_gpus = 2
devices = tf.config.experimental.list_physical_devices(
          device_type)
devices_names = [d.name.split('e:')[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(
          devices=devices_names[:n_gpus])


import numpy as np
import pandas as pd

from scipy import signal
import random
import os

#os.environ["OMP_NUM_THREADS"] = “30”

from tensorflow.keras import backend as K

import h5py

#################################################################################################################################


class CNN_LSTM(BaseModel):
    """CNN_LSTM Model Class"""
    def __init__(self, config):
        super().__init__(config)
        self.num_train = self.config.train.num_training_samples
        self.num_test = self.config.train.num_test_samples
        self.n_samples = self.config.train.n_samples_per_signal
        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epoches
        self.det = self.config.train.detector
        self.depth = self.config.train.depth
        self.lr = self.config.model.layers.learning_rate
        
        self.cnn_filters_1 = self.config.model.layers.CNN_layer_1
        self.cnn_filters_2 = self.config.model.layers.CNN_layer_2
        self.lstm_1 = self.config.model.layers.LSTM_layer_1
        self.lstm_2 = self.config.model.layers.LSTM_layer_2
        self.lstm_3 = self.config.model.layers.LSTM_layer_3
        self.kernel_size= self.config.model.layers.kernel_size
        self.pool_size = self.config.model.layers.pool_size
        self.train_from_checkpoint = self.config.train.train_from_checkpoint
        self.checkpoint_path = self.config.train.checkpoint_path
        
    def load_data(self):
        """Loads and Preprocess data """
                
        # Load training data
        self.strain_train, self.signal_train = DataLoader(self.det, 'train').load_data(self.config.data)
        self.strain_test, self.signal_test = DataLoader(self.det, 'test').load_data(self.config.data)
        
        # Scale the amplitudes of the signals to lie between -1 and 1
        self.strain_train = self._preprocess_data(self.strain_train, self.num_train, self.n_samples)
        self.signal_train = self._preprocess_data(self.signal_train, self.num_train, self.n_samples)
        
        self.strain_test = self._preprocess_data(self.strain_test, self.num_test, self.n_samples)
        self.signal_test = self._preprocess_data(self.signal_test, self.num_test, self.n_samples)

        # Reshape data into overlapping sequences
        self.X_train_noisy, self.X_train_pure = self.reshape_sequences(self.num_train, self.strain_train, self.signal_train)
        self.X_test_noisy, self.X_test_pure = self.reshape_sequences(self.num_test, self.signal_test, self.signal_test)
        
        # Print the shapes of the arrays
        self.reshape_and_print()
        
        
#    def _preprocess_data(self, data, num, samples):
#       """ Scales the amplitudes of training and test set signals """
        
#        new_array = np.zeros((num,samples))

#        for i in range(num):
#            new_array[i][np.where(data[i]>0)] = data[i][data[i]>0]/(np.max(data, axis=1)[i])
#            new_array[i][np.where(data[i]<0)] = data[i][data[i]<0]/abs(np.min(data, axis=1)[i])
        
#        return new_array


    def _preprocess_data(self, data, num, samples):
        new_array = []
        for i in range(num):
            dataset = data[i]
            if((dataset.max() != 0.0) and (dataset.min() != 0.0)):
                maximum = np.max(dataset)
                minimum = np.abs(np.min(dataset))
                for j in range(samples):
                    if(dataset[j] > 0):
                        dataset[j] = dataset[j]/maximum
                    else:
                        dataset[j] = dataset[j]/minimum
            new_array.append(dataset)
        return new_array

        
# Split a univariate sequence into samples
    def split_sequence(self,sequence_noisy,sequence_pure,n_steps):
        X = [] 
        y = []
        for i in range(len(sequence_noisy)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence_noisy)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence_noisy[i:end_ix], sequence_pure[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    
    
    def reshape_sequences(self, num, data_noisy, data_pure):
        n_steps = 10
        arr_noisy = []
        arr_pure = []
        
        for i in range(num):
            X_noisy = data_noisy[i]
            X_pure = data_pure[i]
            X_noisy = np.pad(X_noisy, (10, 10), 'constant', constant_values=(0, 0))
            X_pure = np.pad(X_pure, (10, 10), 'constant', constant_values=(0, 0))
            # split into samples
            X, y = self.split_sequence(X_noisy, X_pure, n_steps)
            arr_noisy.append(X)
            arr_pure.append(y)
    
        arr_noisy = np.asarray(arr_noisy)
        arr_pure = np.asarray(arr_pure)
        
        return arr_noisy, arr_pure
    
        
    def reshape_and_print(self):
        
#        self.X_train_noisy = self.X_train_noisy.reshape(self.X_train_noisy.shape[0], 516, 4, 1)
#        self.X_test_noisy = self.X_test_noisy.reshape(self.X_test_noisy.shape[0], 516, 4, 1)
#        self.X_train_pure = self.X_train_pure.reshape(self.X_train_pure.shape[0], 516, 1)
#        self.X_test_pure = self.X_test_pure.reshape(self.X_test_pure.shape[0], 516, 1)
        
        # Reshape arrays to fit into Keras model
        self.X_train_noisy = self.X_train_noisy[:,:,:,None]
        self.X_test_noisy = self.X_test_noisy[:,:,:,None]
        self.X_train_pure = self.X_train_pure[:,:,None]
        self.X_test_pure = self.X_test_pure[:,:,None]
        
        print('x_train_noisy shape:', self.X_train_noisy.shape)
        print('x_test_noisy shape:', self.X_test_noisy.shape)
        print('x_train_pure shape:', self.X_train_pure.shape)
        print('x_test_pure shape:', self.X_test_pure.shape)

        self.X_train_noisy = self.X_train_noisy.astype("float32")
        self.X_test_noisy = self.X_test_noisy.astype("float32")

        self.X_train_pure = self.X_train_pure.astype("float32")
        self.X_test_pure = self.X_test_pure.astype("float32")


    class TimeDistributedMultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self, num_heads, key_dim, **kwargs):
            super().__init__(**kwargs)
            self.num_heads = num_heads
            self.key_dim = key_dim
            self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        def call(self, inputs):
            # Use tf.shape for dynamic shape and inputs.shape for static shape
            shape = tf.shape(inputs)
            static_shape = inputs.shape
            batch_size, num_subsequences, subsequence_length = shape[0], static_shape[1], static_shape[2]
            features = static_shape[3]

            reshaped_inputs = tf.reshape(inputs, [-1, subsequence_length, features])

            # Apply multi-head attention to each subsequence individually
            attention_output = self.multi_head_attention(reshaped_inputs, reshaped_inputs)

            # Reshape the output back to the original input shape
            output_shape = [-1, num_subsequences, subsequence_length, features]
            return tf.reshape(attention_output, output_shape)

        def compute_output_shape(self, input_shape):
            return input_shape
        
        def get_config(self):
            config = super().get_config()
            config.update({
                'num_heads': self.num_heads,
                'key_dim': self.key_dim
            })
            return config

    class CustomEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self, monitor='val_loss', min_delta=0, patience=40, start_epoch=200, **kwargs):
            super().__init__(**kwargs)
            self.monitor = monitor
            self.min_delta = min_delta
            self.patience = patience
            self.start_epoch = start_epoch
            self.wait = 0
            self.stopped_epoch = 0
            self.best = float('inf')
            self.best_weights = None

        def on_train_begin(self, logs=None):
            # Reset the wait counter if training restarts.
            self.wait = 0
            self.stopped_epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(self.monitor)
            if current is None:
                return

            # Check if early stopping should start being considered (after start_epoch)
            if epoch >= self.start_epoch:
                if current < self.best - self.min_delta:
                    self.best = current
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        self.model.set_weights(self.best_weights)

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0:
                print(f'Epoch {self.stopped_epoch + 1}: early stopping')

    
    
    def build(self):
        
        """ Builds and compiles the model """
        with strategy.scope():
            
            self.model = tf.keras.Sequential()
            
            self.model.add(tf.keras.layers.BatchNormalization(input_shape=(self.X_train_noisy.shape[1:])))
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.cnn_filters_1, self.kernel_size, padding='same', activation='relu')))
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.cnn_filters_2, self.kernel_size, padding='same', activation='relu')))
                        
            
            self.model.add(tf.keras.layers.Dropout(0.20))
            
#            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=self.pool_size, padding='same'))) # For testing best 10 timesteps model, use this layer.
        
            self.model.add(tf.keras.layers.BatchNormalization())
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.cnn_filters_2, self.kernel_size, padding='same', activation='relu')))

#            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(16, self.kernel_size, padding='same', activation='relu')))
      
           
#            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())) # for testing simple and multi-head attention, comment this line. For best 10 timesteps model, use this layer.
        
            self.model.add(tf.keras.layers.BatchNormalization())
        
#            self.model.add(tf.keras.layers.TimeDistributed(self.AttentionLayer())) # for testing simple attention

            self.model.add(self.TimeDistributedMultiHeadAttention(num_heads=2, key_dim=32)) # for testing multi-head attention only
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())) # for testing multi-head attention only

            
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_1, activation = 'tanh', return_sequences=True)))            
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_2, activation = 'tanh', return_sequences=True)))
            
            self.model.add(tf.keras.layers.Dropout(0.20))
            
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_2, activation = 'tanh', return_sequences=True)))  
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_2, activation = 'tanh', return_sequences=True)))  
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
            
            optimizer = tf.keras.optimizers.Adam(lr=self.lr)
    #        self.model.compile(optimizer=optimizer,loss=self.fractal_tanimoto_loss,metrics=['accuracy'])
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    
    ##########
            self.model.summary()
        
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.model)
        
        self.train(checkpoint)
           
    def train(self, checkpoint):
        """Trains the model"""
        with strategy.scope():
            # initialize checkpoints
            dataset_name = "/fred/oz016/Chayan/GW-XNet/checkpoints/Saved_checkpoint"
            checkpoint_directory = "{}/tmp_{}".format(dataset_name, str(hex(random.getrandbits(32))))
            checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        
            # load best model with min validation loss
            if(self.train_from_checkpoint == True):
                checkpoint.restore(self.checkpoint_path)
                        
        
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30) # 0.9, 25
#            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, min_delta=0.0035,mode='auto',restore_best_weights=True) # 45
            
            custom_early_stopping = self.CustomEarlyStopping(monitor='val_loss', min_delta=0, patience=40, start_epoch=200)           
            callbacks_list=[reduce_lr, custom_early_stopping] 
        
            model_history = self.model.fit(self.X_train_noisy, self.X_train_pure, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.05, callbacks=callbacks_list)
        
            checkpoint.save(file_prefix=checkpoint_prefix)
        
            self.model.save("/fred/oz016/Chayan/GW-XNet/model/model_IMRPhenomXPHM_O3b_noise_10_timesteps_30Hz_mse_loss_larger_model_encoder_attention.h5")
#        print("Saved model to disk")

            self.plot_loss_curves(model_history.history['loss'], model_history.history['val_loss'])
        
    
    def plot_loss_curves(self, loss, val_loss):
    
        # summarize history for accuracy and loss
        plt.figure(figsize=(6, 4))
        plt.plot(loss, "r--", label="Loss on training data")
        plt.plot(val_loss, "r", label="Loss on validation data")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.savefig("/fred/oz016/Chayan/GW-XNet/evaluation/Accuracy_curve_IMRPhenomXPHM_10_tsteps_O3b_30Hz_larger_model_mse_loss_encoder_attention.png", dpi=200)
                                       
                                       
    def evaluate(self):
        """Predicts results for the test dataset"""
#        predictions = []
        predictions = self.model.predict(self.X_test_noisy)
        
        score = self.model.evaluate(self.X_test_noisy, self.X_test_pure, verbose=1)
        
        f1 = h5py.File('/fred/oz016/Chayan/GW-XNet/evaluation/results_IMRPhenomXPHM_10_tsteps_O3b_30Hz_larger_model_mse_loss_encoder_attention.hdf', 'w')
        f1.create_dataset('denoised_signals', data=predictions)
        f1.create_dataset('pure_signals', data=self.X_test_pure)
        
        print('\nAccuracy on test data: %0.2f' % score[1])
        print('\nLoss on test data: %0.2f' % score[0])

    
