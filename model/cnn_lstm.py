# -*- coding: utf-8 -*-
"""CNN-LSTM Model"""

'''
 * Copyright (C) 2024 Chayan Chatterjee <chayan.chatterjee@vanderbilt.edu>
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
'''

####################################################### IMPORTS #################################################################

# Internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader

# External
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
import os
from matplotlib import pyplot as plt

plt.switch_backend('agg')

tfd = tfp.distributions
strategy = tf.distribute.MirroredStrategy()

#################################################################################################################################

class CNN_LSTM(BaseModel):
    """CNN-LSTM Model Class"""
    
    def __init__(self, config):
        super().__init__(config)
        self._set_params()

    def _set_params(self):
        """Set model parameters from config"""
        self.n_samples = self.config.train.n_samples_per_signal
        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epochs
        self.det = self.config.train.detector
        self.depth = self.config.train.depth
        self.lr = self.config.model.layers.learning_rate
        self.timesteps = self.config.model.timesteps
        self.model_save_path = self.config.model.model_save_path
        self.results_save_path = self.config.model.results_save_path

        self.cnn_filters_1 = self.config.model.layers.CNN_layer_1
        self.cnn_filters_2 = self.config.model.layers.CNN_layer_2
        self.lstm_1 = self.config.model.layers.LSTM_layer_1
        self.lstm_2 = self.config.model.layers.LSTM_layer_2
        self.kernel_size = self.config.model.layers.kernel_size
        self.pool_size = self.config.model.layers.pool_size
        self.dropout = self.config.model.layers.dropout
        self.num_heads = self.config.model.layers.num_heads_MHA
        self.key_dim = self.config.model.layers.key_dim_MHA
        self.train_from_checkpoint = self.config.train.train_from_checkpoint
        self.checkpoint_path = self.config.train.checkpoint_path

    def load_data(self):
        """Loads and preprocesses data"""
        self.strain_train, self.signal_train = DataLoader(self.det, 'train').load_data(self.config.data)
        self.strain_test, self.signal_test = DataLoader(self.det, 'test').load_data(self.config.data)

        self.strain_train = self._preprocess_data(self.strain_train, self.n_samples)
        self.strain_test = self._preprocess_data(self.strain_test, self.n_samples)

        self.signal_train = self.signal_train / 100.0
        self.signal_test = self.signal_test / 100.0

        self.X_train_noisy, self.X_train_pure = self.reshape_sequences(self.strain_train.shape[0], self.strain_train, self.signal_train)
        self.X_test_noisy, self.X_test_pure = self.reshape_sequences(self.strain_test.shape[0], self.strain_test, self.signal_test)

        self.reshape_and_print()

    def _preprocess_data(self, data, samples):
        """Scales the amplitudes of the signals to lie between -1 and 1"""
        new_array = []
        for i in range(data.shape[0]):
            dataset = data[i]
            if dataset.max() != 0.0 and dataset.min() != 0.0:
                maximum = np.max(dataset)
                minimum = np.abs(np.min(dataset))
                dataset = np.where(dataset > 0, dataset / maximum, dataset / minimum)
            new_array.append(dataset)
        return new_array

    def split_sequence(self, sequence_noisy, sequence_pure, n_steps):
        """Splits a univariate sequence into samples"""
        X, y = [], []
        for i in range(len(sequence_noisy)):
            end_ix = i + n_steps
            if end_ix > len(sequence_noisy) - 1:
                break
            seq_x, seq_y = sequence_noisy[i:end_ix], sequence_pure[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def reshape_sequences(self, num, data_noisy, data_pure):
        """Reshapes data into overlapping sequences"""
        n_steps = self.timesteps
        arr_noisy, arr_pure = [], []

        for i in range(num):
            X_noisy, X_pure = data_noisy[i], data_pure[i]
            X_noisy = np.pad(X_noisy, (n_steps, n_steps), 'constant', constant_values=(0, 0))
            X_pure = np.pad(X_pure, (n_steps, n_steps), 'constant', constant_values=(0, 0))
            X, y = self.split_sequence(X_noisy, X_pure, n_steps)
            arr_noisy.append(X)
            arr_pure.append(y)

        return np.asarray(arr_noisy), np.asarray(arr_pure)

    def reshape_and_print(self):
        """Reshapes arrays to fit into Keras model and prints their shapes"""
        self.X_train_noisy = self.X_train_noisy[..., None]
        self.X_test_noisy = self.X_test_noisy[..., None]
        self.X_train_pure = self.X_train_pure[..., None]
        self.X_test_pure = self.X_test_pure[..., None]

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
            shape = tf.shape(inputs)
            static_shape = inputs.shape
            batch_size, num_subsequences, subsequence_length = shape[0], static_shape[1], static_shape[2]
            features = static_shape[3]
            reshaped_inputs = tf.reshape(inputs, [-1, subsequence_length, features])
            attention_output = self.multi_head_attention(reshaped_inputs, reshaped_inputs)
            output_shape = [-1, num_subsequences, subsequence_length, features]
            return tf.reshape(attention_output, output_shape)

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "num_heads": self.num_heads, "key_dim": self.key_dim}

    @staticmethod
    def negloglik(y, rv_y):
        """Negative log likelihood loss function"""
        return -rv_y.log_prob(y)

    def build(self):
        """Builds and compiles the model"""
        with strategy.scope():
            input_shape = self.X_train_noisy.shape[1:]
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.BatchNormalization()(inputs)

            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.cnn_filters_1, self.kernel_size, padding='same', activation='relu'))(x)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.cnn_filters_2, self.kernel_size, padding='same', activation='relu'))(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = self.TimeDistributedMultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)(x)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_1, activation='tanh', return_sequences=True))(x)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_2, activation='tanh', return_sequences=True))(x)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_2, activation='tanh', return_sequences=True))(x)

            x = tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(1))(x)
            outputs = tfp.layers.IndependentNormal(1)(x)

            self.model = tf.keras.Model(inputs, outputs)
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            self.model.compile(optimizer=optimizer, loss=self.negloglik, metrics=['accuracy'])

            self.model.summary()

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.model)
        self.train(checkpoint)

    def load_saved_model(self):
        """Loads a saved model"""
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            self.model = tf.keras.models.load_model(self.checkpoint_path, custom_objects={
                'TimeDistributedMultiHeadAttention': self.TimeDistributedMultiHeadAttention,
                'IndependentNormal': tfp.layers.IndependentNormal,
                'negloglik': self.negloglik
            })
            self.model.compile(optimizer=optimizer, loss=self.negloglik, metrics=['accuracy'])
            self.model.summary()
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.model)
            self.train(checkpoint)

    def train(self, checkpoint):
        """Trains the model"""
        with strategy.scope():
            dataset_name = "/workspace/chayan_ligo/GW-Denoiser/checkpoints/Saved_checkpoint"
            checkpoint_directory = "{}/tmp_{}".format(dataset_name, str(hex(random.getrandbits(32))))
            checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        
            if self.train_from_checkpoint:
                checkpoint.restore(self.checkpoint_path)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=25)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
            callbacks_list = [reduce_lr, early_stopping]

            model_history = self.model.fit(
                self.X_train_noisy, self.X_train_pure,
                epochs=self.epochs, batch_size=self.batch_size,
                validation_split=0.15, callbacks=callbacks_list
            )

            checkpoint.save(file_prefix=checkpoint_prefix)
            self.model.save(self.model_save_path)

            self.plot_loss_curves(model_history.history['loss'], model_history.history['val_loss'])

    def plot_loss_curves(self, loss, val_loss):
        """Plots loss curves"""
        plt.figure(figsize=(6, 4))
        plt.plot(loss, "r--", label="Loss on training data")
        plt.plot(val_loss, "r", label="Loss on validation data")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(self.results_save_path, dpi=400)

    def predict_with_uncertainty(self, x_test):
        """Predicts with uncertainty"""
        distribution = self.model(x_test[np.newaxis, ...])
        mean_preds = distribution.mean().numpy().squeeze()
        std_preds = distribution.stddev().numpy().squeeze()
        lower_bound = mean_preds - 2 * std_preds
        upper_bound = mean_preds + 2 * std_preds
        return mean_preds, lower_bound, upper_bound

    def evaluate(self):
        """Evaluates the model on test data"""
        mean, m2sd, p2sd = [], [], []
        for i in range(self.X_test_noisy.shape[0]):
            mean_preds, lower_bound_preds, upper_bound_preds = self.predict_with_uncertainty(self.X_test_noisy[i])
            mean.append(mean_preds)
            m2sd.append(lower_bound_preds)
            p2sd.append(upper_bound_preds)

        mean = np.array(mean)
        m2sd = np.array(m2sd)
        p2sd = np.array(p2sd)

        with h5py.File(self.results_save_path, 'w') as f1:
            f1.create_dataset('mean_signals', data=mean)
            f1.create_dataset('pure_signals', data=self.signal_test)
            f1.create_dataset('m2sd_signals', data=m2sd)
            f1.create_dataset('p2sd_signals', data=p2sd)

        print('Results file created!')