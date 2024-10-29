# -*- coding: utf-8 -*-
"""Data Loader"""
import pandas as pd
import h5py
import sys
import numpy as np

class DataLoader:
    """Data Loader class"""
    
    def __init__(self, det, data):
        
        self.det = det
        self.data = data
       
    def load_data(self, data_config, dataset_type):
        """Loads dataset from path"""

        det_dict = {'Hanford': 'h1', 'Livingston': 'l1', 'Virgo': 'v1'}
        
        # Check training or testing data
        if(self.data == 'train'):

            path_dict = {
            'Original data': (data_config.path_train_original_1, data_config.path_train_original_2),
            'NRSurrogate injections': (data_config.path_train_NRSur, ),
            'High mass': (data_config.path_train_high_mass, ),
            'DeepClean': (data_config.path_train_DC, ),
            'IMRPhenomXPHM injections': (data_config.path_train, ),
            'Noise reconstruction': (data_config.path_train_noise_recons, )
        }
            
        elif(self.data == 'test'):

            path_dict = {
            'Original data': (data_config.path_test_original, ),
            'NRSurrogate injections': (data_config.path_test_NRSur, ),
            'High mass': (data_config.path_test_high_mass, ),
            'DeepClean': (data_config.path_test_DC, ),
            'IMRPhenomXPHM injections': (data_config.path_test_1, ),
            'Noise reconstruction': (data_config.path_test_noise_recons, )
        }

        # Check if the dataset type is valid and load data
        if dataset_type in path_dict:
            files = [h5py.File(path, 'r') for path in path_dict[dataset_type]]
            
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}. Please choose from {list(path_dict.keys())}.")

        # Initialize empty lists to store concatenated data
        all_strain = []
        all_signal = []
        all_noise_strain = []
        all_noise_signal = []

        for f in files:
            if hasattr(f, 'injection_samples'):
                strain_data = f['injection_samples'][det_dict[self.det] + '_strain'][()]
                signal_data = f['injection_parameters'][det_dict[self.det] + '_signal_whitened'][()]
                noise_strain_data = f['noise_samples']['l1_strain'][()]
                noise_signal_data = f['noise_parameters']['l1_signal'][()]

            else:
                if(self.data == 'train'):
                    strain_data = f['strain'][0:50000][()]
                    signal_data = f['noise'][0:50000][()]
                    noise_strain_data = f['noise'][50000:75000][()]
                    noise_signal_data = f['noise'][50000:75000][()]
                    
                elif(self.data == 'test'):
                    strain_data = f['strain'][0:1500][()]
                    signal_data = f['noise'][0:1500][()]
                    noise_strain_data = f['noise'][1500:2000][()]
                    noise_signal_data = f['noise'][1500:2000][()]
                    
            all_strain.append(strain_data)
            all_signal.append(signal_data)
            all_noise_strain.append(noise_strain_data)
            all_noise_signal.append(noise_signal_data)

        if(self.data == 'train'):
            # Concatenate all the data from the list
            strain = np.concatenate(all_strain, axis=0)
            signal = np.concatenate(all_signal, axis=0)
            noise_strain = np.concatenate(all_noise_strain, axis=0)
            noise_signal = np.concatenate(all_noise_signal, axis=0)

            # Concatenate the arrays
            concatenated_array = np.concatenate((signal, noise_signal))

            # Shuffle the indices
            shuffled_indices = np.random.permutation(len(concatenated_array))

            # Use the shuffled indices to create a new shuffled array
            shuffled_array_signal = concatenated_array[shuffled_indices]

            # Concatenate the arrays
            concatenated_array_strain = np.concatenate((strain, noise_strain))

            shuffled_array_strain = concatenated_array_strain[shuffled_indices]
    
            strain = shuffled_array_strain
            signal = shuffled_array_signal

        elif(self.data == 'test'):

            # Concatenate all the data from the list
            strain = np.concatenate(all_strain, axis=0)
            signal = np.concatenate(all_signal, axis=0)
            noise_strain = np.concatenate(all_noise_strain, axis=0)
            noise_signal = np.concatenate(all_noise_signal, axis=0)

            # Concatenate the arrays
            concatenated_array_signal = np.concatenate((signal, noise_signal))

            # Concatenate the arrays
            concatenated_array_strain = np.concatenate((strain, noise_strain))
    
            strain = concatenated_array_strain
            signal = concatenated_array_signal

        # Close the files
        for f in files:
            f.close()
        
        return strain, signal
    
    