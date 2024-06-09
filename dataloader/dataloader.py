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
       
    def load_data(self, data_config):
        """Loads dataset from path"""
        
        # Check training or testing data
        if(self.data == 'train'):
 #           df1 = h5py.File(data_config.path_train, 'r')
 #           df2 = h5py.File(data_config.path_train_DC, 'r')
            df1 = h5py.File(data_config.path_train_original_1, 'r')
            df2 = h5py.File(data_config.path_train_original_2, 'r')
 #           df1 = h5py.File(data_config.path_train_NRSur, 'r')
 #           df1 = h5py.File(data_config.path_train_finetune, 'r')
 #           df1 = h5py.File(data_config.path_train_high_mass, 'r')
            
        elif(self.data == 'test'):
#             df1 = h5py.File(data_config.path_test_1, 'r')
#             df1 = h5py.File(data_config.path_test_DC, 'r')
             df1 = h5py.File(data_config.path_test_original, 'r')
             df2 = h5py.File(data_config.path_test_DeepClean, 'r')
#             df1 = h5py.File(data_config.path_test_NRSur, 'r')
#             df1 = h5py.File(data_config.path_test_finetune, 'r')
#            df1 = h5py.File(data_config.path_test_high_mass, 'r')


#            df2 = h5py.File(data_config.path_test_2, 'r')
#            df3 = h5py.File(data_config.path_test_3, 'r')
#            df4 = h5py.File(data_config.path_test_4, 'r')
            
            
        
        # Obtain data for a given detector
        if(self.det == 'Hanford'):
            strain = df['injection_samples']['h1_strain'][()]
            signal = df['injection_parameters']['h1_signal'][()]
            
        elif(self.det == 'Livingston'):
            
            if(self.data == 'train'):
#                strain = df1['injection_samples']['l1_strain'][0:10000][()]                
#                signal = df1['injection_parameters']['l1_signal_whitened'][0:10000][()]
                
                strain_1 = df1['injection_samples']['l1_strain'][()]                
                signal_1 = df1['injection_parameters']['l1_signal_whitened'][()]

                strain_2 = df2['injection_samples']['l1_strain'][()]                
                signal_2 = df2['injection_parameters']['l1_signal_whitened'][()]

                strain = np.concatenate([strain_1, strain_2], axis=0)
                signal = np.concatenate([signal_1, signal_2], axis=0)
            
            
            if(self.data == 'test'):
                strain_1 = df1['injection_samples']['l1_strain'][()]
                strain_2 = df2['injection_samples']['l1_strain'][()]
#                strain_3 = df3['injection_samples']['l1_strain'][0:25][()]
#                strain_4 = df4['injection_samples']['l1_strain'][0:25][()]            
            
                signal_1 = df1['injection_parameters']['l1_signal_whitened'][()]
                signal_2 = df2['injection_parameters']['l1_signal_whitened'][()]
#                signal_3 = df3['injection_parameters']['l1_signal_whitened'][0:25][()]
#                signal_4 = df4['injection_parameters']['l1_signal_whitened'][0:25][()]

                strain = np.concatenate([strain_1, strain_2], axis=0)
                signal = np.concatenate([signal_1, signal_2], axis=0)

    #            strain = strain_1
    #            signal = signal_1
            
#                strain = np.concatenate([strain_1,strain_2,strain_3,strain_4])
#                signal = np.concatenate([signal_1,signal_2,signal_3,signal_4])                  
                
            noise_samples = df1['noise_samples']['l1_strain'][()]
            
            if(noise_samples.shape is not None):
                
                if(self.data =='train'):
                    
                    noise_strain_1 = df1['noise_samples']['l1_strain'][()]
                    noise_signal_1 = df1['noise_parameters']['l1_signal'][()]

                    noise_strain_2 = df2['noise_samples']['l1_strain'][()]
                    noise_signal_2 = df2['noise_parameters']['l1_signal'][()] 

                    noise_strain = np.concatenate([noise_strain_1, noise_strain_2], axis=0)
                    noise_signal = np.concatenate([noise_signal_1, noise_signal_2], axis=0)
                    
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
                                        
                
                if(self.data == 'test'):
                    
                    noise_strain_1 = df1['noise_samples']['l1_strain'][()]
                    noise_signal_1 = df1['noise_parameters']['l1_signal'][()]
                
                    noise_strain_2 = df2['noise_samples']['l1_strain'][()]
                    noise_signal_2 = df2['noise_parameters']['l1_signal'][()]
                
#                    noise_strain_3 = df3['noise_samples']['l1_strain'][()]
#                    noise_signal_3 = df3['noise_parameters']['l1_signal'][()]
                
#                    noise_strain_4 = df4['noise_samples']['l1_strain'][()]
#                    noise_signal_4 = df4['noise_parameters']['l1_signal'][()]

#                    noise_strain = noise_strain_1
#                    noise_signal = noise_signal_1
                
                    noise_strain = np.concatenate([noise_strain_1,noise_strain_2])
                    noise_signal = np.concatenate([noise_signal_1,noise_signal_2])
                    
                    # Concatenate the arrays
                    concatenated_array_signal = np.concatenate((signal, noise_signal))

                    # Concatenate the arrays
                    concatenated_array_strain = np.concatenate((strain, noise_strain))
    
                    strain = concatenated_array_strain
                    signal = concatenated_array_signal
            
#            elif(self.data == 'test'):
#                strain = df['injection_samples']['l1_strain'][0:2000][()]
#                signal = df['injection_parameters']['l1_signal_whitened'][0:2000][()]
            
        elif(self.det == 'Virgo'):
            strain = df['injection_samples']['v1_strain'][()]
            signal = df['injection_parameters']['v1_signal'][()]
            
        else:
            sys.exit('Detector not available. Quitting.')
        
        df1.close()
        
        return strain, signal
    
    