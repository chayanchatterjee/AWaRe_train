# -*- coding: utf-8 -*-
# Author: Chayan Chatterjee
# Last modified: 5th February 2024

"""Model config in json format"""

CFG = {
    "data": {
        "path_train": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_train_IMRPhenomXPHM_signal_noise_30Hz.hdf",
#        "path_train": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_train_IMRPhenomXPHM_signal_noise_SNR-8to30.hdf",
#        "path_train": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_train_IMRPhenomXPHM_signal_noise_SNR-8to40_mchirp_all.hdf",
#        "path_train": "/fred/oz016/Chayan/BBH_sample_files/O3b_train_IMRPhenomXPHM_SNR-20to40_mchirp_all_whitening_fixed.hdf",
# For training Bayesian neural network model        
#        "path_train": "/fred/oz016/Chayan/BBH_sample_files/default_IMBH_train_IMRPhenomXPHM_signal_noise_SNR-20to40_mchirp_all_100k.hdf",


        "path_test_1": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_signal_noise_30Hz.hdf",

#        "path_test_1": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_signal_noise_SNR-8to9.hdf",
#        "path_test_2": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_signal_noise_SNR-9to20.hdf",
#        "path_test_3": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_signal_noise_SNR-20to30.hdf",
#        "path_test_4": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_signal_noise_SNR-30to40.hdf",

#        "path_test_1": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_SNR-8to10_max_mchip_10.hdf",
#        "path_test_2": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_SNR-10to20_max_mchip_10.hdf",
#        "path_test_3": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_SNR-20to30_max_mchip_10.hdf",
#        "path_test_4": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_SNR-30to40_max_mchip_10.hdf",
        
#        "path_test_1": "/fred/oz016/Chayan/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_signal_noise_SNR-8to40_mchirp_all.hdf",
#        "path_test_1": "/fred/oz016/Chayan/BBH_sample_files/O3b_test_IMRPhenomXPHM_SNR-20to40_mchirp_all_whitening_fixed.hdf",
        
# For testing Bayesian neural network model        
#        "path_test_1": "/fred/oz016/Chayan/BBH_sample_files/default_IMBH_test_IMRPhenomXPHM_signal_noise_SNR-20to40_mchirp_all.hdf",


        
        
    },
    "train": {
        "num_training_samples": 60000,
        "num_test_samples": 2100,
        "detector": 'Livingston', # 'Hanford'/'Livingston'/'Virgo'
        "n_samples_per_signal": 2048,
        "batch_size": 1024,
        "epoches": 500,
        "depth": 0,
        "train_from_checkpoint": False,
        "checkpoint_path": '/fred/oz016/Chayan/GW-Denoiser/checkpoints/Saved_checkpoint/tmp_0xb64ed658/ckpt-1', # if train_from_checkpoint == True
        "optimizer": {
            "type": "adam"
        },
    },
    "model": {
#        "input": [516,10],
        "timesteps": 25,

# For original model        
        
        "layers": {
            "CNN_layer_1": 64,
            "CNN_layer_2": 32,
            "LSTM_layer_1": 32,
            "LSTM_layer_2": 32,
            "LSTM_layer_3": 32,
            "Output_layer": 1,
            "kernel_size": 3,
            "pool_size": 2,
            "learning_rate": 1e-3
        },

# For timedistributed dilated CNN model
        
#        "layers": {
#            "CNN_layer_dilated":128,
#            "CNN_layer_1": 64,
#            "CNN_layer_2": 32,
#            "CNN_layer_3": 16,
#            "LSTM_layer_1": 64,
#            "LSTM_layer_2": 64,
#            "LSTM_layer_3": 64,
#            "LSTM_layer_4": 64,
#            "Output_layer": 1,
#            "kernel_size": 3,
#            "pool_size": 2,
#            "learning_rate": 1e-4
            
#        },
        
# For dilated CNN-LSTM model.
        
#        "layers": {
#            "CNN_layer_dilated":128,
#            "CNN_layer_1": 256,
#            "CNN_layer_2": 128,
#            "CNN_layer_3": 128,
#            "LSTM_layer_1": 128,
#            "LSTM_layer_2": 64,
#            "LSTM_layer_3": 128,
#            "LSTM_layer_4": 256,
#            "Output_layer": 1,
#            "kernel_size": 3,
#            "pool_size": 2,
#            "learning_rate": 1e-4
#        },
    }
}
