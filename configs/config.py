# -*- coding: utf-8 -*-
# Author: Chayan Chatterjee
# Last modified: 9th June 2024

"""Model config in json format"""

CFG = {
    "data": {
        "path_train": "/workspace/chayan_ligo/BBH_sample_files/O3b_IMBH_train_IMRPhenomXPHM_signal_noise_30Hz_new.hdf", # has noise+glitches
        "path_train_DC": "/workspace/chayan_ligo/BBH_sample_files/DeepClean_IMRPhenomXPHM_train.hdf",

        "path_train_original_1": "/workspace/chayan_ligo/BBH_sample_files/Original_data_GW190521_params_train.hdf",
        "path_train_original_2": "/workspace/chayan_ligo/BBH_sample_files/Original_data_GW190521_params_train_1.hdf",

        "path_train_finetune": "/workspace/chayan_ligo/BBH_sample_files/O3b_train_IMRPhenomXPHM_SNR-8to30_mass_100to250+glitch_finetune_data.hdf",
        
        "path_train_noise": "/workspace/chayan_ligo/O3b_train_IMRPhenomXPHM_noise_60k_30Hz_new.hdf",

        "path_train_high_mass": "/workspace/chayan_ligo/BBH_sample_files/O3b_train_IMRPhenomXPHM_SNR-8to30_mass_80to400.hdf",
        "path_train_NRSur": "/workspace/chayan_ligo/BBH_sample_files/O3b_train_NRSur7dq4_SNR-8to30_mass_100to250.hdf",


        "path_test_1": "/workspace/chayan_ligo/BBH_sample_files/O3b_IMBH_test_IMRPhenomXPHM_signal_noise_30Hz_new.hdf", # has only noise, no glitches.
        "path_test_DC": "/workspace/chayan_ligo/BBH_sample_files/DeepClean_IMRPhenomXPHM_test.hdf",

        "path_test_original": "/workspace/chayan_ligo/BBH_sample_files/Original_data_GW190521_params_test.hdf",
        "path_test_DeepClean": "/workspace/chayan_ligo/BBH_sample_files/DeepClean_data_GW190521_params_test.hdf",

        "path_test_finetune": "/workspace/chayan_ligo/BBH_sample_files/O3b_test_IMRPhenomXPHM_SNR-8to30_mass_100to250+glitch_finetune_data.hdf",
        "path_test_glitch": "/workspace/chayan_ligo/BBH_sample_files/glitch_data_test.hdf",

        "path_test_high_mass": "/workspace/chayan_ligo/BBH_sample_files/O3b_test_IMRPhenomXPHM_SNR-8to30_mass_80to400.hdf",
        "path_test_NRSur": "/workspace/chayan_ligo/BBH_sample_files/O3b_test_NRSur7dq4_SNR-8to30_mass_100to250.hdf"
       
    },
    "train": {
        "dataset_type": 'IMRPhenomXPHM injections',
        "detector": 'Livingston', # 'Hanford'/'Livingston'/'Virgo'
        "n_samples_per_signal": 2048,
        "batch_size": 512,
        "epochs": 100,
        "train_from_checkpoint": False,
        "checkpoint_path": '/workspace/chayan_ligo/Waveform_reconstruction/AWaRe/checkpoints/Saved_checkpoint/tmp_0xb64ed658/ckpt-1', # if train_from_checkpoint == True
        "optimizer": {
            "type": "adam"
        },
    },
    "model": {
        "timesteps": 10,
        "model_save_path": '/workspace/chayan_ligo/Waveform_reconstruction/AWaRe/model/Saved_models/Trained_model.h5',
        "results_save_path": '/workspace/chayan_ligo/Waveform_reconstruction/AWaRe/evaluation/Saved_results_files/Loss_curve.png',

# For original model        
        
        "layers": {
            "CNN_layer_1": 64,
            "CNN_layer_2": 32,
            "LSTM_layer_1": 32,
            "LSTM_layer_2": 32,
            "LSTM_layer_3": 32,
            "Dropout": 0.22,
            "Output_layer": 1,
            "kernel_size": 3,
            "pool_size": 2,
            "num_heads_MHA": 8,
            "key_dim_MHA": 16,
            "learning_rate": 1e-3
        },
    }
}
