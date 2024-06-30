from __future__ import print_function
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import display, HTML
import tensorflow as tf
import h5py
import argparse
import sys
import os
from pycbc.filter import match, overlap
from pycbc.types.timeseries import TimeSeries, FrequencySeries
from scipy.fft import fft, ifft
import matplotlib as mpl
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import matplotlib_latex_bridge as mlb
mlb.setup_page(**mlb.formats.article_letterpaper_10pt_singlecolumn)
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def Overlap_calc(hp,sp,psd):
    f_low = 20
    m, i = match(hp, sp, psd=psd, low_frequency_cutoff=f_low)
    o = overlap(hp, sp, psd=psd, low_frequency_cutoff=f_low)
    return m

def _preprocess_data(data, num, samples):
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
    return new_array, maximum, minimum


# Split a univariate sequence into samples
#def split_sequence(sequence_noisy,sequence_pure,n_steps):
def split_sequence(sequence_noisy,n_steps):
    X = [] 
#    y = []
    for i in range(len(sequence_noisy)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence_noisy)-1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence_noisy[i:end_ix] 
#        seq_y = sequence_pure[end_ix]
        X.append(seq_x)
#        y.append(seq_y)
#    return array(X), array(y)
    return np.array(X)
    
    
#def reshape_sequences(self, num, data_noisy, data_pure):
def reshape_sequences(num, data_noisy, n_steps):
    n_steps = n_steps
    arr_noisy = []
#    arr_pure = []
        
    for i in range(num):
        X_noisy = data_noisy[i]
#        X_pure = data_pure[i]
        X_noisy = np.pad(X_noisy, (n_steps, n_steps), 'constant', constant_values=(0, 0))
#        X_pure = np.pad(X_pure, (n_steps, n_steps), 'constant', constant_values=(0, 0))
        # split into samples
#        X, y = self.split_sequence(X_noisy, X_pure, n_steps)
        X = split_sequence(X_noisy, n_steps)
        arr_noisy.append(X)
#        arr_pure.append(y)
    
    arr_noisy = np.asarray(arr_noisy)
#    arr_pure = np.asarray(arr_pure)
        
#    return arr_noisy, arr_pure
    return arr_noisy

class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
            self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
            super().build(input_shape)
        
        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            output = x * a
            return output
        
        def compute_output_shape(self,input_shape):
            return (input_shape[0],input_shape[-1])

        def get_config(self):
            return super().get_config()
    
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




class FractalTanimotoLoss(tf.keras.losses.Loss):
        def __init__(lr, base_lr=2e-3, depth=0, smooth=1e-6, **kwargs):
            depth = depth
            learning_rate = lr
            base_lr = base_lr
            smooth = smooth
            super().__init__(**kwargs)
    
        def inner_prod(self, y, x):
            prod = y*x
            prod = K.sum(prod, axis=1)
        
            return prod
    
        def tnmt_base(x, y, scale):

            tpl  = inner_prod(y,x)
            tpp  = inner_prod(y,y)
            tll  = inner_prod(x,x)


            num = tpl + self.smooth
            denum = 0.0
            result = 0.0
            for d in range(depth):
                a = 2.**d
                b = -(2.*a-1.)

                denum = denum + tf.math.reciprocal( a*(tpp+tll) + b *tpl + smooth)
                
            result =  num * denum * scale
            
            return  result*scale
        
        def call(y_true, y_pred):
            
            if(learning_rate < base_lr):
                depth = depth + 5
                
            depth = depth+1
            scale = 1./len(range(depth))
            
            l1 = K.mean(K.square(y_pred - y_true),axis=-1)
            result = tnmt_base(y_true, y_pred, scale)
        
            return  l1 - 0.01*result
        
        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "depth": depth}   
    

class NegativeLogLikelihood:
    def __call__(self, y, rv_y):
        return -rv_y.log_prob(y)


def evaluate_model(strain, obs_run, event_name, unc):
    
    event_list_old_model = ['GW170608', 'GW151226']
    
    strain = strain[None,:]
    strain, maximum, minimum = _preprocess_data(strain, 1, 2048)
    strain = np.array(strain)

    lower_bound = None
    upper_bound = None
    # Instantiate the loss function
    negloglik = NegativeLogLikelihood()

    if(obs_run == 'O3'):
#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_dilated_CNN_25_timesteps.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss})
#        num_steps = 25

#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_30Hz_mse_loss_larger_model_MHA.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss, 'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention})
#        num_steps = 15

############ Model for uncertainty estimation paper ###################
#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_uncertainty.h5', custom_objects={'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention, 'IndependentNormal': tfp.layers.IndependentNormal,'negloglik': negloglik})
#        num_steps = 15
#######################################################################
        
        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_uncertainty_amplitudes_scaled_100.h5', custom_objects={'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention, 'IndependentNormal': tfp.layers.IndependentNormal,'negloglik': negloglik})
        num_steps = 10

#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_NRSur7dq4_O3b_noise_15_timesteps_uncertainty.h5', custom_objects={'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention, 'IndependentNormal': tfp.layers.IndependentNormal,'negloglik': negloglik})
#        num_steps = 15

#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_uncertainty_amplitudes_scaled_100.h5', custom_objects={'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention, 'IndependentNormal': tfp.layers.IndependentNormal,'negloglik': negloglik})
#        num_steps = 10

        
    elif((obs_run == 'O2') and (event_name not in event_list_old_model)):
        
#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/trained_model_test_IMBH_HM_O3b_noise_mass_corrected.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss})
#        num_steps = 10
        
#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_30Hz_mse_loss_larger_model_MHA.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss, 'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention})
#        num_steps = 15

#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_uncertainty.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss, 'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention})
#        num_steps = 15

        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_NRSur7dq4_O3b_noise_15_timesteps_uncertainty.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss, 'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention})
        num_steps = 15


    elif((obs_run == 'O1') and (event_name not in event_list_old_model)):
        
#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/trained_model_test_IMBH_HM_O3b_noise_mass_corrected.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss})
#        num_steps = 10

        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_30Hz_mse_loss_larger_model_MHA.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss, 'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention})
        num_steps = 15
        
#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_uncertainty.h5', custom_objects={'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention, 'IndependentNormal': tfp.layers.IndependentNormal,'negloglik': negloglik})
#        num_steps = 15
        
#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_old_model_4_timesteps_mchirp_all.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss})
#        num_steps = 4
        
    elif event_name in event_list_old_model:
        
#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/trained_model_test_IMBH_HM_O3b_noise.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss})
#        num_steps = 10

        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_30Hz_mse_loss_larger_model_MHA.h5', custom_objects={'FractalTanimotoLoss': FractalTanimotoLoss, 'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention})
        num_steps = 15

    
    X_test_noisy = reshape_sequences(1, strain, num_steps)

    # Reshape arrays to fit into Keras model
    X_test_noisy = X_test_noisy[:,:,:,None]
    X_test_noisy = X_test_noisy.astype("float32")
    
    decoded_signals = model.predict(X_test_noisy, batch_size=1)

    if(unc == True):

        distribution = model(X_test_noisy)
    
        # Extract mean and standard deviation from the distribution
        mean_preds = distribution.mean()
        std_preds = distribution.stddev()
 
        mean_preds = mean_preds.numpy().squeeze()
        std_preds = std_preds.numpy().squeeze()

        # Calculate the upper and lower bounds of the 2-standard deviation interval
        lower_bound = mean_preds - 1.645 * std_preds
        upper_bound = mean_preds + 1.645 * std_preds

        mean_preds, maximum, minimum = _preprocess_data(mean_preds[None,:], 1, 2048)

        new_lower = []
        for value in lower_bound:
            if value > 0.0:
                value = value/maximum # check which maximum and minimum to use
            elif value < 0.0:
                value = value/minimum
            new_lower.append(value)
        
        new_upper = []
        for value in upper_bound:
            if value > 0.0:
                value = value/maximum
            elif value < 0.0:
                value = value/minimum
            new_upper.append(value)  
        
        mean_preds = np.array(mean_preds)
        lower_bound = np.array(new_lower)
        upper_bound = np.array(new_upper)

    return mean_preds.squeeze(), lower_bound, upper_bound, strain


def get_pe_data(event_name):
    
    # Define the base directory and file name
    base_dir = '/fred/oz016/Chayan/samplegen_old/output'
    file_name = event_name+'_maxl_li_signal_time_L1.dat'

    # Construct the full file path
    full_file_path = os.path.join(base_dir, file_name)
    
    with open(full_file_path, 'r') as file:
        time = []
        amp = []
        for line in file:
            # Split the line into parts based on whitespace or specific delimiter
            parts = line.split()  # Use split('\t') for tab-delimited files
        
            try:
                col1 = float(parts[0])
                col2 = float(parts[1])
                time.append(col1)
                amp.append(col2)
            except ValueError:
                print(f"Warning: Can't convert line to floats: {line.strip()}")

    time = np.array(time)
    amp = np.array(amp)

    return amp, time

def get_psd(event_name):
    
    # Define the base directory and file name
    base_dir = '/fred/oz016/Chayan/samplegen_old/output'
    file_name = event_name+'_L1_psd.dat'

    # Construct the full file path
    full_file_path = os.path.join(base_dir, file_name)
    
    with open(full_file_path, 'r') as file:
        freq = []
        psd = []
        for line in file:
            # Split the line into parts based on whitespace or specific delimiter
            parts = line.split()  # Use split('\t') for tab-delimited files
        
            try:
                col1 = float(parts[0])
                col2 = float(parts[1])
                freq.append(col1)
                psd.append(col2)
            except ValueError:
                print(f"Warning: Can't convert line to floats: {line.strip()}")
            
        freq = np.array(freq)
        psd = np.array(psd)

    return psd


def get_PE_reconst(reconst_cwb, gps_time, event_name):

    sampling_rate = 1024.0
    seconds_before_event = 0.8
    seconds_after_event = 0.2                                 
                                 
    pe_reconst, time = get_pe_data(event_name)   

    index_gps_time = np.where((time > gps_time-0.001) & (time < gps_time+0.001))[0][0]
    lower_lim = round(index_gps_time -(sampling_rate*seconds_before_event))
    upper_lim = round(index_gps_time +(sampling_rate*seconds_after_event))
     

    pe_reconst = pe_reconst[lower_lim:upper_lim][None,:]
    reconstructed_pe, maximum, minimum = _preprocess_data(pe_reconst,1,1024)
        
    new_lower = []
    for value in reconst_cwb['amp_post_lower_90_cr,'].values[lower_lim:upper_lim]:
        if value > 0.0:
            value = value/maximum # check which maximum and minimum to use
        elif value < 0.0:
            value = value/minimum
        new_lower.append(value)
        
    new_upper = []
    for value in reconst_cwb['amp_post_upper_90_cr'].values[lower_lim:upper_lim]:
        if value > 0.0:
            value = value/maximum
        elif value < 0.0:
            value = value/minimum
        new_upper.append(value)  

    if(event_name == 'GW190602_175927'):
        reconstructed_pe = reconstructed_pe[::2]
        new_lower = new_lower[::2]
        new_upper = new_upper[::2]

    return reconstructed_pe, new_lower, new_upper


def get_BW_reconst(reconst_BW, gps_time, event_name):

    sample_rate = 1024
    if(event_name == 'GW190412'):
        seg_length = 8.0
        times = reconst_BW['time'].values + (gps_time - seg_length/2)
        times = times[::2]
        BW_reconst = reconst_BW['median'].values[::2]
        lower = reconst_BW['90%_lower'].values[::2]
        upper = reconst_BW['90%_upper'].values[::2]
    
    elif(event_name == 'GW190517_055101'):
        seg_length = 4.0
        times = reconst_BW['time'].values + (gps_time - seg_length/2)
        BW_reconst = reconst_BW['median'].values
        lower = reconst_BW['90%_lower'].values
        upper = reconst_BW['90%_upper'].values

    elif(event_name == 'GW190602_175927'):
        seg_length = 4.0
        times = reconst_BW['time'].values + (gps_time - seg_length/2)
        BW_reconst = reconst_BW['median'].values
        lower = reconst_BW['90%_lower'].values
        upper = reconst_BW['90%_upper'].values
        sample_rate = 512

    else:
        seg_length = 8.0
        times = reconst_BW['time'].values + (gps_time - seg_length/2)
        BW_reconst = reconst_BW['median'].values
        lower = reconst_BW['90%_lower'].values
        upper = reconst_BW['90%_upper'].values

    trigger_index = np.where(times == gps_time)[0][0]
    lower_lim = int(np.round(trigger_index - (0.8*sample_rate)))
    upper_lim = int(np.round(trigger_index + (0.2*sample_rate)))

    BW_reconst = BW_reconst[lower_lim:upper_lim][None,:]
    reconstructed_BW, maximum, minimum = _preprocess_data(BW_reconst,1,sample_rate)

    new_lower = []
    for value in lower[lower_lim:upper_lim]:
        if value > 0.0:
            value = value/maximum # check which maximum and minimum to use
        elif value < 0.0:
            value = value/minimum
        new_lower.append(value)
        
    new_upper = []
    for value in upper[lower_lim:upper_lim]:
        if value > 0.0:
            value = value/maximum
        elif value < 0.0:
            value = value/minimum
        new_upper.append(value)  

    return reconstructed_BW, new_lower, new_upper

def get_cwb_reconst(reconst_cwb, gps_time, event_name):
    
    sampling_rate = 1024.0
    seconds_before_event = 0.8
    seconds_after_event = 0.2                                 
                     
    if 'time,' in reconst_cwb.columns:
        time_data = reconst_cwb['time,']
    elif '#time,' in reconst_cwb.columns:
        time_data = reconst_cwb['#time,']
    else:
        raise ValueError("Neither 'time,' nor '#time,' column is present in the datafile.")
        
    index_gps_time = np.where((time_data.values > gps_time-0.001) & (time_data.values < gps_time+0.001))[0][0]

    lower_lim = round(index_gps_time -(sampling_rate*seconds_before_event))
    upper_lim = round(index_gps_time +(sampling_rate*seconds_after_event))
                     
#    if(compare == 'cWB'):
    cwb_reconst = reconst_cwb['amp_cwb_rec,'].values[lower_lim:upper_lim][None,:]
    reconstructed_cwb, maximum, minimum = _preprocess_data(cwb_reconst,1,1024)
        
    new_lower = []
    for value in reconst_cwb['amp_post_lower_90_cr,'].values[lower_lim:upper_lim]:
        if value > 0.0:
            value = value/maximum # check which maximum and minimum to use
        elif value < 0.0:
            value = value/minimum
        new_lower.append(value)
        
    new_upper = []
    for value in reconst_cwb['amp_post_upper_90_cr'].values[lower_lim:upper_lim]:
        if value > 0.0:
            value = value/maximum
        elif value < 0.0:
            value = value/minimum
        new_upper.append(value)    

    if(event_name == 'GW190602_175927'):
        reconstructed_cwb = reconstructed_cwb[::2]
        new_lower = new_lower[::2]
        new_upper = new_upper[::2]

    return reconstructed_cwb, new_lower, new_upper


def get_snr_frac(decoded_signal, pure_signal):
    
    h_fft = fft(pure_signal)
    template_fft = fft(pure_signal)
    snr_freq = (h_fft * template_fft.conjugate())
    snr_time = np.abs(ifft(snr_freq))
    optimal_snr_ligo = np.sqrt(np.max(snr_time))
    
    h_fft = fft(decoded_signal)
    template_fft = fft(decoded_signal)
    snr_freq = (h_fft * template_fft.conjugate())
    snr_time = np.abs(ifft(snr_freq))
    optimal_snr_ml = np.sqrt(np.max(snr_time))

    return optimal_snr_ml/optimal_snr_ligo


def plot_reconstructed_new(decoded_signals, comparison_data, strain_data, comparisons, upper_bound, lower_bound, psd, event_name, before, after):

    time_buffer = 1
    sample_rate = 1024

    shift = dict()
    
    shift = { 
        'GW190521': 0,
        'GW190412': -40,
        'GW190517_055101': 19
    }
    
    length_signal = (before + after)*sample_rate
#    time = np.linspace(0.0, 1.0, 1024)

    max_index = np.argmax(decoded_signals)
    start_time =  -max_index/(sample_rate*1.0)
    end_time = time_buffer - max_index/(sample_rate*1.0)

    low_index = max(max_index - int(before*sample_rate),0)
    upper_index = min(max_index + int(after*sample_rate),sample_rate)

    time = np.linspace(start_time, end_time, len(decoded_signals))

#    low_index = round((0.8-before)*(sample_rate*time_buffer))
#    upper_index = round((sample_rate*time_buffer) - (0.2-after)*(sample_rate*time_buffer))

    decoded_signals = decoded_signals[low_index:upper_index]
    strain_data = strain_data[low_index:upper_index]
    time = time[low_index:upper_index]
    # Plot comparison methods

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    
    
    fig = plt.figure(figsize=(10,7))
    palette_flare = sns.color_palette("flare", n_colors=6)
    palette_rocket = sns.color_palette("rocket", n_colors=6)
    palette_crest = sns.color_palette("crest", n_colors=6)
    
    # Plot for decoded signals and pure_signals_cwb
    ax_main = fig.add_subplot(211)

#    fig, ax_main = plt.subplots(figsize=(10, 7))

#    colors = iter([palette_crest[3], palette_flare[3], palette_flare[5]])  # More colors can be added if needed

    colors = iter(['tab:blue','tab:orange', 'tab:yellow'])  # More colors can be added if needed

    ax_main.plot(time, strain_data*1.5, label='Whitened strain data', color='lightgrey', alpha=0.4)
    ax_main.plot(time, decoded_signals, label='\\texttt{AWaRe} mean reconstruction', color=palette_flare[1])
    if lower_bound is not None and upper_bound is not None:
        lower_bound = lower_bound[low_index:upper_index]
        upper_bound = upper_bound[low_index:upper_index]
        ax_main.fill_between(time,lower_bound,upper_bound, color=palette_flare[1], label='\\texttt{AWaRe} 90\% C.I.', alpha=0.5)

    for comp_method, data in comparison_data.items():

        color = next(colors)
        reconstructed_signal_ligo, lower_90, upper_90 = data

        # Length of corresponding time series and frequency series
        tlen = int(sample_rate * time_buffer)
        flen = tlen // 2 + 1

        delta_f = 1.0 / time_buffer

        if(comp_method=='BayesWave'):
            reconstructed_signal_ligo = np.roll(reconstructed_signal_ligo, shift=shift[event_name])
            lower_90 = np.roll(lower_90, shift=shift[event_name])
            upper_90 = np.roll(upper_90, shift=shift[event_name])

        reconstructed_signal_ligo = reconstructed_signal_ligo[0][low_index:upper_index]
        
        upper_90 = upper_90[low_index:upper_index]
        lower_90 = lower_90[low_index:upper_index]
        
        X_test_pure_ligo_ts = TimeSeries(np.array(reconstructed_signal_ligo).astype('float64'), delta_t = 1.0/1024)
        decoded_signals_ts = TimeSeries(np.array(decoded_signals).astype('float64'), delta_t = 1.0/1024)
        psd = FrequencySeries(psd, delta_f=delta_f)

        X_test_pure_ligo_ts.resize(tlen)
        decoded_signals_ts.resize(tlen)
        psd.resize(flen)

        if(X_test_pure_ligo_ts.max() != 0.0):
    #        m_cwb = Overlap_calc(X_test_pure_cwb_ts,decoded_signals_ts,psd)
            m_ligo = Overlap_calc(X_test_pure_ligo_ts,decoded_signals_ts,psd)
        else:
    #        m_cwb = 1
            m_ligo = 1
        
    #    snr_frac_cwb = get_snr_frac(decoded_signals.squeeze(), pure_signals_cwb)
        snr_frac_ligo = get_snr_frac(decoded_signals.squeeze(), reconstructed_signal_ligo)

        
        
    #    signal_axes.plot(time, pure_signals_cwb, label='cWB max likelihood L1 waveform')
    #    signal_axes.plot(time, reconstructed_signal_ligo, label=f'{comp_method} max. likelihood reconstruction', color=color)
        ax_main.fill_between(time, lower_90, upper_90, color=color, label=f'{comp_method} 90\% C.I.')

    ax_main.set_ylabel('Rescaled amplitudes', fontsize=15)
    ax_main.set_xlabel('Time (in secs)', fontsize=15)
    ax_main.tick_params(axis='y', labelsize=15)
    ax_main.tick_params(axis='x', labelsize=15)

#    if before == 0.8 and after == 0.2:
#        ax_main.legend(loc='lower left', fontsize=12)

    ax_main.legend(loc='lower left', fontsize=12)

    plt.savefig(f'Paper_plots_uncertainty/Test_real_events/{event_name}_reconstruction_comparison_{comp_method}_b={before}_a={after}_with_strain_combined_poster.png', bbox_inches='tight', dpi=400)
    print('Plot generated!')
    print('Overlap (LIGO) = {m_ligo}, SNR_frac (LIGO) = {snr_frac_ligo}'.format(m_ligo=np.round(m_ligo,2), snr_frac_ligo=np.round(snr_frac_ligo,2)))


def plot_reconstructed_zoom_in(decoded_signals, comparison_data, strain_data, comparisons, upper_bound, lower_bound, psd, event_name, before, after):

    time_buffer = 1
    sample_rate = 1024

    if(event_name == 'GW190602_175927'):
        sample_rate = 512

    shift = dict()
    
    shift = { 
        'GW190521': 0,
        'GW190412': -40,
        'GW190517_055101': 19,
        'GW190602_175927': 2
    }
    
    length_signal = 1.0*sample_rate
#    time = np.linspace(0.0, 1.0, 1024)

    max_index = np.argmax((comparison_data['cWB'][0][0]))

#    max_index = np.argmax(decoded_signals)
    
    start_time =  -max_index/(sample_rate*1.0)
    end_time = time_buffer - max_index/(sample_rate*1.0)

    low_index = max(max_index - int(before*sample_rate),0)
    upper_index = min(max_index + int(after*sample_rate),sample_rate)

    time = np.linspace(start_time, end_time, len(decoded_signals))

#    low_index = round((0.8-before)*(sample_rate*time_buffer))
#    upper_index = round((sample_rate*time_buffer) - (0.2-after)*(sample_rate*time_buffer))

    decoded_signals = decoded_signals[low_index:upper_index]
    strain_data = strain_data[low_index:upper_index]
    time = time[low_index:upper_index]
    # Plot comparison methods

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    
    
    fig = plt.figure(figsize=(10,7))
    palette_flare = sns.color_palette("flare", n_colors=6)
    palette_viridis = sns.color_palette("viridis", n_colors=6)
    palette_crest = sns.color_palette("crest", n_colors=6)
    palette_set2 = sns.color_palette("Set2", 4)
    
    # Plot for decoded signals and pure_signals_cwb
    ax_main = fig.add_subplot(211)

    # Ensure the zoom box does not go out of the figure bounds
    zoom_width = 0.5
    zoom_height = 0.5  # height of the zoom box in figure coordinates, adjustable
    
    zoom_x = 1.1  # top = 1.1
    zoom_y = 0.55 # top = 0.8, middle = 0.5, bottom = 0.1

    # Define the region of interest for zooming
    x1, x2 = -0.05, 0.05 # Adjust these values as per your needs

    # Add the zoomed subplot within the calculated coordinates
    ax_zoom = fig.add_axes([zoom_x, zoom_y, zoom_width, zoom_height])

#    fig, ax_main = plt.subplots(figsize=(10, 7))

    colors = iter(['tab:blue', 'tab:orange'])  # More colors can be added if needed

#    colors = iter(['tab:blue','tab:green', 'tab:yellow'])  # More colors can be added if needed

    ax_main.plot(time, strain_data*1.5, label='Whitened strain data', color='lightgrey', alpha=0.4)
    ax_main.plot(time, decoded_signals, label='\\texttt{AWaRe} mean reconstruction', color=palette_flare[1])

    ax_zoom.plot(time, strain_data*1.5, label='Whitened strain data', color='lightgrey', alpha=0.4)
    ax_zoom.plot(time, decoded_signals, label='\\texttt{AWaRe} mean reconstruction', color=palette_flare[1])

    
    if lower_bound is not None and upper_bound is not None:
        lower_bound = lower_bound[low_index:upper_index]
        upper_bound = upper_bound[low_index:upper_index]
#        ax_main.fill_between(time,lower_bound,upper_bound, color='tab:orange', label='\\texttt{AWaRe} 90\% C.I.', alpha=0.5)
#        ax_zoom.fill_between(time,lower_bound,upper_bound, color='tab:orange', label='\\texttt{AWaRe} 90\% C.I.', alpha=0.5)

        ax_main.fill_between(time,lower_bound,upper_bound, color=palette_flare[1], label='\\texttt{AWaRe} 90\% C.I.', alpha=0.4)
        ax_zoom.fill_between(time,lower_bound,upper_bound, color=palette_flare[1], label='\\texttt{AWaRe} 90\% C.I.', alpha=0.4)


    for comp_method, data in comparison_data.items():

        color = next(colors)
        reconstructed_signal_ligo, lower_90, upper_90 = data

        # Length of corresponding time series and frequency series
        tlen = int(sample_rate * time_buffer)
        flen = tlen // 2 + 1

        delta_f = 1.0 / time_buffer

        if(comp_method=='BayesWave'):
            if(event_name in shift.keys()):
                reconstructed_signal_ligo = np.roll(reconstructed_signal_ligo, shift=shift[event_name])
                lower_90 = np.roll(lower_90, shift=shift[event_name])
                upper_90 = np.roll(upper_90, shift=shift[event_name])
        
        reconstructed_signal_ligo = reconstructed_signal_ligo[0][low_index:upper_index]
        
        upper_90 = upper_90[low_index:upper_index]
        lower_90 = lower_90[low_index:upper_index]
        
        X_test_pure_ligo_ts = TimeSeries(np.array(reconstructed_signal_ligo).astype('float64'), delta_t = 1.0/sample_rate)
        decoded_signals_ts = TimeSeries(np.array(decoded_signals).astype('float64'), delta_t = 1.0/sample_rate)
        psd = FrequencySeries(psd, delta_f=delta_f)

        X_test_pure_ligo_ts.resize(tlen)
        decoded_signals_ts.resize(tlen)
        psd.resize(flen)

        if(X_test_pure_ligo_ts.max() != 0.0):
    #        m_cwb = Overlap_calc(X_test_pure_cwb_ts,decoded_signals_ts,psd)
            m_ligo = Overlap_calc(X_test_pure_ligo_ts,decoded_signals_ts,psd)
        else:
    #        m_cwb = 1
            m_ligo = 1
        
    #    snr_frac_cwb = get_snr_frac(decoded_signals.squeeze(), pure_signals_cwb)
        snr_frac_ligo = get_snr_frac(decoded_signals.squeeze(), reconstructed_signal_ligo)

        
        # Calculate the normalized figure coordinates for the zoom box
    #    trans = ax_main.transData + fig.transFigure.inverted()
    #    zoom_x1, _ = trans.transform((-before, 0))
    #    zoom_x2, _ = trans.transform((after, 0))

    #    signal_axes.plot(time, pure_signals_cwb, label='cWB max likelihood L1 waveform')
    #    signal_axes.plot(time, reconstructed_signal_ligo, label=f'{comp_method} max. likelihood reconstruction', color=color)
        ax_main.fill_between(time, lower_90, upper_90, color=color, alpha=0.8, label=f'{comp_method} 90\% C.I.')

    #    ax_zoom = fig.add_axes([0.8, 0.5, 0.8, 0.3])  # External axes for the zoomed region
        ax_zoom.fill_between(time, lower_90, upper_90, color=color, alpha=0.8, label=f'{comp_method} 90\% C.I.')
        ax_zoom.set_xlim(x1, x2)
    #    ax_zoom.set_xticklabels([])
        ax_zoom.set_yticklabels([])
    #    ax_main.tick_params(axis='y', labelsize=15)
        ax_zoom.tick_params(axis='x', labelsize=20)
    #    ax_zoom.legend(loc='lower left', fontsize=12)

    # Draw lines connecting the main plot and the zoomed region
    mark_inset(ax_main, ax_zoom, loc1=2, loc2=4, fc="none", ec="0.5")

    ax_main.set_xlim(time[0], time[-1])
    ax_main.set_ylabel('Rescaled amplitudes', fontsize=20)
    ax_main.set_xlabel('Time from merger (s)', fontsize=20)
    ax_main.tick_params(axis='y', labelsize=20)
    ax_main.tick_params(axis='x', labelsize=20)

#    if before == 0.8 and after == 0.2:
#        ax_main.legend(loc='lower left', fontsize=12)

    ax_main.legend(loc='lower left', fontsize=18)

    plt.savefig(f'Paper_plots_uncertainty/Test_real_events/{event_name}_reconstruction_comparison_{comp_method}_b={before}_a={after}_with_strain_combined_zoom_in_new_3.svg', bbox_inches='tight', dpi=400)
    print('Plot generated!')
    print('Overlap (LIGO) = {m_ligo}, SNR_frac (LIGO) = {snr_frac_ligo}'.format(m_ligo=np.round(m_ligo,2), snr_frac_ligo=np.round(snr_frac_ligo,2)))


def plot_reconstructed_old(decoded_signals, pure_signals_cwb, pure_signals_pe, upper_90, lower_90, psd, event_name, before, after):
    
#    time_buffer = 104.0/1024
    time_buffer = 1
    f_low = 20
    sample_rate = 1024
    
    length_signal = (before + after)*sample_rate
    
    low_index = round((0.8-before)*(sample_rate*time_buffer))
    upper_index = round((sample_rate*time_buffer) - (0.2-after)*(sample_rate*time_buffer))
    
    pure_signals_cwb = pure_signals_cwb[low_index:upper_index]
    pure_signals_pe = pure_signals_pe[low_index:upper_index]
    decoded_signals = decoded_signals[low_index:upper_index]
    
    upper_90 = upper_90[low_index:upper_index]
    lower_90 = lower_90[low_index:upper_index]


    fig = plt.figure(figsize=(10,5))
    
    time = np.linspace(-before, after, round(length_signal))

#    time = np.linspace(-0.80, 0.20, 1024)[920:1024]
    

    # Length of corresponding time series and frequency series
    tlen = int(sample_rate * time_buffer)
    flen = tlen // 2 + 1

    delta_f = 1.0 / time_buffer
#    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)
    
    X_test_pure_cwb_ts = TimeSeries(pure_signals_cwb.squeeze(), delta_t = 1.0/1024)
    X_test_pure_pe_ts = TimeSeries(pure_signals_pe.squeeze(), delta_t = 1.0/1024)
    decoded_signals_ts = TimeSeries(decoded_signals.squeeze(), delta_t = 1.0/1024)
    psd = FrequencySeries(psd, delta_f=delta_f)

    X_test_pure_cwb_ts.resize(tlen)
    X_test_pure_pe_ts.resize(tlen)
    decoded_signals_ts.resize(tlen)
    psd.resize(flen)

    if(X_test_pure_cwb_ts.max() != 0.0):
        m_cwb = Overlap_calc(X_test_pure_cwb_ts,decoded_signals_ts,psd)
        m_pe = Overlap_calc(X_test_pure_pe_ts,decoded_signals_ts,psd)
    else:
        m_cwb = 1
        m_pe = 1
        
    snr_frac = get_snr_frac(decoded_signals.squeeze(), pure_signals)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    signal_axes = fig.add_subplot(211)
    signal_axes.set_title('Overlap = {m}, SNR fraction recovered = {snr_frac}'.format(m=np.round(m,2), snr_frac=np.round(snr_frac,2)))
#    signal_axes.set_title('SNR fraction recovered = {snr_frac}'.format(snr_frac=np.round(snr_frac,2)))
    signal_axes.plot(time,pure_signals, linewidth=1.2, label = compare+' maximum likelihood L1 signal')
    signal_axes.plot(time,decoded_signals, linewidth=1.2, label ='ML reconstructed L1 signal', c='red', linestyle='dashed')
    signal_axes.fill_between(time,lower_90,upper_90, color='lightgrey', label=compare+'-LALInference 90% C.I.')
    signal_axes.set_ylabel('Rescaled Amplitudes')
    signal_axes.set_xlabel('Time from merger (in secs)')
    signal_axes.legend()
    
    plt.savefig('Test_plots/'+str(event_name)+'_reconstruction_L1_'+str(compare)+'_comparison.png', bbox_inches='tight', facecolor='w', transparent=False, dpi=200)
    
    print('Plot generated!')


def get_comparison_data(comparison_methods, gps_time, event_name):
    # This function retrieves comparison data for selected methods
    comparison_data = {}
    for method in comparison_methods:
        if method == 'cWB':

            try:
                cWB_file_path = os.path.join("/fred/oz016/Chayan/samplegen_old/output", f"{event_name}_rec_signal_time_L1.dat")
                reconst_cWB = pd.read_csv(cWB_file_path, header=0, delimiter=" ")

            except FileNotFoundError:
                print(f"File not found: {cWB_file_path}")
        
            comparison_data['cWB'] = get_cwb_reconst(reconst_cWB, gps_time, event_name)     
        
        elif method == 'BayesWave':

            try:
                BW_file_path = os.path.join("/fred/oz016/Chayan/samplegen_old/output", f"{event_name}_BayesWave_L1.txt")
                reconst_BW = pd.read_csv(BW_file_path, header=0, delimiter=" ")

            except FileNotFoundError:
                print(f"File not found: {BW_file_path}")

            comparison_data['BayesWave'] = get_BW_reconst(reconst_BW, gps_time, event_name)

        elif method == 'PE':
            
            try:
                cWB_file_path = os.path.join("/fred/oz016/Chayan/samplegen_old/output", f"{event_name}_rec_signal_time_L1.dat")
                reconst_cWB = pd.read_csv(cWB_file_path, header=0, delimiter=" ")

            except FileNotFoundError:
                print(f"File not found: {cWB_file_path}")
        

            comparison_data['PE'] = get_PE_reconst(reconst_cWB, gps_time, event_name)

    return comparison_data

    
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Plot the reconstructed GW waveforms from autoencoder and PE/cWB")

    # Add arguments
    parser.add_argument("observation_run", type=str, help="Provide the name of the LIGO-Virgo observation run (O1/O2/O3)")
    parser.add_argument("event_name", type=str, help="The GW event id (example: 'GW150914')")
#    parser.add_argument("detector", type=str, help="The detector name ('H1'/'L1')")
    parser.add_argument("-c", "--comparison", nargs='+', help="Comparison methods like BayesWave, cWB, PE", default=["BayesWave", "cWB"])
    parser.add_argument("-b", "--before", type=float, help="Seconds before the event for zoomed plots", default=0.8)
    parser.add_argument("-a", "--after", type=float, help="Seconds after the event for zoomed plots", default=0.2)
    parser.add_argument("-unc", "--uncertainty", type=bool, help="Plot ML uncertainty as well or just max likelihood", default=True)
    
    valid_runs = ['O1', 'O2', 'O3']
    valid_detectors = ['H1', 'L1']
    
    valid_events_O1 = ['GW150914', 'GW151012', 'GW151226']
    O1_events_gps_times = [1126259462.391, 1128678900.445, 1135136350.648]
    O1_events_ids = ['GW150914-v3', 'GW151012', 'GW151226'] 
    
    O1_events_gps_dict = {event_id: gps_time for event_id, gps_time in zip(valid_events_O1, O1_events_gps_times)}
    O1_events_id_dict = {event: event_name for event, event_name in zip(valid_events_O1, O1_events_ids)}
    
    
    valid_events_O2 = ['GW170104', 'GW170608', 'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823']
    O2_events_gps_times = [1167559936.599, 1180922494.492, 1185389807.326, 1186302519.745, 1186741861.523, 1187058327.080, 1187529256.517]
    O2_events_ids = ['GW170104-v2', 'GW170608-v3', 'GW170729-v1', 'GW170809-v1', 'GW170814-v3', 'GW170818-v1', 'GW170823-v1']
    
    O2_events_gps_dict = {event_id: gps_time for event_id, gps_time in zip(valid_events_O2, O2_events_gps_times)}
    O2_events_id_dict = {event: event_name for event, event_name in zip(valid_events_O2, O2_events_ids)}
    
    
    valid_events_O3 = ['GW190521', 'GW191109_010717', 'GW190602_175927', 'GW190412', 'GW190519_153544', 'GW190517_055101', 'GW190503_185404']
    O3_events_gps_times = [1242442967.448, 1257296855.222, 1243533585.085, 1239082262.201, 1242315362.379, 1242107479.820, 1240944862.285]
    O3_events_ids = ['GW190521-v3', 'GW191109_010717-v1', 'GW190602_175927-v1', 'GW190412-v3', 'GW190519_153544-v1', 'GW190517_055101-v1', 'GW190503_185404-v1']
    
    O3_events_gps_dict = {event_id: gps_time for event_id, gps_time in zip(valid_events_O3, O3_events_gps_times)}
    O3_events_id_dict = {event: event_name for event, event_name in zip(valid_events_O3, O3_events_ids)}
    
    
    valid_comparisons = ['PE', 'cWB', 'BayesWave']
        
    # Parse arguments
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(e.message)
        sys.exit(1)

    # Check if the name is valid
    if args.observation_run not in valid_runs:
        print(f"Error: Observation runs must be one of {valid_runs}.")
        sys.exit(1)
    
    if args.observation_run == 'O1' and args.event_name not in valid_events_O1:
        print(f"Error: O1 events must be one of {valid_events_O1}.")
        sys.exit(1)
        
    if args.observation_run == 'O2' and args.event_name not in valid_events_O2:
        print(f"Error: O2 events must be one of {valid_events_O2}.")
        sys.exit(1)
        
    if args.observation_run == 'O3' and args.event_name not in valid_events_O3:
        print(f"Error: O3 events must be one of {valid_events_O3}.")
        sys.exit(1)
        
    for comparison in args.comparison:
        if comparison not in valid_comparisons: 
            print(f"Error: Comparisons must be one of {valid_comparisons}.")
            sys.exit(1)
        
    if args.before > 0.8:
        print(f"Error: Seconds before merger cannot be greater than 0.8.")
        sys.exit(1)
        
    if args.after > 0.2:
        print(f"Error: Seconds after merger cannot be greater than 0.2.")
        sys.exit(1)
    
    
    if(args.observation_run == 'O1'):
        event_gps_time = O1_events_gps_dict[str(args.event_name)]
        event_id = O1_events_id_dict[str(args.event_name)]
    
    elif(args.observation_run == 'O2'):
        event_gps_time = O2_events_gps_dict[str(args.event_name)]
        event_id = O2_events_id_dict[str(args.event_name)]
    
    elif(args.observation_run == 'O3'):
        event_gps_time = O3_events_gps_dict[str(args.event_name)]
        event_id = O3_events_id_dict[str(args.event_name)]
        
    real_event_file_path = os.path.join('/fred/oz016/Chayan/samplegen_old/output', f'real_events_{args.event_name}.hdf')
    # Open the file and read the data
    with h5py.File(real_event_file_path, 'r') as f1:
        strain = f1[event_id]['l1_strain'][()]
    
    # Get AWaRe reconstruction
    reconstructed_signal_ml, lower_bound, upper_bound, strain_data = evaluate_model(strain, args.observation_run, args.event_name, args.uncertainty)

    if(args.event_name == 'GW190602_175927'):
        reconstructed_signal_ml = reconstructed_signal_ml[::4]
        lower_bound = lower_bound[::4]
        upper_bound = upper_bound[::4]
        strain_data = strain_data[0][::4]

    # Get comparison data based on user input
    comparison_data = get_comparison_data(args.comparison, event_gps_time, args.event_name)
    
#    if(args.comparison == 'cWB'):

#        try:
#            cWB_file_path = os.path.join("/fred/oz016/Chayan/samplegen_old/output", f"{args.event_name}_rec_signal_time_L1.dat")
#            reconst_cWB = pd.read_csv(cWB_file_path, header=0, delimiter=" ")

#        except FileNotFoundError:
#            print(f"File not found: {cWB_file_path}")
        
#        reconstructed_signal, lower_90, upper_90 = get_cwb_reconst(reconst_cWB, event_gps_time, args.event_name)     
        
#    elif(args.comparison == 'BayesWave'):

#        try:
#            BW_file_path = os.path.join("/fred/oz016/Chayan/samplegen_old/output", f"{args.event_name}_BayesWave_L1.txt")
#            reconst_BW = pd.read_csv(BW_file_path, header=0, delimiter=" ")

#        except FileNotFoundError:
#            print(f"File not found: {BW_file_path}")
        
#        reconstructed_signal, lower_90, upper_90 = get_BW_reconst(reconst_BW, event_gps_time, args.event_name)     
        
#    elif(args.comparison == 'PE'):
         
#        try:
#            cWB_file_path = os.path.join("/fred/oz016/Chayan/samplegen_old/output", f"{args.event_name}_rec_signal_time_L1.dat")
#            reconst_cWB = pd.read_csv(cWB_file_path, header=0, delimiter=" ")

#        except FileNotFoundError:
#            print(f"File not found: {cWB_file_path}")
        

#        reconstructed_signal, lower_90, upper_90 = get_PE_reconst(reconst_cWB, event_gps_time, args.event_name)
    
    
    psd = get_psd(args.event_name)                 
        
#    plot_reconstructed_ml(reconstructed_signal_ml[::2][0:1024].astype('float64'), np.squeeze(reconstructed_signal).astype('float64'), np.squeeze(strain_data)[::2][0:1024].astype('float64'), args.comparison, upper_90, lower_90, lower_bound[::2][0:1024], upper_bound[::2][0:1024], psd, args.event_name, args.before, args.after)
    
    if(args.event_name == 'GW190602_175927'):

        plot_reconstructed_zoom_in(reconstructed_signal_ml[0:512], comparison_data, strain_data[0:512], args.comparison, upper_bound[0:512], lower_bound[0:512], psd, args.event_name, args.before, args.after)
    
    else:
        plot_reconstructed_zoom_in(reconstructed_signal_ml[::2][0:1024], comparison_data, strain_data[0][::2][0:1024], args.comparison, upper_bound[::2][0:1024], lower_bound[::2][0:1024], psd, args.event_name, args.before, args.after)

#    plot_reconstructed_new(reconstructed_signal_ml[::2][0:1024], comparison_data, strain_data[0][::2][0:1024], args.comparison, upper_bound[::2][0:1024], lower_bound[::2][0:1024], psd, args.event_name, args.before, args.after)

    # Print the results
#    print(f"Hello, you are plotting {args.event_name} from {args.observation_run}! You are comparing autoencoder results with {args.comparison}. Time = {input_data['time,'].values[0]}")

if __name__ == "__main__":
    main()
