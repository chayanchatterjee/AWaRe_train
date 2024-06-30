from __future__ import print_function
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import numpy as np
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

#        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_IMRPhenomXPHM_O3b_noise_15_timesteps_uncertainty.h5', custom_objects={'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention, 'IndependentNormal': tfp.layers.IndependentNormal,'negloglik': negloglik})
#        num_steps = 15

        model = tf.keras.models.load_model('/fred/oz016/Chayan/GW-Denoiser/model/model_NRSur7dq4_O3b_noise_15_timesteps_uncertainty.h5', custom_objects={'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention, 'IndependentNormal': tfp.layers.IndependentNormal,'negloglik': negloglik})
        num_steps = 15

        
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
        lower_bound = mean_preds - 2 * std_preds
        upper_bound = mean_preds + 2 * std_preds

    return mean_preds, lower_bound, upper_bound


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

    return amp

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


def get_ligo_reconst(reconst_cwb, gps_time, event_name):
    
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
    
#    elif(compare == 'PE'):
    pe_reconst = get_pe_data(event_name)
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

    return reconstructed_cwb, reconstructed_pe, new_lower, new_upper


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


def plot_reconstructed(decoded_signals, pure_signals_cwb, pure_signals_pe, upper_90, lower_90, lower_bound, upper_bound, psd, event_name, before, after):
    time_buffer = 1
    sample_rate = 1024
    
    length_signal = (before + after)*sample_rate
    
    low_index = round((0.8-before)*(sample_rate*time_buffer))
    upper_index = round((sample_rate*time_buffer) - (0.2-after)*(sample_rate*time_buffer))
    
    pure_signals_cwb = pure_signals_cwb[low_index:upper_index]
    pure_signals_pe = pure_signals_pe[low_index:upper_index]
    decoded_signals = decoded_signals[low_index:upper_index]
    
    upper_90 = upper_90[low_index:upper_index]
    lower_90 = lower_90[low_index:upper_index]

    if(lower_bound is not None):
        lower_bound = lower_bound[low_index:upper_index]
        upper_bound = upper_bound[low_index:upper_index]

    time = np.linspace(-before, after, round(length_signal))

    # Create subplots
#    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    
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
        
    snr_frac_cwb = get_snr_frac(decoded_signals.squeeze(), pure_signals_cwb)
    snr_frac_pe = get_snr_frac(decoded_signals.squeeze(), pure_signals_pe)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    
    fig = plt.figure(figsize=(10,7))
    
    # Plot for decoded signals and pure_signals_cwb
    signal_axes = fig.add_subplot(211)
    signal_axes.set_title('Overlap (cWB) = {m_cwb}, Overlap (LAL) = {m_pe}, SNR_frac (cWB) = {snr_frac_cwb}, SNR_frac (LAL) = {snr_frac_pe}'.format(m_cwb=np.round(m_cwb,2), m_pe=np.round(m_pe,2), snr_frac_cwb=np.round(snr_frac_cwb,2),snr_frac_pe=np.round(snr_frac_pe,2)))
#    signal_axes.plot(time, decoded_signals, label='\\texttt{AWaRe} reconstructed waveform', color='red', linestyle='dashed')
    signal_axes.plot(time, decoded_signals, label='\\texttt{AWaRe} reconstructed waveform', color='red')
    signal_axes.plot(time, pure_signals_cwb, label='cWB max likelihood L1 waveform')
    signal_axes.plot(time, pure_signals_pe, label='LAL max likelihood L1 waveform', color='green')
    signal_axes.fill_between(time, lower_90, upper_90, color='grey', label='cWB 90\% C.I.', alpha=0.6)
    
    if(lower_bound is not None):
        signal_axes.fill_between(time,lower_bound,upper_bound, color='lightpink', label='\\texttt{AWaRe} 90\% C.I.', alpha=0.5)
    
    signal_axes.set_ylabel('Rescaled amplitudes', fontsize=15)
    signal_axes.set_xlabel('Time from merger (in secs)', fontsize=15)
    signal_axes.tick_params(axis='y', labelsize=15)
    signal_axes.tick_params(axis='x', labelsize=15)
    signal_axes.legend(loc='lower left', fontsize=12)

    # Plot for pure_signals_pe
#    ax2.set_title('Overlap = {m_pe}, SNR fraction recovered = {snr_frac_pe}'.format(m_pe=np.round(m_pe,2), snr_frac_pe=np.round(snr_frac_pe,2)))
#    ax2.plot(time, decoded_signals, label='ML reconstructed L1 signal', color='red', linestyle='dashed')
#    ax2.plot(time, pure_signals_cwb, label=' maximum likelihood L1 signal')
#    ax2.fill_between(time, lower_90, upper_90, color='lightgrey', label='cWB 90% C.I.')
#    ax2.set_ylabel('Rescaled_amplitudes')
#    ax2.set_xlabel('Time from merger (in secs)')
#    ax2.legend()

    plt.savefig(f'Paper_plots_uncertainty/{event_name}_reconstruction_comparison_b={before}_a={after}_NRSur7dq4.png', bbox_inches='tight', dpi=200)
    print('Plot generated!')
    print('Overlap (cWB) = {m_cwb}, Overlap (LAL) = {m_pe}, SNR_frac (cWB) = {snr_frac_cwb}, SNR_frac (LAL) = {snr_frac_pe}'.format(m_cwb=np.round(m_cwb,2), m_pe=np.round(m_pe,2), snr_frac_cwb=np.round(snr_frac_cwb,2),snr_frac_pe=np.round(snr_frac_pe,2)))


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

    
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Plot the reconstructed GW waveforms from autoencoder and PE/cWB")

    # Add arguments
    parser.add_argument("observation_run", type=str, help="Provide the name of the LIGO-Virgo observation run (O1/O2/O3)")
    parser.add_argument("event_name", type=str, help="The GW event id (example: 'GW150914')")
    parser.add_argument("detector", type=str, help="The detector name ('H1'/'L1')", default='L1')
#    parser.add_argument("-c", "--comparison", type=str, help="Compare with PE/cWB", default="PE")
    parser.add_argument("-b", "--before", type=float, help="Seconds before the event for zoomed plots", default=0.8)
    parser.add_argument("-a", "--after", type=float, help="Seconds after the event for zoomed plots", default=0.2)
    parser.add_argument("-unc", "--uncertainty", type=bool, help="Plot ML uncertainty as well or just max likelihood", default=True)
    
    valid_runs = ['O1', 'O2', 'O3']

    # Define a dictionary for GPS times (as an example)
    detector_gps_times = {
        'O1': {
            'L1': {
                'GW150914': 1126259462.391,
                'GW151012': 1128678900.445,
                'GW151226': 1135136350.648
            },
        },
        'O2': {
            'L1': {
                'GW170104': 1167559936.599,
                'GW151012': 1128678900.445,
                'GW151226': 1135136350.648
            },
        }
    }
    
    ###### TODO: Replace these lists with dictionary as above. But event names also need to be mapped to their 
    ############ corresponding event_ids in the files like GW150914 and GW150914-v3 ##########################
    
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
    
    
#    valid_comparisons = ['PE', 'cWB']
        

    # Parse arguments
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(e.message)
        sys.exit(1)


    # Check if the detector is valid
    if args.detector not in ['H1', 'L1']:
        print(f"Error: Detector must be one of ['H1', 'L1'].")
        sys.exit(1)

    # Check if the event is valid for the given detector
    if args.event_name not in detector_gps_times[args.detector]:
        print(f"Error: Event {args.event_name} not found for detector {args.detector}.")
        sys.exit(1)

    # Get the GPS time for the event and detector
    event_gps_time = detector_gps_times[args.detector][args.event_name]


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
        
#    if args.comparison not in valid_comparisons:
#        print(f"Error: Comparisons must be one of {valid_comparisons}.")
#        sys.exit(1)
        
    if args.before > 0.8:
        print(f"Error: Seconds before merger cannot be greater than 0.8.")
        sys.exit(1)
        
    if args.after > 0.2:
        print(f"Error: Seconds after merger cannot be greater than 0.2.")
        sys.exit(1)
        
     
    try:
        cWB_file_path = os.path.join("/fred/oz016/Chayan/samplegen_old/output", f"{args.event_name}_rec_signal_time_L1.dat")
        reconst_cWB = pd.read_csv(cWB_file_path, header=0, delimiter=" ")
    except FileNotFoundError:
        print(f"File not found: {cWB_file_path}")
    
    
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
    
    reconstructed_signal_ml, lower_bound, upper_bound = evaluate_model(strain, args.observation_run, args.event_name, args.uncertainty)
    reconstructed_signal_cwb, reconstructed_signal_pe, lower_90, upper_90 = get_ligo_reconst(reconst_cWB, event_gps_time, args.event_name)     
    psd = get_psd(args.event_name)                 
    
#    plot_reconstructed(reconstructed_signal_ml[0][::2][0:1024].astype('float64'), np.squeeze(reconstructed_signal_cwb).astype('float64'), np.squeeze(reconstructed_signal_pe).astype('float64'), upper_90, lower_90, lower_bound[::2][0:1024], upper_bound[::2][0:1024], psd, args.event_name, args.before, args.after)
    plot_reconstructed(reconstructed_signal_ml[::2][0:1024].astype('float64'), np.squeeze(reconstructed_signal_cwb).astype('float64'), np.squeeze(reconstructed_signal_pe).astype('float64'), upper_90, lower_90, lower_bound[::2][0:1024], upper_bound[::2][0:1024], psd, args.event_name, args.before, args.after)
    
    
    # Print the results
#    print(f"Hello, you are plotting {args.event_name} from {args.observation_run}! You are comparing autoencoder results with {args.comparison}. Time = {input_data['time,'].values[0]}")

if __name__ == "__main__":
    main()
