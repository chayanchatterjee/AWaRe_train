import h5py
from numpy import random
import numpy as np

f1 = h5py.File('/workspace/ligo_data/ml-training-strategies/data/train_signals.hdf', 'r')
f2 = h5py.File('/workspace/ligo_data/ml-training-strategies/data/train_noise.hdf', 'r')

strain = []
snr_array = []
rescaled_waveforms = []
noise = []

for i in range(f1['data']['0'].shape[0]):

    snr = random.default_rng().uniform(low=5, high=15)
    strain.append(f2['data']['0'][i].squeeze() + (snr*f1['data']['0'][i]).squeeze())
    snr_array.append(snr)
    rescaled_waveforms.append((snr*f1['data']['0'][i]).squeeze())
    noise.append(f2['data']['0'][i].squeeze())

strain = np.array(strain)
rescaled_waveforms = np.array(rescaled_waveforms)
snr_array = np.array(snr_array)
noise = np.array(noise)

f3 = h5py.File('/workspace/ligo_data/AWaRe_train/noise_recons_snr_5-15.hdf', 'w')
f3.create_dataset('strain', data=strain)
f3.create_dataset('rescaled_waveforms', data=rescaled_waveforms)
f3.create_dataset('snr_array', data=snr_array)
f3.create_dataset('noise', data=noise)

f1.close()
f2.close()
f3.close()

