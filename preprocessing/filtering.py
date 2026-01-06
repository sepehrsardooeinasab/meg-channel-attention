from scipy.signal import butter, filtfilt, freqz, iirnotch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import glob
import os


def butter_bandpass_filter(signal, lowcut, highcut, sampling_rate, show=False):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(N=3, Wn=[low, high], btype='bandpass', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    
    if show:
        w, h = freqz(b, a, worN=2000)
        plt.plot(nyquist / np.pi * w, abs(h), color="#000000")
        plt.show()
    
    return filtered_signal


def notch60_filter(signal, sampling_rate, show=False):
    nyquist = 0.5 * sampling_rate

    b, a = iirnotch(60, 30, sampling_rate)
    filtered_signal = filtfilt(b, a, signal)
    
    if show:
        w, h = freqz(b, a, worN=2000)
        plt.plot(nyquist / np.pi * w, abs(h), color="#000000")
        plt.show()

    return filtered_signal


# Constants
src_base = 'data_average_channels'
dest_base = 'data_ds12avg'
downsampling_rate = 12
butterworth_bands = (0.3, 90)
show = False


# Create destination directory
shutil.rmtree(dest_base, ignore_errors=True)
os.makedirs(dest_base, exist_ok=True)


# Create directory structure
for subject_dir in sorted(glob.glob(os.path.join(src_base, 'Sub*'))):
    subject_name = os.path.basename(subject_dir)
    subject_dest = os.path.join(dest_base, subject_name)
    shutil.rmtree(subject_dest, ignore_errors=True)
    os.makedirs(subject_dest, exist_ok=True)

    for trial_dir in sorted(glob.glob(os.path.join(subject_dir, 'MNI*'))):
        trial_name = os.path.basename(trial_dir)
        trial_dest = os.path.join(subject_dest, trial_name)
        os.makedirs(trial_dest, exist_ok=True)


# Downsample, filter and save data
for subject_dir in tqdm(sorted(glob.glob(os.path.join(src_base, 'Sub*')))):
    for trial_dir in sorted(glob.glob(os.path.join(subject_dir, 'MNI*'))):
        for epoch_path in sorted(glob.glob(os.path.join(trial_dir, 'data_block*'))):
            epoch = pd.read_csv(epoch_path, index_col=0).transpose()
            downsampled_epoch = pd.DataFrame()

            for column in epoch.columns:
                signal_bandpassed = butter_bandpass_filter(epoch[column], *butterworth_bands, 2400, show=show)
                downsampled_bandpassed = signal_bandpassed[::downsampling_rate]
                downsampled_epoch[column] = pd.Series(downsampled_bandpassed)

            downsampled_epoch = downsampled_epoch.transpose()
            relative_path = os.path.relpath(epoch_path, start=src_base)
            output_path = os.path.join(dest_base, relative_path)
            downsampled_epoch.to_csv(output_path, index=True, index_label='Areas')