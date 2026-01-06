from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import glob
import os


# Step 1: Find MEG channels present in all trials
num_trials = 0
channels_repeats = {}

for subject_dir in sorted(glob.glob(os.path.join('data', '*'))):
    for trial_dir in sorted(glob.glob(os.path.join(subject_dir, 'MNI*'))):
        num_trials += 1
        channels = loadmat(os.path.join(trial_dir, 'channel_ctf_acc1.mat'))['Channel'][0]
        for channel in channels:
            if channel[4][0]=='MEG':
                channel_name = channel[5][0]
                channels_repeats[channel_name] = channels_repeats.get(channel_name, 0) + 1

channels_intersection = [channel for channel in channels_repeats.keys() if channels_repeats[channel]==num_trials]
with open('channels.txt', 'w') as f:
    for channel in channels_intersection:
        f.write(channel + '\n')

print(f"Total trials: {num_trials}")
print(f"Channels in all trials: {len(channels_intersection)}")


# Step 2: Create output directory structure
dest_base = 'data_average_channels'
shutil.rmtree(dest_base, ignore_errors=True)
os.makedirs(dest_base, exist_ok=True)

for subject_dir in sorted(glob.glob(os.path.join('data', '*'))):
    subject_name = os.path.basename(subject_dir)
    subject_dest = os.path.join(dest_base, subject_name)
    os.makedirs(subject_dest, exist_ok=True)

    for trial_dir in sorted(glob.glob(os.path.join(subject_dir, 'MNI*'))):
        trial_name = os.path.basename(trial_dir)
        trial_dest = os.path.join(subject_dest, trial_name)
        os.makedirs(trial_dest, exist_ok=True)


# Step 3: Average channels in same area and save
for subject_dir in tqdm(sorted(glob.glob(os.path.join('data', '*')))):
    for trial_dir in sorted(glob.glob(os.path.join(subject_dir, 'MNI*'))):
        channels = loadmat(os.path.join(trial_dir, 'channel_ctf_acc1.mat'))['Channel'][0]
        areas = {'MLT':[], 'MLF':[], 'MLC':[], 'MLP':[], 'MLO':[], 'MZ':[],
                'MRT':[], 'MRF':[], 'MRC':[], 'MRP':[], 'MRO':[]}
                
        for i, channel in enumerate(channels):
            channel_name = channel[5][0]
            if channel_name in channels_intersection:
                for area in areas.keys():
                    if channel_name.startswith(area):
                        areas[area].append(i)
        
        for epoch_address in sorted(glob.glob(os.path.join(trial_dir, 'data_block*'))):
            epoch_file = loadmat(epoch_address)['F']
            epoch_df = pd.DataFrame()

            for area in areas.keys():
                sum_channels = np.zeros((12000))
                for channel_index in areas[area]:
                    sum_channels += epoch_file[channel_index]
                
                mean_channels = sum_channels / len(areas[area])
                epoch_df[area] = pd.Series(mean_channels)

            epoch_df = epoch_df.transpose()
            relative_path = os.path.relpath(epoch_address, start='data')
            output_path = os.path.join(dest_base, os.path.splitext(relative_path)[0] + '.csv')
            epoch_df.to_csv(output_path, index=True, index_label='Areas')