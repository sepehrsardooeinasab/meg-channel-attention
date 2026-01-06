import pandas as pd
import tensorflow as tf
import os
import numpy as np
import random
import warnings
import seaborn as sns
import sys
import glob


def has_layer(model, layer_name):
    try:
        model.get_layer(layer_name)
        return True
    except ValueError:
        return False
    

def set_style():
    try:
        # Check if running in IPython/Jupyter
        from IPython import get_ipython
        ip = get_ipython()
        if ip:
            ip.run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': None}")
    except:
        pass 

    pd.set_option('display.float_format','{:,.3f}'.format)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 11)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.precision', 3)
    sns.set(style='whitegrid')


def disable_log():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings('ignore')


def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
#     tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def check_versions(show_executable=False):
    if show_executable:
        print(sys.executable)
    print(f"Python {sys.version}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    print("GPU is", "available" if len(tf.config.list_physical_devices('GPU'))>0 else "NOT AVAILABLE")


def shuffle_subjects(all_subjects_address):
    hc = list()
    adhd = list()
    for subject_address in all_subjects_address:
        if 108<=int((subject_address.split('/')[-1])[-3:])<=133:
            adhd.append(subject_address)
        else:
            hc.append(subject_address)

    random.shuffle(adhd)
    random.shuffle(hc)

    shuffled_all_subjects_address = list()
    for i in range(len(hc)):
        shuffled_all_subjects_address.append(hc[i])
        shuffled_all_subjects_address.append(adhd[i])

    assert len(shuffled_all_subjects_address)==50
    assert len(adhd)==25
    assert len(hc)==25
        
    return shuffled_all_subjects_address


def get_trials_from_subjects(all_subjects_address):
    all_trials_address = [glob.glob(subjects_address + '/MNI*')[0] for subjects_address in all_subjects_address]
    
    for trial_address in all_trials_address:
        assert len(glob.glob(trial_address + '/*'))==60

    return all_trials_address


def smart_display(*args):
    try:
        from IPython.display import display
        for arg in args:
            display(arg)
    except ImportError:
        for arg in args:
            print(arg.to_string(index=False) if isinstance(arg, pd.DataFrame) else str(arg))


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_epoch_activity(csv_path, highlight_channels=None, highlight_colors=None,
    scale_factor=1e14, smoothing_window=10, figsize=(12, 5)):

    # ---- Load and Process ----
    an_epoch = pd.read_csv(csv_path, index_col=0)
    scaled_epoch = an_epoch * scale_factor
    smoothed_epoch = scaled_epoch.rolling(window=smoothing_window, axis=0, min_periods=1).mean()
    smoothed_epoch = smoothed_epoch.transpose()

    # ---- Plotting ----
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    # Loop through channels in original order
    for ch in smoothed_epoch.columns:
        is_highlighted = highlight_channels and ch in highlight_channels

        smoothed_epoch[ch].plot(
            ax=ax,
            label=ch if is_highlighted else None,  # only label highlighted
            linewidth=2 if is_highlighted else 1,
            alpha=1.0 if is_highlighted else 0.4,
            color=(highlight_colors.get(ch, 'black') if is_highlighted else 'gray'),
            zorder=3 if is_highlighted else 1)

    # ---- Final Styling ----
    ax.set_xlabel("Time Index", fontsize=14)
    ax.set_ylabel(f"Amplitude (×1e{int(np.log10(scale_factor))})", fontsize=14)

    ax.legend(title=6*" "+"Channels"+6*" ", loc='upper left', bbox_to_anchor=(1.01, 1))

    ax.grid(True)
    plt.tight_layout()
    plt.show()
