import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_hex
from utils import utils

def plot_attention_heatmap(x_series: pd.Series, 
                           use_channel_attention=True, 
                           show_hex=True):

    if not use_channel_attention:
        return None  # Do nothing if disabled

    # Create custom dark blue to dark red colormap
    blue_red_cmap = LinearSegmentedColormap.from_list(
        'blue_red_muted', ['#1f3b99', '#991f1f'])
    # blue_red_cmap = LinearSegmentedColormap.from_list(
    #     'blue_red_dynamic', ['#0000FF', '#FF0000'])
    # blue_red_cmap = LinearSegmentedColormap.from_list(
    #     'blue_red_soft', ['#6699FF', '#FF6666'])

    # Normalize values for color mapping
    min_val, max_val = x_series.min(), x_series.max()
    normalized = (x_series - min_val) / (max_val - min_val)

    # Generate hex color codes
    hex_colors = normalized.apply(lambda v: to_hex(blue_red_cmap(v)))

    # Build color summary table
    color_df = pd.DataFrame({
        col: [val, hex_color] 
        for col, val, hex_color in zip(x_series.index, x_series.values, hex_colors.values)},
            index=["Attention", "Hex"])

    if show_hex:
        utils.smart_display(color_df)

    # Prepare heatmap data
    df = pd.DataFrame([x_series.values], columns=x_series.index)
    vmin = round(df.values.min(), 3)
    vmax = round(df.values.max(), 3)
    ticks = np.round(np.linspace(vmin, vmax, num=5), 3)

    # Plot heatmap
    plt.figure(figsize=(max(6, len(x_series) * 0.6), 2))
    ax = sns.heatmap(
        df,
        cmap=blue_red_cmap,
        annot=True,
        fmt=".3f",
        linewidths=0.,
        linecolor='gray',
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={
            'orientation': 'horizontal',
            'shrink': 1,
            'aspect': 90,
            'pad': 0.1,
            'ticks': ticks})
    
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.yticks([])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return color_df if show_hex else None