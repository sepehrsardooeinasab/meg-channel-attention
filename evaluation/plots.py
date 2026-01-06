from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, loss_ylim=(-0.1, 1.5), acc_ylim=(-5, 105), train_color='tab:blue', val_color='tab:orange'):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))  # 2 columns

    # --- Loss Plot ---
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].plot(history.epoch, np.array(history.history['loss']), label='Train', color=train_color)
    axes[0].plot(history.epoch, np.array(history.history['val_loss']), label='Val', color=val_color)
    axes[0].set_ylim(loss_ylim)
    axes[0].xaxis.set_major_locator(MultipleLocator(10))
    axes[0].legend(loc='upper right')

    # --- Accuracy Plot ---
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].plot(history.epoch, np.array(history.history['accuracy']) * 100, label='Train', color=train_color)
    axes[1].plot(history.epoch, np.array(history.history['val_accuracy']) * 100, label='Val', color=val_color)
    axes[1].set_ylim(acc_ylim)
    axes[1].xaxis.set_major_locator(MultipleLocator(10))
    axes[1].yaxis.set_major_locator(MultipleLocator(10))
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_accuracy_histograms(val_acc_list, test_acc_list, color_val='skyblue', color_test='salmon'):
    bins = np.arange(0, 105, 10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # --- Validation Histogram ---
    axes[0].hist(np.array(val_acc_list), bins=bins, color=color_val, alpha=1)
    axes[0].set_title('Validation Accuracy')
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(np.arange(0, 101, 10))
    axes[0].set_xlim([0, 100])
    axes[0].set_yticks(np.arange(0, 51, 2))
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Test Histogram ---
    axes[1].hist(np.array(test_acc_list), bins=bins, color=color_test, alpha=1)
    axes[1].set_title('Test Accuracy')
    axes[1].set_xlabel('Accuracy (%)')
    axes[1].set_xticks(np.arange(0, 101, 10))
    axes[1].set_xlim([0, 100])
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()