from utils import utils
import pandas as pd
import numpy as np
import os

def summarize_metrics(metrics_all):
    """Return a summarized DataFrame of training, validation, and test metrics."""
    results = pd.DataFrame.from_dict(metrics_all, orient='index').transpose()

    def extract_metrics(prefix):
        cols = [f"{prefix}_{metric}" for metric in ['loss', 'accuracy', 'specificity', 'sensitivity', 'kappa']]
        return results[cols].set_axis(['loss', 'accuracy', 'specificity', 'sensitivity', 'kappa'], axis=1)

    train = extract_metrics('train')
    val = extract_metrics('val')
    test = extract_metrics('test')

    return pd.concat([train, val, test], axis=0).set_axis(['train', 'validation', 'test'], axis=0)


def summarize_attention(intermediate_outputs_pd):
    """Return mean and std from attention outputs, if available."""
    if intermediate_outputs_pd is not None and not intermediate_outputs_pd.empty:
        return intermediate_outputs_pd.describe().loc[['mean', 'std']]
    return None


def summarize_test_accs(test_names, test_accs):
    """Return a DataFrame of test accuracies indexed by subject name."""
    df = pd.DataFrame(test_names).set_index(0)
    df.index = df.index.map(os.path.basename)
    df['test_acc'] = test_accs
    df.index.name = None
    return df.sort_index()


def bucketize_accuracies(val_accs, test_accs):
    """Return binned accuracy counts for validation and test sets."""

    def count_bins(data):
        return [(data[(data[0] >= i) & (data[0] < i + 10)]).shape[0] for i in range(0, 101, 10)]

    df_val = pd.DataFrame(val_accs)
    df_test = pd.DataFrame(test_accs)

    bin_labels = [f"[{i}-{i+10})" for i in range(0, 101, 10)]
    val_counts = count_bins(df_val)
    test_counts = count_bins(df_test)

    df1 = pd.DataFrame([val_counts], columns=bin_labels)
    df2 = pd.DataFrame([test_counts], columns=bin_labels)

    return pd.concat([df1, df2]).set_axis(['validation', 'test'], axis=0)


def summarize_all(metrics_all, intermediate_outputs_pd, test_names, test_accs,
    val_accs, model_name="Draft", save=True, show_metrics=True,
    show_attention=True, show_test_acc=True, show_histogram=True):
    results_dir = os.path.join('results', model_name)
    os.makedirs(results_dir, exist_ok=True)

    metrics_df = summarize_metrics(metrics_all)
    if show_metrics:
        utils.smart_display(metrics_df)
    if save:
        metrics_df.to_csv(os.path.join(results_dir, "metrics.csv"), index=True)

    attention_df = summarize_attention(intermediate_outputs_pd)
    if attention_df is not None:
        if show_attention:
            utils.smart_display(attention_df)
        if save:
            attention_df.to_csv(os.path.join(results_dir, "attention.csv"), index=True)

    test_acc_df = summarize_test_accs(test_names, test_accs)
    if show_test_acc:
        utils.smart_display(test_acc_df)
    if save:
        test_acc_df.to_csv(os.path.join(results_dir, "test_accuracy_per_subject.csv"), index=True)

    histogram_df = bucketize_accuracies(val_accs, test_accs)
    if show_histogram:
        utils.smart_display(histogram_df)
    if save:
        histogram_df.to_csv(os.path.join(results_dir, "accuracy_histogram_bins.csv"), index=True)