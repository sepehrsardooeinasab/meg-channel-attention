import numpy as np
import pandas as pd
import tensorflow as tf
from data import load_data
from models import meg_models
from utils import utils
from evaluation import plots
from training import metrics, callbacks, loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import os 


def kfold_train_multi_val(all_subjects_address, config):
    intermediate_outputs = []
    metrics_history = []
    test_names = []
    test_accs = []
    val_accs = []

    num_epoch_subjects = config.get("num_epoch_subjects", 60)
    model_name = config.get("model_name", "Draft")
    num_val_subjects = config.get("num_val_subjects", 2)
    epochs = config.get("epochs", 100)
    learning_rate = config.get("learning_rate", 1e-3)
    batch_size = config.get("batch_size", 16)
    verbose = config.get("verbose", False)
    mode = config.get("mode", 4)
    show_summary = config.get("show_summary", True)
    show_plots = config.get("show_plots", True)

    channel_names = ["MLT", "MLF", "MLC", "MLP", "MLO", "MZ", "MRT", "MRF", "MRC", "MRP", "MRO"]
    all_trials_address = utils.get_trials_from_subjects(all_subjects_address)

    if not os.path.exists('weights'):
        os.mkdir('weights')
    if not os.path.exists(os.path.join('weights', model_name)):
        os.mkdir(os.path.join('weights', model_name))

    X, Y = load_data.load_data(all_trials_address)
    num_subjects = len(all_subjects_address)

    for test_index in range(num_subjects):
        X_train, Y_train = [], []
        X_val, Y_val = [], []
        X_test, Y_test = [], []

        # Select validation subjects
        val_indices = []
        offset = 1
        while len(val_indices) < num_val_subjects:
            candidate = (test_index + offset) % num_subjects
            if candidate != test_index:
                val_indices.append(candidate)
            offset += 1

        train_indices = [i for i in range(num_subjects) if i not in ([test_index] + val_indices)]

        # Prepare training, validation, and test sets
        for idx in train_indices:
            X_train.append(X[num_epoch_subjects * idx:num_epoch_subjects * (idx + 1), :, :])
            Y_train.append(Y[num_epoch_subjects * idx:num_epoch_subjects * (idx + 1)])

        x_percent_train = 30
        for idx in val_indices:
            subject_data = X[num_epoch_subjects * idx:num_epoch_subjects * (idx + 1), :, :]
            subject_labels = Y[num_epoch_subjects * idx:num_epoch_subjects * (idx + 1)]

            total_samples = subject_data.shape[0]
            perm = np.random.permutation(total_samples)
            subject_data = subject_data[perm]
            subject_labels = subject_labels[perm]

            train_samples = int(np.floor((x_percent_train / 100.0) * total_samples))
            if train_samples <= 0 or train_samples >= total_samples:
                train_samples = max(1, total_samples - 1)

            X_train_part = subject_data[:train_samples]
            Y_train_part = subject_labels[:train_samples]
            X_val_part = subject_data[train_samples:]
            Y_val_part = subject_labels[train_samples:]

            X_train.append(X_train_part)
            Y_train.append(Y_train_part)
            X_val.append(X_val_part)
            Y_val.append(Y_val_part)

        X_test.append(X[num_epoch_subjects * test_index:num_epoch_subjects * (test_index + 1), :, :])
        Y_test.append(Y[num_epoch_subjects * test_index:num_epoch_subjects * (test_index + 1)])
        test_names.append(all_subjects_address[test_index].split('/')[-1])

        # Stack and normalize
        X_train = np.concatenate(X_train, axis=0)
        Y_train = np.concatenate(Y_train)
        X_val = np.concatenate(X_val, axis=0)
        Y_val = np.concatenate(Y_val)
        X_test = np.concatenate(X_test, axis=0)
        Y_test = np.concatenate(Y_test)

        X_train, X_val, X_test = load_data.normalize(X_train, X_val, X_test, min_max=False)

        X_train = np.transpose(X_train,(0, 2, 1))
        X_val   = np.transpose(X_val,  (0, 2, 1))
        X_test  = np.transpose(X_test, (0, 2, 1))

        tf.keras.backend.clear_session()
        model = meg_models.MEGNet(X_train.shape[1], X_train.shape[2], config)

        loss_fn = loss.LossWithAttentionEntropy(model) if utils.has_layer(model, "channel_attention") else 'binary_crossentropy'
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_fn, metrics=['accuracy'])

        if test_index == 0:
            initial_weights_path = os.path.join('weights', model_name, 'InitialWeight.h5')
            model.save_weights(initial_weights_path)
            if show_summary:
                try:
                    image_path = os.path.join("models", "model.png")
                    plot_model(model, to_file=image_path, show_shapes=True, dpi=300)#show_layer_names=True)
                    from IPython.display import display, Image
                    display(Image(filename=image_path, width=500))
                except Exception as e:
                    model.summary()

        model_history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=(X_val, Y_val),
            shuffle=True,
            callbacks=callbacks.callbacks(model_name, test_index, mode=mode, verbose=verbose),
            workers=4,
            use_multiprocessing=True,
            max_queue_size=20)

        iter_weights_path = os.path.join('weights', model_name, f'Iter{test_index}.h5')
        model.save_weights(iter_weights_path)
        
        metrics_1fold = metrics.calculate_metrics_1fold(model, X_train, Y_train, X_val, Y_val, X_test, Y_test)
        metrics_history.append(metrics_1fold)

        if show_plots:
            df_metrics = pd.DataFrame([{
                'Iteration': test_index,
                'Train Acc': round(metrics_1fold['train_accuracy'], 3),
                'Val Acc': round(metrics_1fold['val_accuracy'], 3),
                'Test Acc': round(metrics_1fold['test_accuracy'], 3)}])
            
            df_metrics.index.name = None

            try:
                from IPython.display import display
                display(df_metrics.style.format({
                    'Train Acc': '{:.3f}',
                    'Val Acc': '{:.3f}',
                    'Test Acc': '{:.3f}'}).hide(axis='index'))
            except:
                print(df_metrics.to_string(index=False))

            plots.plot_history(model_history, train_color='black', val_color='crimson')

        val_accs.append(metrics_1fold['val_accuracy'])
        test_accs.append(metrics_1fold['test_accuracy'])

        # Extract normalized attention
        try:
            att_weights = model.attention_extractor.predict(X_test, verbose=0, batch_size=batch_size)
            att_weights = np.squeeze(att_weights)
            if np.all(np.abs(att_weights - 1.0) < 1e-5):
                raise ValueError("Attention output is all ones")
            intermediate_outputs.append(att_weights)
        except Exception as e:
            pass

    intermediate_outputs_pd = None
    if intermediate_outputs:
        intermediate_outputs = np.vstack(intermediate_outputs)
        intermediate_outputs_pd = pd.DataFrame(intermediate_outputs, columns=channel_names)

    return (metrics.calculate_metrics_kfold(metrics_history),
        val_accs,
        test_accs,
        test_names,
        intermediate_outputs_pd)