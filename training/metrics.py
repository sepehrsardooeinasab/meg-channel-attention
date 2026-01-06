from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, accuracy_score
import numpy as np

def calculate_metrics_1fold(model, X_train, Y_train, X_val, Y_val, X_test, Y_test):
    # Evaluate: get loss and accuracy
    results_train = model.evaluate(X_train, Y_train, verbose=0)
    train_loss = results_train[0]
    train_acc = results_train[1]

    results_val = model.evaluate(X_val, Y_val, verbose=0)
    val_loss = results_val[0]
    val_acc = results_val[1]

    results_test = model.evaluate(X_test, Y_test, verbose=0)
    test_loss = results_test[0]
    test_acc = results_test[1]

    # Predict labels (after thresholding at 0.5)
    Y_predict_train = np.rint(model.predict(X_train, verbose=0))
    Y_predict_val = np.rint(model.predict(X_val, verbose=0))
    Y_predict_test = np.rint(model.predict(X_test, verbose=0))

    # Confusion matrices
    cm_train = confusion_matrix(Y_train, Y_predict_train, labels=[0, 1])
    cm_val = confusion_matrix(Y_val, Y_predict_val, labels=[0, 1])
    cm_test = confusion_matrix(Y_test, Y_predict_test, labels=[0, 1])

    # Metrics
    metrics = {}
    metrics['train_loss'] = train_loss
    metrics['train_accuracy'] = train_acc * 100
    metrics['train_specificity'] = cm_train[0][0] / (cm_train[0][0] + cm_train[0][1]) * 100
    metrics['train_sensitivity'] = cm_train[1][1] / (cm_train[1][0] + cm_train[1][1]) * 100
    metrics['train_kappa'] = cohen_kappa_score(Y_train, Y_predict_train)

    metrics['val_loss'] = val_loss
    metrics['val_accuracy'] = val_acc * 100
    metrics['val_specificity'] = cm_val[0][0] / (cm_val[0][0] + cm_val[0][1]) * 100 if np.min(Y_val) == 0 else None
    metrics['val_sensitivity'] = cm_val[1][1] / (cm_val[1][0] + cm_val[1][1]) * 100 if np.max(Y_val) == 1 else None
    metrics['val_kappa'] = cohen_kappa_score(Y_val, Y_predict_val)

    metrics['test_loss'] = test_loss
    metrics['test_accuracy'] = test_acc * 100
    metrics['test_specificity'] = cm_test[0][0] / (cm_test[0][0] + cm_test[0][1]) * 100 if np.min(Y_test) == 0 else None
    metrics['test_sensitivity'] = cm_test[1][1] / (cm_test[1][0] + cm_test[1][1]) * 100 if np.max(Y_test) == 1 else None
    metrics['test_kappa'] = cohen_kappa_score(Y_test, Y_predict_test)

    return metrics

def calculate_metrics_kfold(metrics_history):
    metrics = {}
    for key in metrics_history[0].keys():
        values = []
        for dic in metrics_history:
            values.append(dic[key])
        values = np.array(values)
        values = values[values != None]
        metrics[key] = "{:.3f} \u00B1 {:.3f}".format(np.mean(values), np.std(values))
    return metrics