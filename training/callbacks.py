from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import tensorflow as tf

def callbacks(model_name, iteration_number, mode=0, verbose=True):
    class DelayedEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self, monitor='val_loss', patience=30, start_epoch=30, restore_best_weights=True, verbose=1):
            super(DelayedEarlyStopping, self).__init__()
            self.monitor = monitor
            self.patience = patience
            self.start_epoch = start_epoch
            self.restore_best_weights = restore_best_weights
            self.verbose = verbose
            self.wait = 0
            self.stopped_epoch = 0
            self.best_weights = None
            self.best = float('inf')

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(self.monitor)
            if epoch < self.start_epoch:
                return
            if current is None:
                return

            if current < self.best:
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    if self.verbose:
                        print(f"\nEpoch {self.stopped_epoch+1}: early stopping (delayed)")
                    if self.restore_best_weights and self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    self.model.stop_training = True
    
    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, start_lr_decay, lr_decay_step, lr_decay_factor, start_val_save):
            self.start_lr_decay = start_lr_decay
            self.lr_decay_step = lr_decay_step
            self.lr_decay_factor = lr_decay_factor
            self.start_val_save = start_val_save
            self.best_val_accuracy = 0
            self.best_weights = None
    
        def on_epoch_end(self, epoch, logs=None):
            if epoch>=self.start_val_save and logs["val_accuracy"]>=self.best_val_accuracy:
                self.best_weights = self.model.get_weights()
                self.best_val_accuracy = logs["val_accuracy"]
        
        def on_train_end(self, logs=None):
            if self.best_weights!=None:
                self.model.set_weights(self.best_weights)

    # def schedule(epoch, learning_rate):
    #     if epoch>=self.start_lr_decay and epoch%self.lr_decay_step==0:
    #         learning_rate *= lr_decay_factor 
    #     return learning_rate

    CustomCallback1 = CustomCallback(30, 30, 0.5, 90)
    CustomCallback2 = CustomCallback(50, 50, 0.25, 1000)
    # LearningRateScheduler1 = LearningRateScheduler(schedule, verbose=False)
    EarlyStopping1 = DelayedEarlyStopping(monitor='val_loss', patience=30, verbose=verbose, restore_best_weights=True, start_epoch=30)
    ReduceLROnPlateau1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=verbose)

    if mode==1:
        return [CustomCallback1]
    elif mode==2:
        return [EarlyStopping1, ReduceLROnPlateau1]
    elif mode==3:
        return [CustomCallback2]
    else:
        return None