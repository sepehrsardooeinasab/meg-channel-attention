import tensorflow as tf

class LossWithAttentionEntropy(tf.keras.losses.Loss):
    def __init__(self, model, base_loss_fn=tf.keras.losses.BinaryCrossentropy()):
        super().__init__()
        self.model = model
        self.base_loss_fn = base_loss_fn

    def call(self, y_true, y_pred):
        base_loss = self.base_loss_fn(y_true, y_pred)
        attention_entropy = self.model.get_layer("channel_attention").attention_entropy_loss
        return base_loss + attention_entropy