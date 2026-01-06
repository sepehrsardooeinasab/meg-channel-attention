import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation, AveragePooling2D,
    Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D,
    SpatialDropout2D, Dropout, Add, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal, Zeros, HeNormal
from tensorflow.keras.regularizers import l2

class TemporalAvgAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(TemporalAvgAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.time_dim = input_shape[2]
        self.shared_dense_one = Dense(self.time_dim // self.ratio, activation='relu',
                                    kernel_initializer='he_normal', use_bias=True)
        self.shared_dense_two = Dense(self.time_dim, activation='sigmoid',
                                    kernel_initializer='he_normal', use_bias=True)
        super(TemporalAvgAttention, self).build(input_shape)

    def call(self, inputs):
        # Pool across channel and feature axis
        avg_pool = tf.reduce_mean(inputs, axis=[1, 3])  # -> (Batch, Time)

        # Dense attention
        x = self.shared_dense_one(avg_pool)
        x = self.shared_dense_two(x)  # (Batch, Time)

        # Expand dims
        x = tf.expand_dims(x, axis=1)  # (Batch, 1, Time)
        x = tf.expand_dims(x, axis=-1)  # (Batch, 1, Time, 1)

        return inputs * x
    
class TemporalDualAttention	(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(TemporalDualAttention	, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.time_dim = input_shape[2]  # (B, C, T, D)
        self.shared_dense_one = Dense(self.time_dim // self.ratio, activation='relu',
                                    kernel_initializer='he_normal', use_bias=True)
        self.shared_dense_two = Dense(self.time_dim, kernel_initializer='he_normal', use_bias=True)
        super(TemporalDualAttention	, self).build(input_shape)

    def call(self, inputs):
        # (B, C, T, D) input
        avg_pool = tf.reduce_mean(inputs, axis=[1, 3], keepdims=False)  # (B, T)
        max_pool = tf.reduce_max(inputs, axis=[1, 3], keepdims=False)   # (B, T)

        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))  # (B, T)
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))  # (B, T)

        attention = tf.nn.softmax(avg_out + max_out, axis=1)  # (B, T)
        attention = tf.expand_dims(tf.expand_dims(attention, axis=1), axis=-1)  # (B, 1, T, 1)

        return inputs * attention
    
class ChannelAvgAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, entropy_reg_weight=0.01, **kwargs):
        super(ChannelAvgAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.entropy_reg_weight = entropy_reg_weight
        self.attention_entropy_loss = 0

    def build(self, input_shape):
        self.channel_dim = input_shape[1]  # (B, C, T, 1)
        self.shared_dense_one = Dense(self.channel_dim // self.ratio, activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True)
        self.shared_dense_two = Dense(self.channel_dim,
                                    kernel_initializer='he_normal',
                                    use_bias=True)
        super(ChannelAvgAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[2, 3])  # (B, C)

        out = self.shared_dense_two(self.shared_dense_one(avg_pool))  # (B, C)

        attention = tf.nn.softmax(out, axis=1)  # normalize over channels (C)

        if self.entropy_reg_weight > 0:
            entropy = -tf.reduce_sum(attention * tf.math.log(attention + 1e-6), axis=1)  # (B,)
            entropy = tf.reduce_mean(entropy)  # scalar
            self.attention_entropy_loss = self.entropy_reg_weight * entropy
        else:
            self.attention_entropy_loss = 0.0

        attention = tf.expand_dims(tf.expand_dims(attention, axis=-1), axis=-1)  # (B, C, 1, 1)

        return inputs * attention, attention

class ChannelDualAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, entropy_reg_weight=0.01, **kwargs):
        super(ChannelDualAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.entropy_reg_weight = entropy_reg_weight
        self.attention_entropy_loss = 0

    def build(self, input_shape):
        self.channel_dim = input_shape[1]  # channel-first: (B, C, T, 1)
        self.shared_dense_one = Dense(self.channel_dim // self.ratio, activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True)
        self.shared_dense_two = Dense(self.channel_dim, kernel_initializer='he_normal',
                                    use_bias=True)
        super(ChannelDualAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[2, 3], keepdims=True)  # (B, C, T, 1) -> (B, C, 1, 1)
        max_pool = tf.reduce_max(inputs, axis=[2, 3], keepdims=True)   # (B, C, 1, 1)
        avg_pool = tf.squeeze(avg_pool, axis=[2, 3])
        max_pool = tf.squeeze(max_pool, axis=[2, 3])
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))
        #attention = tf.nn.sigmoid(avg_out + max_out)  # (B, C)
        attention = tf.nn.softmax(avg_out + max_out, axis=1)         # Normalize across channels
        
        if self.entropy_reg_weight>0:
            entropy = -tf.reduce_sum(attention * tf.math.log(attention + 1e-6), axis=1)  # (B,)
            entropy = tf.reduce_mean(entropy)  # scalar
            self.attention_entropy_loss = self.entropy_reg_weight * entropy
        else:
            self.attention_entropy_loss = 0.0  # scalar
            
        attention = tf.expand_dims(tf.expand_dims(attention, axis=-1), axis=-1)  # (B, C, 1, 1)
        return inputs * attention, attention
          
def MEGNet(Chans, Samples, config):
    dropoutRate = config.get("dropoutRate", 0.5)
    kernLength = config.get("kernLength", 32)
    F1 = config.get("F1", 10)
    D = config.get("D", 2)
    norm_rate = config.get("norm_rate", 0.25)
    dropoutType = config.get("dropoutType", 'Dropout')
    activeFunc = config.get("activeFunc", 'elu')
    bnMomentum = config.get("bnMomentum", 0.9)
    bnEpsilon = config.get("bnEpsilon", 1e-3)
    use_channel_attention = config.get("use_channel_attention", False)
    use_temporal_attention = config.get("use_temporal_attention", True)
    reg_weight = config.get("reg_weight", 0.0)
    channel_ratio = config.get("channel_ratio", 8)
    temporal_ratio = config.get("temporal_ratio", 3)

    F2 = F1 * D
    dropoutType = Dropout if dropoutType == 'Dropout' else SpatialDropout2D
    activeFunc = tf.nn.silu if activeFunc == 'silu' else 'elu'

    input1 = Input(shape=(Chans, Samples, 1))

    if use_channel_attention:
        channel_attention_layer = ChannelAvgAttention(ratio=channel_ratio, entropy_reg_weight=reg_weight, name='channel_attention')
        t1, attention_out = channel_attention_layer(input1)
        t1 = BatchNormalization(momentum=bnMomentum, epsilon=bnEpsilon, axis=-1)(t1)
    else:
        t1 = input1
        attention_out = tf.ones_like(tf.reduce_mean(input1, axis=[2, 3], keepdims=True))

    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(t1)
    block1 = BatchNormalization(momentum=bnMomentum, epsilon=bnEpsilon, axis=-1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(momentum=bnMomentum, epsilon=bnEpsilon, axis=-1)(block1)
    block1 = Activation(activeFunc)(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(momentum=bnMomentum, epsilon=bnEpsilon, axis=-1)(block2)
    block2 = Activation(activeFunc)(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    if use_temporal_attention:
        block2 = TemporalAvgAttention(ratio=temporal_ratio)(block2)

    flatten = Flatten(name='flatten')(block2)
    dense = Dense(1, name='dense', kernel_constraint=max_norm(norm_rate), kernel_regularizer=tf.keras.regularizers.l2(1e-4))(flatten)
    sigmoid = Activation('sigmoid', name='sigmoid')(dense)

    model = Model(inputs=input1, outputs=sigmoid)
    model.attention_extractor = Model(inputs=input1, outputs=attention_out)
    return model

def dual_global_pool(inputs):
    gap = GlobalAveragePooling2D(name='gap')(inputs)
    gmp = GlobalMaxPooling2D(name='gmp')(inputs)
    return Concatenate()([gap, gmp])
    
def depthwise_res_block(inputs, F2, kernel_size=(1, 16), name_prefix='res_block'):
    x = SeparableConv2D(F2, kernel_size, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)

    residual = Conv2D(F2, (1, 1), padding='same', use_bias=False)(residual)
    residual = BatchNormalization()(residual)

    x = Add()([x, residual])
    x = Activation('relu')(x)
    return x