import tensorflow as tf
from tensorflow.keras import layers, Model

def build_discriminator():
    """改进的判别器结构，添加自注意力机制"""
    inputs = tf.keras.Input(shape=(32, 32, 3))
    
    # 特征提取部分使用更深层的网络
    x = layers.Conv2D(64, (5, 5), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # 新增自注意力模块
    class SelfAttentionLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
        def build(self, input_shape):
            self.channels = input_shape[-1]
            self.q_conv = layers.Conv2D(self.channels//8, 1)
            self.k_conv = layers.Conv2D(self.channels//8, 1)
            self.v_conv = layers.Conv2D(self.channels, 1)
            self.add_layer = layers.Add()
            super().build(input_shape)
            
        def call(self, x):
            batch_size = tf.shape(x)[0]
            
            q = self.q_conv(x)
            k = self.k_conv(x)
            v = self.v_conv(x)
            
            # 使用动态reshape处理空间维度
            q = tf.reshape(q, [batch_size, -1, self.channels//8])
            k = tf.reshape(k, [batch_size, -1, self.channels//8])
            v = tf.reshape(v, [batch_size, -1, self.channels])
            
            attention = tf.matmul(q, k, transpose_b=True)
            attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(self.channels//8, tf.float32)))
            
            attended = tf.matmul(attention, v)
            attended = tf.reshape(attended, tf.shape(x))
            return self.add_layer([x, attended])
        
        def compute_output_shape(self, input_shape):
            return input_shape
    
    x = SelfAttentionLayer()(x)
    
    # 添加全局特征提取
    global_features = layers.GlobalAveragePooling2D()(x)
    
    global_features = layers.Flatten()(global_features)

    # 真假辨别分支
    validity_branch = layers.Dense(128)(global_features)  # 直接基于 global_features
    validity_branch = layers.LeakyReLU(negative_slope=0.2)(validity_branch)
    validity = layers.Dense(1, activation='sigmoid', name='validity')(validity_branch)
    
    # 分类分支
    class_branch = layers.Dense(256)(global_features)
    class_branch = layers.LeakyReLU(negative_slope=0.2)(class_branch)
    # 将普通Dropout改为空间Dropout（需保持4D输入）
    class_branch = layers.Reshape((1, 1, 256))(class_branch)  # 新增：将特征转换为4D张量
    class_branch = layers.SpatialDropout2D(0.5)(class_branch)
    class_branch = layers.Flatten()(class_branch)  # 新增：恢复为2D张量
    classification = layers.Dense(100, activation='softmax', name='classification')(class_branch)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=[validity, classification])
    print("Discriminator summary:")
    model.summary()
    return model