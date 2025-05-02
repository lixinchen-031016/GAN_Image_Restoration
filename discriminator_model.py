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
    
    # 新增卷积层
    x = layers.Conv2D(512, (3, 3), strides=2, padding='same')(x)  # 新增下采样层
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # 改进的自注意力模块
    class SelfAttentionLayer(layers.Layer):
        def __init__(self, reduction_ratio=8, **kwargs):
            super().__init__(**kwargs)
            self.reduction_ratio = reduction_ratio
            
        def build(self, input_shape):
            self.channels = input_shape[-1]
            self.q_conv = layers.Conv2D(self.channels//self.reduction_ratio, 1)
            self.k_conv = layers.Conv2D(self.channels//self.reduction_ratio, 1)
            self.v_conv = layers.Conv2D(self.channels, 1)
            self.gamma = tf.Variable(0.0, trainable=True)  # 可学习的缩放参数
            self.add_layer = layers.Add()
            super().build(input_shape)
            
        def call(self, x):
            batch_size = tf.shape(x)[0]
            
            q = self.q_conv(x)
            k = self.k_conv(x)
            v = self.v_conv(x)
            
            # 使用动态reshape处理空间维度
            q = tf.reshape(q, [batch_size, -1, self.channels//self.reduction_ratio])
            k = tf.reshape(k, [batch_size, -1, self.channels//self.reduction_ratio])
            v = tf.reshape(v, [batch_size, -1, self.channels])
            
            attention = tf.matmul(q, k, transpose_b=True)
            attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(self.channels//self.reduction_ratio, tf.float32)))
            
            attended = tf.matmul(attention, v)
            attended = tf.reshape(attended, tf.shape(x))
            return (1 - self.gamma) * x + self.gamma * attended  # 可学习的残差连接
        
        def compute_output_shape(self, input_shape):
            return input_shape
    
    x = SelfAttentionLayer(reduction_ratio=4)(x)  # 增加注意力模块的分辨率
    
    # 添加更深层的特征提取
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    # 修改空间金字塔池化层配置（原pool_size=4会导致负维度）
    pool1 = layers.GlobalAveragePooling2D()(x)
    pool2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    pool3 = layers.AveragePooling2D(pool_size=(1, 1))(x)  # 修改为1x1池化
    
    # 多尺度特征融合
    pool2 = layers.Flatten()(pool2)
    pool3 = layers.Flatten()(pool3)
    global_features = layers.Concatenate()([pool1, pool2, pool3])

    # 真假辨别分支
    validity_branch = layers.Dense(256)(global_features)  # 增加神经元数量
    validity_branch = layers.LeakyReLU(negative_slope=0.2)(validity_branch)
    validity_branch = layers.Dense(128)(validity_branch)  # 新增层
    validity_branch = layers.LeakyReLU(negative_slope=0.2)(validity_branch)
    validity = layers.Dense(1, activation='sigmoid', name='validity')(validity_branch)
    
    # 分类分支
    class_branch = layers.Dense(512)(global_features)  # 增加神经元数量
    class_branch = layers.LeakyReLU(negative_slope=0.2)(class_branch)
    class_branch = layers.Dense(256)(class_branch)  # 新增层
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