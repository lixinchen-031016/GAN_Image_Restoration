import tensorflow as tf
from keras import layers, Model

def build_generator():
    """改进的生成器结构，用于图像修补"""
    inputs = tf.keras.Input(shape=(32, 32, 3))  # 输入带遮罩的图像
    masks = tf.keras.Input(shape=(32, 32, 1))   # 输入遮罩位置

    # 特征提取分支
    x = layers.Conv2D(64, (5, 5), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)  # 使用Swish激活函数

    # 添加残差块
    x_residual = x

    # 增加特征提取深度
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 新增残差块1
    residual = layers.Conv2D(128, (1, 1), padding='same')(x)  # 匹配通道数
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.add([x, residual])  # 残差连接
    
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 新增残差块2
    residual = layers.Conv2D(256, (1, 1), padding='same')(x)  # 匹配通道数
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.add([x, residual])  # 残差连接
    
    # 新增第三层下采样
    x = layers.Conv2D(512, (3, 3), strides=2, padding='same')(x)  # 新增下采样层
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 新增残差块3
    residual = layers.Conv2D(512, (1, 1), padding='same')(x)  # 匹配通道数
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.add([x, residual])  # 残差连接
    
    # 修改遮罩路径：增加下采样以匹配特征图尺寸
    m = layers.Conv2D(64, (5, 5), padding='same')(masks)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)

    # 添加下采样操作
    m = layers.Conv2D(64, (3, 3), strides=2, padding='same')(m)  # 第一次下采样到16x16
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)

    m = layers.Conv2D(64, (3, 3), strides=2, padding='same')(m)  # 第二次下采样到8x8
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)
    
    # 新增第三次下采样到4x4
    m = layers.Conv2D(64, (3, 3), strides=2, padding='same')(m)  # 新增下采样层
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)

    # 融合两种特征
    combined = layers.Concatenate()([x, m])  # 现在 x 和 m 的尺寸应该都是(None, 8, 8, ...) 

    # 使用空洞卷积扩大感受野
    for rate in [2, 4, 8]:
        combined = layers.Conv2D(512, (3, 3), padding='same', dilation_rate=rate)(combined)  # 增加通道数
        combined = layers.BatchNormalization()(combined)
        combined = layers.Activation('swish')(combined)
        
        # 新增自注意力模块
        class SelfAttentionBlock(layers.Layer):
            def call(self, x):
                batch_size = tf.shape(x)[0]
                channels = x.shape[-1]  # 使用静态通道数
                
                # 使用1x1卷积生成query/key/value
                q = layers.Conv2D(channels//8, 1)(x)
                k = layers.Conv2D(channels//8, 1)(x)
                v = layers.Conv2D(channels, 1)(x)

                # 使用动态reshape处理空间维度
                q = tf.reshape(q, [batch_size, -1, channels//8])
                k = tf.reshape(k, [batch_size, -1, channels//8])
                v = tf.reshape(v, [batch_size, -1, channels])

                # 计算注意力权重
                attention = tf.matmul(q, k, transpose_b=True)
                attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(channels//8, tf.float32)))

                # 应用注意力到value
                attended = tf.matmul(attention, v)
                
                # 恢复原始空间维度
                attended = tf.reshape(attended, tf.shape(x))
                return layers.Add()([x, attended])

            def compute_output_shape(self, input_shape):
                return input_shape  # 明确指定输出形状与输入相同
        
        combined = SelfAttentionBlock()(combined)
        
        # 新增SE注意力模块
        se = layers.GlobalAveragePooling2D()(combined)
        se = layers.Dense(512//16, activation='relu')(se)  # 增加通道数
        se = layers.Dense(512, activation='sigmoid')(se)
        combined = layers.multiply([combined, se])
    
    # 新增残差注意力模块
    class ResidualAttentionBlock(layers.Layer):
        def __init__(self, channels):
            super().__init__()
            self.channels = channels
            
        def build(self, input_shape):
            self.conv1 = layers.Conv2D(self.channels, (3, 3), padding='same')
            self.bn1 = layers.BatchNormalization()
            self.conv2 = layers.Conv2D(self.channels, (3, 3), padding='same')
            self.bn2 = layers.BatchNormalization()
            self.attention = SelfAttentionBlock()
            super().build(input_shape)
            
        def call(self, x):
            residual = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = layers.Activation('swish')(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = layers.Activation('swish')(x)
            x = self.attention(x)
            return layers.Add()([x, residual])
    
    # 添加多个残差注意力模块
    for _ in range(2):  # 添加两个残差注意力模块
        combined = ResidualAttentionBlock(512)(combined)
    
    # 上采样层重构图像（替换转置卷积为UpSampling+普通卷积组合）
    # 修改点1：替换第一次上采样
    x = layers.UpSampling2D(size=(2, 2))(combined)  # 替代Conv2DTranspose
    x = layers.Conv2D(512, (3, 3), padding='same')(x)  # 增加通道数
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 修改点2：替换第二次上采样
    x = layers.UpSampling2D(size=(2, 2))(x)  # 替代Conv2DTranspose
    x = layers.Conv2D(256, (3, 3), padding='same')(x)  # 增加通道数
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 新增第三次上采样
    x = layers.UpSampling2D(size=(2, 2))(x)  # 新增上采样层
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 残差连接融合
    x = layers.Cropping2D(cropping=((0, 0), (0, 0)))(x_residual)  # 调整尺寸匹配
    x = layers.Concatenate()([x, x_residual])
    
    # 精细化输出
    x = layers.Conv2D(256, (3, 3), padding='same')(x)  # 增加通道数
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 输出层
    outputs = layers.Conv2D(3, (1, 1), padding='same', activation='sigmoid')(x)  # 使用1x1卷积进行特征组合
    
    return Model(inputs=[inputs, masks], outputs=outputs)