import tensorflow as tf
from tensorflow.keras import layers, Model

def build_discriminator():
    """
    构建改进的判别器模型，包含以下核心组件：
    1. 深层卷积网络提取多尺度特征
    2. 自注意力机制增强长距离依赖建模
    3. 多尺度特征融合提升判别能力
    4. 双任务输出（真伪判断+分类）
    """
    # 输入层：定义32x32 RGB图像输入
    inputs = tf.keras.Input(shape=(32, 32, 3))  # 输入维度[None,32,32,3]
    
    # 特征提取部分使用更深层的网络（共4个卷积块）
    # 第1卷积块：64通道，5x5卷积核，输出尺寸16x16
    x = layers.Conv2D(64, (5, 5), strides=2, padding='same')(inputs)  # 大卷积核捕获基础特征
    x = layers.BatchNormalization()(x)  # 加速训练并稳定分布
    x = layers.LeakyReLU(negative_slope=0.2)(x)  # 保留部分负值避免神经元死亡
    x = layers.Dropout(0.3)(x)  # 防止特征过拟合
    
    # 第2卷积块：128通道，3x3卷积核，输出尺寸8x8
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)  # 进一步下采样
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # 第3卷积块：256通道，3x3卷积核，输出尺寸4x4
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # 第4卷积块：512通道，3x3卷积核，输出尺寸2x2（新增下采样层）
    x = layers.Conv2D(512, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # 改进的自注意力模块（通道缩减比改为4）
    class SelfAttentionLayer(layers.Layer):
        """自注意力机制模块，增强长距离依赖建模能力
        参数:
            reduction_ratio: 通道压缩比例，控制计算复杂度
        """
        def __init__(self, reduction_ratio=8, **kwargs):
            super().__init__(**kwargs)
            self.reduction_ratio = reduction_ratio
            
        def build(self, input_shape):
            """初始化Q/K/V变换层"""
            self.channels = input_shape[-1]
            # 1x1卷积降低通道数，减少计算量
            self.q_conv = layers.Conv2D(self.channels//self.reduction_ratio, 1)
            self.k_conv = layers.Conv2D(self.channels//self.reduction_ratio, 1)
            self.v_conv = layers.Conv2D(self.channels, 1)
            # 可学习的缩放参数，初始为0，保证训练稳定性
            self.gamma = tf.Variable(0.0, trainable=True)  
            self.add_layer = layers.Add()
            super().build(input_shape)
            
        def call(self, x):
            """前向传播计算注意力机制"""
            batch_size = tf.shape(x)[0]
            
            # 生成查询/键/值向量
            q = self.q_conv(x)  # [None, H, W, C/r]
            k = self.k_conv(x)  # [None, H, W, C/r]
            v = self.v_conv(x)  # [None, H, W, C]
            
            # 动态reshape处理空间维度
            q = tf.reshape(q, [batch_size, -1, self.channels//self.reduction_ratio])
            k = tf.reshape(k, [batch_size, -1, self.channels//self.reduction_ratio])
            v = tf.reshape(v, [batch_size, -1, self.channels])
            
            # 计算注意力权重
            attention = tf.matmul(q, k, transpose_b=True)  # [N, HW, HW]
            # 使用温度系数调节分布陡峭程度
            attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(self.channels//self.reduction_ratio, tf.float32)))
            
            # 应用注意力到值向量
            attended = tf.matmul(attention, v)  # [N, HW, C]
            attended = tf.reshape(attended, tf.shape(x))  # [N, H, W, C]
            return (1 - self.gamma) * x + self.gamma * attended  # 可学习的残差连接
        
        def compute_output_shape(self, input_shape):
            return input_shape
    
    x = SelfAttentionLayer(reduction_ratio=4)(x)  # 增加注意力模块的分辨率
    
    # 后续特征处理
    x = layers.Conv2D(512, (3, 3), padding='same')(x)  # 512通道卷积增强特征表达
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    # 多尺度特征提取（调整池化配置）
    # 全局平均池化提取整体特征
    pool1 = layers.GlobalAveragePooling2D()(x)  # [None, 512]
    # 2x2池化保留局部结构信息
    pool2 = layers.AveragePooling2D(pool_size=(2, 2))(x)  # [None,1,1,512]
    # 1x1池化保留原始空间信息
    pool3 = layers.AveragePooling2D(pool_size=(1, 1))(x)  # [None,2,2,512]
    
    # 特征融合
    pool2 = layers.Flatten()(pool2)  # [None, 512]
    pool3 = layers.Flatten()(pool3)  # [None, 2048]
    # 多尺度特征拼接：512(global)+512(local)+2048(original) = 3072维
    global_features = layers.Concatenate()([pool1, pool2, pool3])  

    # 真假辨别分支
    validity_branch = layers.Dense(256)(global_features)
    validity_branch = layers.LeakyReLU(negative_slope=0.2)(validity_branch)
    validity_branch = layers.Dense(128)(validity_branch)  # 新增特征变换层
    validity_branch = layers.LeakyReLU(negative_slope=0.2)(validity_branch)
    # 输出层：sigmoid激活表示真假概率
    validity = layers.Dense(1, activation='sigmoid', name='validity')(validity_branch)
    
    # 分类分支
    class_branch = layers.Dense(512)(global_features)
    class_branch = layers.LeakyReLU(negative_slope=0.2)(class_branch)
    class_branch = layers.Dense(256)(class_branch)  # 新增特征降维层
    class_branch = layers.LeakyReLU(negative_slope=0.2)(class_branch)
    class_branch = layers.Reshape((1, 1, 256))(class_branch)  # 转换为4D张量
    class_branch = layers.SpatialDropout2D(0.5)(class_branch)  # 空间Dropout增强泛化
    class_branch = layers.Flatten()(class_branch)  # 恢复为2D张量
    # 输出层：softmax激活表示类别概率分布
    classification = layers.Dense(100, activation='softmax', name='classification')(class_branch)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=[validity, classification])
    print("Discriminator summary:")
    model.summary()
    return model