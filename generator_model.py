import tensorflow as tf
from keras import layers, Model

def build_generator():
    """
    改进的生成器结构，采用多尺度特征融合和注意力机制，包含：
    1. 特征提取分支（含多级下采样）
    2. 遮罩处理分支
    3. 特征融合与空洞卷积
    4. 注意力增强模块
    5. 多阶段上采样重构
    """
    # 输入层：带遮罩的32x32 RGB图像
    inputs = tf.keras.Input(shape=(32, 32, 3))  # 主输入维度[None,32,32,3]
    masks = tf.keras.Input(shape=(32, 32, 1))   # 遮罩输入维度[None,32,32,1]

    # 特征提取分支（逐步下采样）
    # 初始卷积提取基础特征
    x = layers.Conv2D(64, (5, 5), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)  # Swish激活函数提升非线性表达能力

    # 保存初始特征用于后续残差连接
    x_residual = x

    # 增加特征提取深度（共3次下采样）
    # 第1下采样：128通道，3x3卷积，输出尺寸16x16
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 残差块1
    residual = layers.Conv2D(128, (1, 1), padding='same')(x)  # 1x1卷积匹配通道数
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.add([x, residual])  # 残差连接保留细节信息
    
    # 第2下采样：256通道，3x3卷积，输出尺寸8x8
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 残差块2
    residual = layers.Conv2D(256, (1, 1), padding='same')(x)  # 通道匹配
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.add([x, residual])  # 残差连接提升梯度流动
    
    # 第3下采样：512通道，3x3卷积，输出尺寸4x4
    x = layers.Conv2D(512, (3, 3), strides=2, padding='same')(x)  # 新增下采样层
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 残差块3
    residual = layers.Conv2D(512, (1, 1), padding='same')(x)  # 通道匹配
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.add([x, residual])  # 残差连接保留深层特征
    
    # 遮罩路径处理（与特征路径同步下采样）
    # 初始卷积处理遮罩信息
    m = layers.Conv2D(64, (5, 5), padding='same')(masks)  # 初始卷积
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)

    # 第1下采样到16x16
    m = layers.Conv2D(64, (3, 3), strides=2, padding='same')(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)

    # 第2下采样到8x8
    m = layers.Conv2D(64, (3, 3), strides=2, padding='same')(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)
    
    # 新增第3次下采样到4x4
    m = layers.Conv2D(64, (3, 3), strides=2, padding='same')(m)  # 保持空间尺寸一致
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)

    # 特征融合（通道拼接）
    # 合并特征和遮罩信息，维度[None,4,4,512+64=576]
    combined = layers.Concatenate()([x, m])  

    # 残差注意力模块（结合局部和全局特征）
    class ResidualAttentionBlock(layers.Layer):
        def __init__(self, channels):
            super().__init__()
            self.channels = channels
            
        def build(self, input_shape):
            # 卷积堆叠
            self.conv1 = layers.Conv2D(self.channels, (3, 3), padding='same')
            self.bn1 = layers.BatchNormalization()
            self.conv2 = layers.Conv2D(self.channels, (3, 3), padding='same')
            self.bn2 = layers.BatchNormalization()
            # 注意力增强
            self.attention = SelfAttentionBlock()
            super().build(input_shape)
            
        def call(self, x):
            residual = x
            # 特征变换
            x = self.conv1(x)
            x = self.bn1(x)
            x = layers.Activation('swish')(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = layers.Activation('swish')(x)
            # 注意力增强
            x = self.attention(x)
            return layers.Add()([x, residual])  # 残差连接保留原始信息
    
    # 添加多个残差注意力模块（从2个增加到3个）
    for _ in range(3):  # 增加残差注意力模块数量
        combined = ResidualAttentionBlock(combined.shape[-1])(combined)  # 使用实际通道数替换固定值 512
        
    # 空洞卷积扩大感受野（多尺度特征提取）
    for rate in [2, 4, 8]:
        # 第一次卷积
        combined = layers.Conv2D(512, (3, 3), padding='same', dilation_rate=rate)(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Activation('swish')(combined)

        # 第二次卷积（新增）
        combined = layers.Conv2D(512, (3, 3), padding='same', dilation_rate=rate)(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Activation('swish')(combined)

        # 自注意力模块增强特征交互
        combined = SelfAttentionBlock()(combined)
        
        # SE注意力模块（通道注意力）
        se = layers.GlobalAveragePooling2D()(combined)  # 压缩空间维度
        se = layers.Dense(512//16, activation='relu')(se)  # 降维到32维
        se = layers.Dense(512, activation='sigmoid')(se)  # 恢复通道维度
        combined = layers.multiply([combined, se])  # 通道加权：[None,4,4,512] .* [None,512]
    
    # 上采样层重构图像（使用UpSampling+普通卷积替代转置卷积）
    # 第1上采样：4x4 → 8x8
    x = layers.UpSampling2D(size=(2, 2))(combined)
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 第2上采样：8x8 → 16x16
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 第3上采样：16x16 → 32x32
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 残差连接融合
    # 调整尺寸匹配
    x = layers.Cropping2D(cropping=((0, 0), (0, 0)))(x_residual)  # 不改变尺寸
    # 融合局部细节和全局特征：[None,32,32,128] + [None,32,32,64]
    x = layers.Concatenate()([x, x_residual])  
    
    # 精细化输出处理
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # 输出层（1x1卷积进行特征组合）
    # 使用sigmoid将像素值限制在0-1范围
    outputs = layers.Conv2D(3, (1, 1), padding='same', activation='sigmoid')(x)  # 输出修复后的图像
    
    return Model(inputs=[inputs, masks], outputs=outputs)

# 将SelfAttentionBlock类定义提前至全局作用域
class SelfAttentionBlock(layers.Layer):
    def call(self, x):
        batch_size = tf.shape(x)[0]
        channels = x.shape[-1]

        q = layers.Conv2D(channels//8, 1)(x)  # 确保是可训练卷积层
        k = layers.Conv2D(channels//8, 1)(x)
        v = layers.Conv2D(channels, 1)(x)

        # 动态reshape处理空间维度
        q = tf.reshape(q, [batch_size, -1, channels//8])
        k = tf.reshape(k, [batch_size, -1, channels//8])
        v = tf.reshape(v, [batch_size, -1, channels])

        # 计算注意力权重
        attention = tf.matmul(q, k, transpose_b=True)
        attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(channels//8, tf.float32)))

        # 应用注意力到value
        attended = tf.matmul(attention, v)
        attended = tf.reshape(attended, tf.shape(x))
        return layers.Add()([x, attended])  # 残差连接保留原始信息

    def compute_output_shape(self, input_shape):
        return input_shape  # 明确指定输出形状与输入相同
