import tensorflow as tf
from keras import layers, Model

def build_generator():
    """改进的生成器结构，用于图像修补"""
    inputs = tf.keras.Input(shape=(32, 32, 3))  # 输入带遮罩的图像
    masks = tf.keras.Input(shape=(32, 32, 1))   # 输入遮罩位置

    # 特征提取分支
    x = layers.Conv2D(64, (5, 5), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    # 添加残差块
    x_residual = x

    # 增加特征提取深度
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    # 新增残差块1
    residual = layers.Conv2D(128, (1, 1), padding='same')(x)  # 匹配通道数
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.add([x, residual])  # 残差连接
    
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    # 新增残差块2
    residual = layers.Conv2D(256, (1, 1), padding='same')(x)  # 匹配通道数
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.add([x, residual])  # 残差连接
    
    # 修改遮罩路径：增加下采样以匹配特征图尺寸
    m = layers.Conv2D(64, (5, 5), padding='same')(masks)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)

    # 添加下采样操作
    m = layers.Conv2D(64, (3, 3), strides=2, padding='same')(m)  # 第一次下采样到16x16
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)

    m = layers.Conv2D(64, (3, 3), strides=2, padding='same')(m)  # 第二次下采样到8x8
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)

    # 融合两种特征
    combined = layers.Concatenate()([x, m])  # 现在 x 和 m 的尺寸应该都是(None, 8, 8, ...) 

    # 使用空洞卷积扩大感受野
    for rate in [2, 4, 8]:
        combined = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=rate)(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Activation('relu')(combined)
        
        # 新增SE注意力模块
        se = layers.GlobalAveragePooling2D()(combined)
        se = layers.Dense(256//16, activation='relu')(se)
        se = layers.Dense(256, activation='sigmoid')(se)
        combined = layers.multiply([combined, se])
    
    # 上采样层重构图像（替换转置卷积为UpSampling+普通卷积组合）
    # 修改点1：替换第一次上采样
    x = layers.UpSampling2D(size=(2, 2))(combined)  # 替代Conv2DTranspose
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 修改点2：替换第二次上采样
    x = layers.UpSampling2D(size=(2, 2))(x)  # 替代Conv2DTranspose
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 残差连接融合
    x = layers.Cropping2D(cropping=((0, 0), (0, 0)))(x_residual)  # 调整尺寸匹配
    x = layers.Concatenate()([x, x_residual])
    
    # 精细化输出
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 输出层
    outputs = layers.Conv2D(3, (1, 1), padding='same', activation='sigmoid')(x)  # 使用1x1卷积进行特征组合
    
    return Model(inputs=[inputs, masks], outputs=outputs)