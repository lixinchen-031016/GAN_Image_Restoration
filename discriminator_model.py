import tensorflow as tf
from tensorflow.keras import layers, Model

def build_discriminator():
    """改进的判别器结构，用于图像真假判断和分类"""
    inputs = tf.keras.Input(shape=(32, 32, 3))  # 正确输入形状
    
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
    
    # 添加全局特征提取
    global_features = layers.GlobalAveragePooling2D()(x)
    
    # 真假辨别分支
    validity_branch = layers.Dense(128)(global_features)  # 直接基于 global_features
    validity_branch = layers.LeakyReLU(negative_slope=0.2)(validity_branch)
    validity = layers.Dense(1, activation='sigmoid', name='validity')(validity_branch)
    
    # 分类分支
    class_branch = layers.Dense(256)(global_features)  # 同上，直接使用 global_features
    class_branch = layers.LeakyReLU(negative_slope=0.2)(class_branch)
    class_branch = layers.Dropout(0.5)(class_branch)
    classification = layers.Dense(100, activation='softmax', name='classification')(class_branch)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=[validity, classification])
    print("Discriminator summary:")
    model.summary()
    return model