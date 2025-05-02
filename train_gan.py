# 在文件开头添加导入
import os
from tqdm import tqdm

# 在文件开头添加Metal设备配置
import tensorflow as tf

# 检查并配置Metal加速
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 启用内存增长而不是预分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 设置TensorFlow使用Metal GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Using Metal GPU:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")

from generator_model import build_generator
from discriminator_model import build_discriminator
from data_loader import load_cifar100_with_mask
import numpy as np

# 参数设置
BATCH_SIZE = 256
EPOCHS = 1500
INITIAL_LEARNING_RATE = 0.0001
MIN_LEARNING_RATE = 0.00001
SAVE_INTERVAL = 100
WARMUP_EPOCHS = int(EPOCHS * 0.1)
SMOOTH_START = 0.2  # 新增初始平滑系数
SMOOTH_END = 0.01   # 新增最终平滑系数

# 新增动态标签平滑系数计算函数
def smooth_schedule(epoch):
    if epoch < WARMUP_EPOCHS:
        return SMOOTH_START
    # 余弦衰减从WARMUP_EPOCHS到EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return SMOOTH_END + 0.5 * (SMOOTH_START - SMOOTH_END) * (1 + tf.cos(np.pi * progress))

# 学习率衰减函数（修改为包含预热阶段的余弦退火）
def decayed_learning_rate(epoch):
    # 预热阶段：线性增加学习率
    if epoch < WARMUP_EPOCHS:
        warmup_progress = epoch / WARMUP_EPOCHS
        return MIN_LEARNING_RATE + (INITIAL_LEARNING_RATE - MIN_LEARNING_RATE) * warmup_progress
    
    # 余弦退火阶段
    cosine_epoch = epoch - WARMUP_EPOCHS
    cosine_decay = tf.cos(np.pi * cosine_epoch / (EPOCHS - WARMUP_EPOCHS))
    return MIN_LEARNING_RATE + 0.5 * (INITIAL_LEARNING_RATE - MIN_LEARNING_RATE) * (1 + cosine_decay)

# 构建并编译判别器（关键修改：删除编译后的冻结操作）
generator = build_generator()
discriminator = build_discriminator()

# 新增：编译判别器
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE * 0.5,  # 判别器学习率为生成器的一半
        clipnorm=1.0  # 添加梯度裁剪
    ),
    loss={'validity': 'binary_crossentropy', 'classification': 'sparse_categorical_crossentropy'},
    metrics={'validity': 'accuracy', 'classification': 'accuracy'}
)

# 打印可训练变量信息
print(f"Discriminator trainable variables count: {len(discriminator.trainable_variables)}")
if len(discriminator.trainable_variables) == 0:
    print("警告：判别器没有可训练变量！请检查模型结构")

# 编译生成器
generator.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=INITIAL_LEARNING_RATE, 
    beta_1=0.5, 
    beta_2=0.9  # 调整二阶动量参数
))

# 修正联合模型构建部分（解决NameError）
# 定义生成器输入
masked_input = tf.keras.Input(shape=(32, 32, 3), name='masked_image')
noise_input = tf.keras.Input(shape=(32, 32, 1), name='noise_input')

# 生成假图像
fake_images = generator([masked_input, noise_input])

# 获取判别器输出
valid, aux = discriminator(fake_images)

# 构建联合模型时使用明确定义的输入
combined = tf.keras.Model(
    inputs=[masked_input, noise_input], 
    outputs=[valid, aux]
)

combined.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                 optimizer=tf.keras.optimizers.Adam(INITIAL_LEARNING_RATE, 0.5))

# 新增：显式设置判别器可训练状态
discriminator.trainable = True

# 加载数据
(train_data, x_train_masked, train_labels), (test_data, x_test_masked, test_labels) = load_cifar100_with_mask()

# 添加这一行以避免NameError
x_train_real = train_data  # 将原始未加遮罩的图像赋值给x_train_real

# 学习率衰减函数
def decayed_learning_rate(epoch):
    cosine_decay = 0.5 * (1 + tf.cos(np.pi * epoch / EPOCHS))
    return MIN_LEARNING_RATE + (INITIAL_LEARNING_RATE - MIN_LEARNING_RATE) * cosine_decay

# 修改点1：将train_step定义移到循环外部以避免重复追踪
@tf.function
def train_step(real_imgs, masked_imgs, epoch):
    # 每个epoch后更新学习率
    lr = decayed_learning_rate(epoch)
    generator.optimizer.learning_rate.assign(lr)
    discriminator.optimizer.learning_rate.assign(lr)

    # 确保判别器参数可训练
    discriminator.trainable = True  # 关键修改：每次训练判别器前显式启用
    print("实际可训练变量数量:", len(discriminator.trainable_variables))  # 添加调试信息

    # 训练判别器
    noise = tf.random.normal([BATCH_SIZE, 32, 32, 1])  # 噪声输入
    fake_imgs = generator([masked_imgs, noise], training=True)

    # 训练判别器
    with tf.GradientTape() as tape:
        d_real_validity, d_real_class = discriminator(tf.convert_to_tensor(real_imgs))
        d_loss_real_validity = tf.keras.losses.binary_crossentropy(np.ones((BATCH_SIZE, 1)), d_real_validity)
        d_loss_real_class = tf.keras.losses.sparse_categorical_crossentropy(np.zeros((BATCH_SIZE, 1)), d_real_class)
        d_loss_real = tf.reduce_mean(d_loss_real_validity + d_loss_real_class)

        d_fake_validity, d_fake_class = discriminator(tf.convert_to_tensor(fake_imgs))
        d_loss_fake_validity = tf.keras.losses.binary_crossentropy(
            tf.zeros_like(d_fake_validity) + smooth_schedule(epoch)/2,
            d_fake_validity
        )
        d_loss_fake_class = tf.keras.losses.sparse_categorical_crossentropy(np.zeros((BATCH_SIZE, 1)), d_fake_class)
        d_loss_fake = tf.reduce_mean(d_loss_fake_validity + d_loss_fake_class)

        total_d_loss = d_loss_real + d_loss_fake

    grads = tape.gradient(total_d_loss, discriminator.trainable_variables)
    grads = [g for g in grads if g is not None]
    clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    discriminator.optimizer.apply_gradients(zip(clipped_grads, discriminator.trainable_variables))

    # 训练生成器前冻结判别器
    discriminator.trainable = False  # 冻结判别器

    # 训练生成器（关键修改：直接计算损失而非使用train_on_batch）
    with tf.GradientTape() as tape:
        # 生成假图像（关键修改：直接使用前向传播）
        fake_imgs = generator([masked_imgs, noise], training=True)
        # 获取判别器输出
        valid_pred, class_pred = discriminator(fake_imgs)
        
        # 计算两种损失分量
        validity_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(valid_pred) - smooth_schedule(epoch)/2,  # 真实标签也添加动态平滑
            valid_pred
        )
        class_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.zeros((BATCH_SIZE, 1)), class_pred
        )
        
        # 添加图像重建损失
        reconstruction_loss = tf.reduce_mean(tf.abs(fake_imgs - real_imgs))
        total_g_loss = tf.reduce_mean(
            validity_loss + class_loss + 0.1 * reconstruction_loss  # 添加重建损失项
        )
        
    grads = tape.gradient(total_g_loss, generator.trainable_variables)
    clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    generator.optimizer.apply_gradients(zip(clipped_grads, generator.trainable_variables))

    # 训练生成器后恢复判别器可训练状态
    # 注意：这个恢复操作应放在生成器训练结束后、下一个判别器训练前
    # 可以考虑在下一次进入train_step之前由外部控制

    # 打印进度（修改损失显示方式）
    tf.print(f"Epoch {epoch+1}/{EPOCHS} | D Loss Real: {d_loss_real} | D Loss Fake: {d_loss_fake} | G Loss: {total_g_loss}")
    
    # 返回生成器总损失用于监控
    return total_g_loss

# 修改点2：移除模型保存操作到函数外部
# 删除以下代码：
# if (epoch+1) % SAVE_INTERVAL == 0:
#     generator.save_weights(f"models/generator_epoch_{epoch+1}.weights.h5")
#     discriminator.save_weights(f"models/discriminator_epoch_{epoch+1}.weights.h5")

# 在数据加载时添加预取和缓存优化
idx = np.random.randint(0, x_train_real.shape[0], BATCH_SIZE)
real_imgs = tf.data.Dataset.from_tensors(x_train_real[idx]).prefetch(tf.data.AUTOTUNE)
masked_imgs = tf.data.Dataset.from_tensors(x_train_masked[idx]).prefetch(tf.data.AUTOTUNE)

# 修改点3：重构训练循环（添加进度条和移除float32转换）
def train_step(real_imgs, masked_imgs, epoch):
    # 每个epoch后更新学习率
    lr = decayed_learning_rate(epoch)
    generator.optimizer.learning_rate.assign(lr)
    discriminator.optimizer.learning_rate.assign(lr)

    # 确保判别器参数可训练
    discriminator.trainable = True  # 关键修改：每次训练判别器前显式启用
    print("实际可训练变量数量:", len(discriminator.trainable_variables))  # 添加调试信息

    # 训练判别器
    noise = tf.random.normal([BATCH_SIZE, 32, 32, 1])  # 噪声输入
    fake_imgs = generator([masked_imgs, noise], training=True)

    # 训练判别器
    with tf.GradientTape() as tape:
        d_real_validity, d_real_class = discriminator(tf.convert_to_tensor(real_imgs))
        d_loss_real_validity = tf.keras.losses.binary_crossentropy(np.ones((BATCH_SIZE, 1)), d_real_validity)
        d_loss_real_class = tf.keras.losses.sparse_categorical_crossentropy(np.zeros((BATCH_SIZE, 1)), d_real_class)
        d_loss_real = tf.reduce_mean(d_loss_real_validity + d_loss_real_class)

        d_fake_validity, d_fake_class = discriminator(tf.convert_to_tensor(fake_imgs))
        d_loss_fake_validity = tf.keras.losses.binary_crossentropy(
            tf.zeros_like(d_fake_validity) + smooth_schedule(epoch)/2,
            d_fake_validity
        )
        d_loss_fake_class = tf.keras.losses.sparse_categorical_crossentropy(np.zeros((BATCH_SIZE, 1)), d_fake_class)
        d_loss_fake = tf.reduce_mean(d_loss_fake_validity + d_loss_fake_class)

        total_d_loss = d_loss_real + d_loss_fake

    grads = tape.gradient(total_d_loss, discriminator.trainable_variables)
    grads = [g for g in grads if g is not None]
    clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    discriminator.optimizer.apply_gradients(zip(clipped_grads, discriminator.trainable_variables))

    # 训练生成器前冻结判别器
    discriminator.trainable = False  # 冻结判别器

    # 训练生成器（关键修改：直接计算损失而非使用train_on_batch）
    with tf.GradientTape() as tape:
        # 生成假图像（关键修改：直接使用前向传播）
        fake_imgs = generator([masked_imgs, noise], training=True)
        # 获取判别器输出
        valid_pred, class_pred = discriminator(fake_imgs)
        
        # 计算两种损失分量
        validity_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(valid_pred) - smooth_schedule(epoch)/2,  # 真实标签也添加动态平滑
            valid_pred
        )
        class_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.zeros((BATCH_SIZE, 1)), class_pred
        )
        
        # 添加图像重建损失
        reconstruction_loss = tf.reduce_mean(tf.abs(fake_imgs - real_imgs))
        total_g_loss = tf.reduce_mean(
            validity_loss + class_loss + 0.1 * reconstruction_loss  # 添加重建损失项
        )
        
    grads = tape.gradient(total_g_loss, generator.trainable_variables)
    clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    generator.optimizer.apply_gradients(zip(clipped_grads, generator.trainable_variables))

    # 训练生成器后恢复判别器可训练状态
    # 注意：这个恢复操作应放在生成器训练结束后、下一个判别器训练前
    # 可以考虑在下一次进入train_step之前由外部控制

    # 打印进度（修改损失显示方式）
    tf.print(f"Epoch {epoch+1}/{EPOCHS} | D Loss Real: {d_loss_real} | D Loss Fake: {d_loss_fake} | G Loss: {total_g_loss}")
    
    # 返回生成器总损失用于监控
    return total_g_loss

# 修改点2：移除模型保存操作到函数外部
# 删除以下代码：
# if (epoch+1) % SAVE_INTERVAL == 0:
#     generator.save_weights(f"models/generator_epoch_{epoch+1}.weights.h5")
#     discriminator.save_weights(f"models/discriminator_epoch_{epoch+1}.weights.h5")

# 在数据加载时添加预取和缓存优化
idx = np.random.randint(0, x_train_real.shape[0], BATCH_SIZE)
real_imgs = tf.data.Dataset.from_tensors(x_train_real[idx]).prefetch(tf.data.AUTOTUNE)
masked_imgs = tf.data.Dataset.from_tensors(x_train_masked[idx]).prefetch(tf.data.AUTOTUNE)

# 修改点3：重构训练循环（添加最佳模型保存逻辑）
# 在训练循环前添加最佳损失跟踪变量
best_g_loss = float('inf')  # 添加变量初始化

for epoch in tqdm(range(EPOCHS), desc='Training GAN', unit='epoch'):
    current_lr = decayed_learning_rate(epoch)
    generator.optimizer.learning_rate.assign(current_lr)
    discriminator.optimizer.learning_rate.assign(current_lr * 0.5)  # 保持判别器学习率为生成器的一半
    
    epoch_g_losses = []
    
    # 执行训练步骤
    for r_imgs, m_imgs in zip(real_imgs, masked_imgs):
        g_loss = train_step(r_imgs, m_imgs, epoch)
        epoch_g_losses.append(g_loss.numpy())  # 收集生成器损失
    
    # 计算平均生成器损失
    avg_g_loss = np.mean(epoch_g_losses)
    
    # 保存最佳模型（当生成器损失降低时）
    if avg_g_loss < best_g_loss and epoch > 800:
        os.makedirs("models", exist_ok=True)
        generator.save_weights("models/generator_best.weights.h5")
        discriminator.save_weights("models/discriminator_best.weights.h5")
        best_g_loss = avg_g_loss
        tf.print(f"\n保存最佳模型（epoch {epoch+1} 损失：{avg_g_loss:.4f})")
    
    # 周期保存当前模型
    if (epoch+1) % SAVE_INTERVAL == 0:
        os.makedirs("models", exist_ok=True)
        generator.save_weights(f"models/generator_epoch_{epoch+1}.weights.h5")
        discriminator.save_weights(f"models/discriminator_epoch_{epoch+1}.weights.h5")

    # 添加学习率日志输出
    tf.print(f"当前学习率: 生成器={current_lr:.6f}, 判别器={current_lr*0.5:.6f}")
