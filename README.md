# CNN-GAN图像修复项目

## 项目概述
本项目基于生成对抗网络（GAN）实现图像修复功能，主要包含以下模块：
- 生成器模型（Generator）：学习修复被遮挡的图像区域
- 判别器模型（Discriminator）：判断图像真实性并进行分类
- 训练模块：实现对抗训练过程
- 推理模块：提供图像修复和结果可视化功能
- 数据加载器：处理CIFAR-100数据集并生成遮挡样本

## 文件结构
```
cnn_gan_test/ 
├── data_loader.py # 数据加载与预处理模块 
├── generator_model.py # 生成器网络定义 
├── discriminator_model.py # 判别器网络定义 
├── train_gan.py # 训练流程控制 
├── inference.py # 图像修复与可视化 
├── models/ # 预训练模型存储 
├── results/ # 修复结果保存 
└── README.md # 项目文档
```

## 详细功能说明

### 1. data_loader.py
#### `load_cifar100_with_mask()`
- 功能：加载CIFAR-100数据集并生成遮挡样本
- 处理流程：
  1. 从本地路径加载CIFAR-100训练集和测试集
  2. 对图像进行归一化处理（0-1范围）
  3. 添加随机矩形遮挡（mask_size=8）
  4. 生成三通道遮挡矩阵
- 返回：
  - (原始训练集, 遮挡训练集, 训练标签)
  - (原始测试集, 遮挡测试集, 测试标签)

#### `add_random_mask(images, mask_size=8)`
- 功能：生成渐变边缘遮挡
- 参数：
  - images: 输入图像数组
  - mask_size: 遮挡区域大小
- 特点：
  - 生成带有2像素渐变边缘的遮挡
  - 返回遮挡图像和对应的mask矩阵

### 2. generator_model.py
#### `build_generator()`
- 网络架构：


  输入层（32x32x3图像 + 32x32x1噪声） 
  ↓
  Conv2D(64,5x5) → BatchNorm → LeakyReLU
  ↓
  残差连接分支
  ↓
  下采样层（128→256通道） 
  ↓
  空洞卷积层（dilation_rate=2,4,8）
  ↓
  转置卷积上采样层（256→128通道）
  ↓
  残差连接融合
  ↓
  输出层（Conv2D 3通道，sigmoid激活）

- 特点：
  - 使用空洞卷积扩大感受野
  - 残差连接保留原始特征
  - 噪声输入增强生成多样性

### 3. discriminator_model.py
#### `build_discriminator()`
- 双分支架构：

  输入层（32x32x3图像）
  ↓
  Conv2D(64→128→256) + BatchNorm + LeakyReLU
  ↓
  GlobalAveragePooling
  ↓
  分支1：有效性判别（128→1神经元，sigmoid激活）
  分支2：图像分类（256→100神经元，softmax激活）
- 特点：
  - 联合训练真伪判别和图像分类
  - 全局平均池化替代全连接层
  - Dropout层防止过拟合

### 4. train_gan.py
#### 核心参数
```
python
BATCH_SIZE = 256  # 增大batch_size提升训练稳定性
EPOCHS = 1000      # 延长训练轮数
INITIAL_LEARNING_RATE = 0.0001  # 降低初始学习率
MIN_LEARNING_RATE = 0.00001      # 添加最小学习率限制
SMOOTH = 0.1  # 新增标签平滑系数
SAVE_INTERVAL = 100  # 新增模型保存间隔参数
WARMUP_EPOCHS = int(EPOCHS * 0.1)  # 新增预热周期参数（总训练周期的10%）
```
#### `train_step()`
- 训练流程：
  1. 余弦衰减调整学习率
  2. 判别器训练：
     - 计算真实/生成图像的损失
     - 梯度裁剪（clipnorm=1.0）
     - 应用标签平滑（SMOOTH=0.1）
  3. 生成器训练：
     - 组合对抗损失和L1重建损失
     - 冻结判别器参数
- 创新点：
  - 添加图像重建损失项（权重0.1）
  - 动态学习率调整策略

### 5. inference.py
#### `repair_specific_image()`
- 功能：完整图像修复流程
- 处理步骤：
  1. 图像分块处理（32x32）
  2. 并行修复每个图像块
  3. 重叠区域平均融合
  4. 结果后处理（0-255范围转换）

#### `process_and_stitch()`
- 关键算法：
  1. 为每个图像块生成正态分布噪声
  2. 使用生成器修复遮挡区域
  3. 创建计数矩阵记录像素处理次数
  4. 加权平均消除拼接痕迹

## 使用指南

### 训练模型
```
bash
python train_gan.py \
    --batch_size 256 \
    --epochs 500 \
    --learning_rate 0.0001
```
### 图像修复
```
python
from inference import repair_specific_image

generator = load_trained_generator("models/generator_epoch_500.weights.h5")
original, masked, repaired = repair_specific_image(
    generator, 
    image_path="input.jpg"
)
```
## 性能优化
- 使用Metal GPU加速（M1/M2芯片）
- 数据预取（tf.data.Dataset）
- 混合精度训练
- 梯度裁剪（clipnorm=1.0）

## 结果示例
![修复效果](results/repaired_sample.png)

这个程序 **不能** 在 CUDA 环境中直接运行，需要进行代码修改和环境配置调整。以下是详细的分析：

---

## ❌ 一、当前存在的问题

### 1. **Metal GPU 配置冲突（仅 macOS）**
在 `train_gan.py` 中有如下代码：
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Using Metal GPU:", gpus[0])
    except RuntimeError as e:
        print(e)
```


#### 问题说明：
- 这段代码是为了适配 Apple 的 **Metal GPU**（`tf-metal` 插件），而不是标准的 NVIDIA CUDA 环境。
- 如果你使用的是 NVIDIA GPU 和 CUDA，这段代码会导致 TensorFlow **无法识别你的 GPU设备**。

#### ✅ 解决方案：
删除或注释掉上述代码块，并确保使用标准的 TensorFlow GPU 初始化方式，例如：
```python
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)
```


---

### 2. **CUDA 支持版本不兼容**

#### 检查点：
- 当前项目依赖的 TensorFlow 版本是否支持你本地安装的 CUDA Toolkit？
- 请检查以下版本匹配关系：

| TensorFlow Version | Python Version | CUDA Toolkit | cuDNN |
|--------------------|----------------|---------------|--------|
| ≥ 2.10             | 3.7–3.10       | 11.2          | 8.1     |
| 2.11+              | 3.7–3.11       | 11.8          | 8.6     |

> ⚠️ 提示：TensorFlow 2.10 及以上版本不再支持 CUDA 11.2，默认使用 11.8。如果系统中安装了旧版本 CUDA，请升级驱动和工具链。

---


---

### 4. **数据加载部分未正确处理遮罩输入**

在 `data_loader.py` 中定义的 `add_random_mask` 函数返回值是 `(masked_images, masks)`，但 `load_cifar100_with_mask()` 返回了 `(train_data, x_train_masked, train_labels)`，即原始图像、遮罩图像、标签。

在训练时，生成器期望输入是 `(masked_input, noise_input)`，而实际传入的数据缺少明确的 `mask` 输入张量。

#### ✅ 解决方案：
确保你在训练过程中构造了完整的输入元组：
```python
for real_imgs, masked_imgs, labels in dataset:
    # 构造噪声输入
    noise = tf.random.normal([BATCH_SIZE, 32, 32, 1])
    # 使用 (masked_imgs, noise) 作为输入
    ...
```


---

## ✅ 二、可以正常工作的部分（无需修改）

### ✔️ 1. **模型结构兼容 CUDA**
- 所有的 Keras 层（如 `Conv2D`, `BatchNormalization`, `LeakyReLU`）都是基于 TensorFlow 实现的，天然支持 CUDA 加速。
- GAN 结构合理，适合图像修复任务。

### ✔️ 2. **训练框架兼容 GPU**
- 使用了 `tf.function` 和 `tf.GradientTape`，这些 API 是 TensorFlow 推荐用于 GPU 训练的方式。
- 已经使用 `Adam` 优化器并应用了梯度裁剪，有助于稳定训练。

---

## 🛠️ 三、推荐修改清单

| 文件名         | 修改内容                                                                 | 建议操作                     |
|----------------|--------------------------------------------------------------------------|------------------------------|
| `train_gan.py` | 删除或注释与 Metal 相关的 GPU 设置代码                                   | 注释或替换为标准 GPU 初始化  |
| `train_gan.py` | 在每次训练判别器前添加 `discriminator.trainable = True`                  | 补充代码                     |
| `train_gan.py` | 确保 `real_imgs` 和 `masked_imgs` 正确从数据集中提取                    | 检查数据管道                 |
| `data_loader.py` | 检查 `add_random_mask` 是否输出与模型输入一致的遮罩尺寸                   | 调整遮罩维度为 `(32, 32, 1)` |
| `generator_model.py` | 输入遮罩维度为 `(32, 32, 1)`，需确保数据集输出也是该格式                | 统一输入维度                 |

---

## ✅ 四、部署建议

### ✅ 安装步骤（Ubuntu/Windows）
```bash
# 安装与 CUDA 兼容的 TensorFlow
pip install tensorflow==2.12.0

# 查看可用 GPU 设备
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```


### ✅ 验证 CUDA 是否生效
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```


预期输出应类似：
```
Num GPUs Available: 1
```


---

## ✅ 总结

| 评估项               | 是否满足 |
|----------------------|----------|
| 是否能在 CUDA 环境下运行 | ❌ 否   |
| 是否只需小范围修改即可运行 | ✅ 是   |
| 是否包含 GPU 不兼容代码 | ✅ 是   |
| 模型结构是否合理       | ✅ 是   |
| 数据流是否清晰         | ✅ 是   |

---

## ✅ 最终结论

> **这个程序不能在 CUDA 环境中直接运行，但通过少量的代码修改（主要是 Metal GPU 配置和判别器冻结逻辑）就可以顺利运行。**

如果你完成上述修改后仍然遇到问题，欢迎提供具体的报错信息，我可以进一步帮你定位。
