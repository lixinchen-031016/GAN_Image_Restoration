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
BATCH_SIZE = 256         # 训练批量大小
EPOCHS = 500             # 总训练轮数
INITIAL_LEARNING_RATE = 0.0001  # 初始学习率
MIN_LEARNING_RATE = 0.00001      # 最小学习率
SAVE_INTERVAL = 50      # 模型保存间隔
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


