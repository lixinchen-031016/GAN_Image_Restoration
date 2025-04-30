import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import numpy as np

def load_cifar100_with_mask():
    # 指定本地CIFAR-100数据集路径（需要用户提前准备好）
    file_path = '/Users/lixinchen/PycharmProjects/cnn_gan_test/cifar-100-python/'  # 请根据实际路径修改

    import os
    import pickle

    def load_batch(fpath):
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            data = d[b'data']
            labels = d[b'fine_labels']
            data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float32") / 255.0
            return data, np.array(labels)

    train_data, train_labels = load_batch(os.path.join(file_path, 'train'))
    test_data, test_labels = load_batch(os.path.join(file_path, 'test'))

    # 添加随机遮罩
    def add_random_mask(images=None, mask_size=8):
        """对指定图片或随机CIFAR-100图片应用遮罩"""
        if images is None:
            # 加载CIFAR-100数据集
            (train_data, _, _), _ = load_cifar100_with_mask()
            # 随机选择一张图片
            idx = np.random.randint(0, len(train_data))
            images = train_data[idx:idx+1]
        
        masked_images = []
        masks = []
        
        for img in images:
            # 随机选择遮罩位置
            top = np.random.randint(0, 32 - mask_size)
            left = np.random.randint(0, 32 - mask_size)
            
            # 创建遮罩版本
            masked_img = img.copy()
            masked_img[top:top+mask_size, left:left+mask_size] = 0
            
            # 创建二值遮罩矩阵（三通道）
            mask = np.ones((32, 32, 3))  # 改为三通道遮罩
            mask[top:top+mask_size, left:left+mask_size] = 0
            
            masked_images.append(masked_img)
            masks.append(mask)
        
        return np.array(masked_images), np.array(masks)

    # 生成带遮罩的训练数据
    x_train_masked, masks_train = add_random_mask(train_data)
    x_test_masked, masks_test = add_random_mask(test_data)

    # 修改返回值，使遮罩图像对应标签(y_train)，而非原始图像
    return (train_data, x_train_masked, train_labels), (test_data, x_test_masked, test_labels)  # 返回原始图像、遮罩图像、标签

# 可视化示例
import matplotlib.pyplot as plt

def show_samples(originals, masked):
    plt.figure(figsize=(10, 5))
    for i in range(5):
        # 原始图像
        plt.subplot(2, 5, i+1)
        plt.imshow(originals[i])
        plt.axis('off')

        # 遮罩图像
        plt.subplot(2, 5, i+6)
        plt.imshow(masked[i])
        plt.axis('off')
    plt.show()

# 在脚本中调用可视化函数
if __name__ == "__main__":
    # 正确接收全部六个返回值
    (original_train, masked_train, labels_train), (original_test, masked_test, labels_test) = load_cifar100_with_mask()
    
    # 展示前5个样本（原始图像和遮罩图像）
    show_samples(original_train[:5], masked_train[:5])

