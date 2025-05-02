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
        if images is None:
            idx = np.random.randint(0, len(train_data))
            images = train_data[idx:idx+1]

        masked_images = []
        masks = []

        for img in images:
            top = np.random.randint(0, 32 - mask_size)
            left = np.random.randint(0, 32 - mask_size)

            masked_img = img.copy()
            masked_img[top:top+mask_size, left:left+mask_size] = 0

            mask = np.ones((32, 32, 3))
            mask[top:top+mask_size, left:left+mask_size] = 0

            masked_images.append(masked_img)
            masks.append(mask)

        return np.array(masked_images), np.array(masks)

    x_train_masked, masks_train = add_random_mask(train_data)
    x_test_masked, masks_test = add_random_mask(test_data)

    # 修复返回值结构
    return (train_data, x_train_masked, train_labels), (test_data, x_test_masked, test_labels)

# 在脚本中调用可视化函数
if __name__ == "__main__":
    (original_train, masked_train, labels_train), (original_test, masked_test, labels_test) = load_cifar100_with_mask()
    show_samples(original_train[:5], masked_train[:5])

