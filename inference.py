import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from generator_model import build_generator
import os
# 新增导入语句
from data_loader import load_cifar100_with_mask
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim  # 新增导入

def load_trained_generator(weight_path):
    """加载生成器权重"""
    generator = build_generator()
    generator.load_weights(weight_path)
    return generator

def repair_image(generator, masked_img):
    """使用生成器修复图像"""
    # 假设噪声尺寸与训练时一致
    noise = tf.random.normal([1, 32, 32, 1])  # batch_size=1
    
    # 模型预测
    repaired_img = generator.predict([np.expand_dims(masked_img, axis=0), noise])
    return np.squeeze(repaired_img, axis=0)  # 移除batch维度

def show_repair_result(original, masked, repaired, psnr_value, ssim_value):  # 修改函数签名
    """可视化修复效果"""
    plt.figure(figsize=(15, 5))

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')

    # 遮罩图像
    plt.subplot(1, 3, 2)
    plt.title("Masked Image")
    plt.imshow(masked)
    plt.axis('off')

    # 修复结果
    plt.subplot(1, 3, 3)
    plt.title(f"Repaired Image\nPSNR: {psnr_value:.2f} dB\nSSIM: {ssim_value:.4f}")  # 添加指标显示
    plt.imshow(repaired)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def split_image_into_patches(image, patch_size=(32, 32)):
    """将图像分割为patch_size大小的小块"""
    patches = []
    h, w, c = image.shape
    ph, pw = patch_size
    
    for i in range(0, h, ph):
        for j in range(0, w, pw):
            # 确保不超出边界
            if i+ph <= h and j+pw <= w:
                patch = image[i:i+ph, j:j+pw]
                patches.append(patch)
    
    return patches, (h, w, c)

def process_and_stitch(patches, mask_patches, generator):  # 新增mask_patches参数
    """添加重叠区域处理以减少拼接痕迹"""
    processed_patches = []
    
    for patch, mask_patch in zip(patches, mask_patches):  # 使用传入的mask patch
        # 删除固定种子设置，允许不同patch使用不同噪声
        # 修改噪声生成方式，使其与训练分布一致
        noise = np.random.normal(0, 1, (1, 32, 32, 1))  # 每次生成新噪声
        
        # 直接使用传入的mask patch
        repaired_patch = generator.predict([np.expand_dims(mask_patch, axis=0), noise])
        processed_patches.append(np.squeeze(repaired_patch, axis=0))
    
    # 获取第一个小块的尺寸
    ph, pw, pc = processed_patches[0].shape
    
    # 计算输出尺寸
    total_h = ((patches[0].shape[0] - 1) // ph + 1) * ph
    total_w = ((patches[0].shape[1] - 1) // pw + 1) * pw
    
    # 初始化输出图像和计数器
    output_img = np.zeros((total_h, total_w, pc))
    count = np.zeros((total_h, total_w, pc))  # 用于计数每个像素被处理的次数
    
    # 将处理后的小块拼接回来
    idx = 0
    for i in range(0, total_h, ph):
        for j in range(0, total_w, pw):
            output_img[i:i+ph, j:j+pw] += processed_patches[idx]
            count[i:i+ph, j:j+pw] += 1
            idx += 1
            
    # 避免除以零
    count[count == 0] = 1
    
    # 取平均值得到最终图像
    return output_img / count

def repair_specific_image(generator, image=None, image_path=None, mask_image=None):  # 新增mask_image参数
    """修复指定图片，支持直接传入图像数组或图片路径"""
    if image is None and image_path is None:
        raise ValueError("必须提供图像数组或图片路径")
    if mask_image is None:  # 新增遮罩图像校验
        raise ValueError("必须提供遮罩图像")
        
    if image is None:
        # 读取图片
        img = plt.imread(image_path)
        
        # 如果是透明通道图片，仅保留RGB通道
        if img.shape[2] == 4:
            img = img[:, :, :3]
            
        # 调整到[0,1]范围
        img = img.astype(np.float32) / 255.0
    else:
        img = image
    
    # 分割图像为32x32的小块
    patches, original_shape = split_image_into_patches(img)
    mask_patches, _ = split_image_into_patches(mask_image)  # 分割遮罩图像
    
    # 处理每个小块并拼接结果
    repaired_img = process_and_stitch(patches, mask_patches, generator)
    
    # 获取原始图像尺寸
    orig_h, orig_w = original_shape[:2]
    
    # 裁剪到原始尺寸
    repaired_img = repaired_img[:orig_h, :orig_w, :]
    
    # 将修复后的图像从[0,1]范围转换为[0,255]并转为8位整数
    repaired_img = (repaired_img * 255).astype(np.uint8)
    
    # 返回原始图像、遮罩图像和修复后的图像（直接使用传入的遮罩图像）
    masked_img = mask_image[:orig_h, :orig_w, :]  # 使用传入遮罩的裁剪版本
    return img, masked_img, repaired_img

if __name__ == "__main__":
    # 路径到保存的权重文件（修改为正确的生成器路径）
    generator_weight_path = "models/generator_epoch_1000.weights.h5"  # 将discriminator改为generator
    
    # 加载训练好的生成器
    generator = load_trained_generator(generator_weight_path)
    
    # 示例：从CIFAR-100数据集中加载图片
    (_, _, _), (original_test, masked_test, labels_test) = load_cifar100_with_mask()
    
    # 选择测试集中第一张图片
    test_idx = 0
    test_img = original_test[test_idx]
    test_mask = masked_test[test_idx]  # 获取对应的遮罩图像

    # 修复数据集中的图片（新增传入遮罩图像）
    original, masked, repaired = repair_specific_image(generator, image=test_img, mask_image=test_mask)
    
    # 计算PSNR和SSIM
    original_uint8 = (original * 255).astype(np.uint8)  # 转换为uint8格式
    psnr_value = psnr(original_uint8, repaired)  # 计算PSNR
    ssim_value = ssim(original_uint8, repaired, multichannel=True, channel_axis=2)  # 计算SSIM
    print(f"PSNR: {psnr_value:.2f} dB\nSSIM: {ssim_value:.4f}")
    
    # 展示和保存结果
    show_repair_result(original, masked, repaired, psnr_value, ssim_value)  # 传递指标参数

    # 确保结果目录存在
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)  # 如果目录已存在，不会报错
    
    # 保存修复后的图像
    plt.imsave(os.path.join(result_dir, "repaired_sample.png"), repaired)