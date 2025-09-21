import tensorflow as tf
from tensorflow import keras
import os
import random
import numpy as np

def load_and_split_datasets(data_dir, image_size=(224, 224), batch_size=2, split_ratios=(0.8, 0.1, 0.1)):
    # 设置随机种子确保划分一致性
    random.seed(123)
    np.random.seed(123)
    tf.random.set_seed(123)
    
    # 获取所有类别目录
    class_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"找到 {len(class_dirs)} 个类别: {class_dirs}")
    
    train_files = []
    val_files = []
    test_files = []
    
    # 为每个类别分别划分数据
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        # 获取所有文件并过滤图像文件
        all_files = os.listdir(class_path)
        files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"类别 '{class_name}' 找到 {len(all_files)} 个文件，其中 {len(files)} 个图像文件")
        
        if len(files) == 0:
            print(f"警告: 类别 '{class_name}' 没有找到图像文件!")
            continue
            
        # 构建完整路径
        files = [os.path.join(class_path, f) for f in files]
            
        # 按固定顺序排序文件以确保一致性
        files.sort()
        # 随机打乱文件列表（但使用固定种子确保一致性）
        random.shuffle(files)
        
        # 按比例划分
        n_total = len(files)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        # 确保至少有一个样本用于验证和测试（如果可能）
        if n_train == 0 and n_total > 0:
            n_train = 1
        if n_val == 0 and n_total > n_train:
            n_val = 1
            
        train_files.extend([(f, class_dirs.index(class_name)) for f in files[:n_train]])
        val_files.extend([(f, class_dirs.index(class_name)) for f in files[n_train:n_train+n_val]])
        test_files.extend([(f, class_dirs.index(class_name)) for f in files[n_train+n_val:]])
    
    print(f"\n数据集划分完成:")
    print(f"  训练集: {len(train_files)} 个样本")
    print(f"  验证集: {len(val_files)} 个样本")
    print(f"  测试集: {len(test_files)} 个样本")
    
    # 创建tf.data.Dataset
    def create_dataset(file_label_list, dataset_name):
        if len(file_label_list) == 0:
            print(f"警告: {dataset_name} 数据集为空!")
            # 创建一个空的数据集作为占位符
            def empty_generator():
                return
                yield  # 这里永远不会执行
            
            empty_dataset = tf.data.Dataset.from_generator(
                empty_generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, len(class_dirs)), dtype=tf.float32)
                )
            )
            return empty_dataset.batch(batch_size)
        
        def generator():
            for file_path, label in file_label_list:
                yield file_path, label
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        def process_path(file_path, label):
            # 读取图像
            image = tf.io.read_file(file_path)
            # 使用更具体的解码函数
            try:
                image = tf.image.decode_image(image, channels=3)
            except:
                # 如果decode_image失败，尝试其他解码方法
                try:
                    image = tf.image.decode_jpeg(image, channels=3)
                except:
                    try:
                        image = tf.image.decode_png(image, channels=3)
                    except:
                        print(f"无法解码图像: {file_path}")
                        # 返回一个默认图像
                        image = tf.zeros([*image_size, 3], dtype=tf.uint8)
            
            # 检查图像形状
            image_shape = tf.shape(image)
            # 如果图像没有有效的形状，使用默认图像
            if tf.reduce_any(tf.equal(image_shape, 0)):
                print(f"警告: 图像 {file_path} 形状无效: {image_shape}")
                image = tf.zeros([*image_size, 3], dtype=tf.uint8)
            
            # 确保图像是3D张量
            image = tf.ensure_shape(image, [None, None, 3])
            image = tf.image.resize(image, image_size)
            image = tf.cast(image, tf.float32) / 255.0  # 归一化
            
            # 创建one-hot编码
            label = tf.one_hot(label, depth=len(class_dirs))
            return image, label
        
        dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_ds = create_dataset(train_files, "训练集")
    val_ds = create_dataset(val_files, "验证集")
    test_ds = create_dataset(test_files, "测试集")
    
    # 为训练集添加属性以兼容现有代码
    train_ds.class_names = class_dirs
    
    return train_ds, val_ds, test_ds