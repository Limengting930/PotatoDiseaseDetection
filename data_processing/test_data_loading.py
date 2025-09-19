#用于测试数据是否加载成功
from data_processing.data_loader import load_and_split_datasets
import tensorflow as tf
import os

# 设置随机种子确保一致性
tf.random.set_seed(123)

# 数据路径
DATA_PATH = r"E:\dataset\PotatoGPT\minidata\potato"

print("="*50)
print("🔍 验证数据加载和划分...")
print("="*50)

# 检查数据路径是否存在
if not os.path.exists(DATA_PATH):
    print(f"❌ 错误: 数据路径不存在: {DATA_PATH}")
    exit(1)

print(f"正在检查数据路径: {DATA_PATH}")
if os.path.isdir(DATA_PATH):
    subdirs = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    print(f"找到子目录: {subdirs}")
    
    # 检查每个子目录中的文件
    for subdir in subdirs:
        subdir_path = os.path.join(DATA_PATH, subdir)
        files = os.listdir(subdir_path)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  {subdir}: {len(files)} 个文件 ({len(image_files)} 个图像文件)")
else:
    print(f"❌ 错误: {DATA_PATH} 不是一个目录")
    exit(1)

# 加载和划分数据集
print("\n开始加载和划分数据集...")
try:
    train_ds, valid_ds, test_ds = load_and_split_datasets(DATA_PATH)
    print("✅ 数据加载完成")
except Exception as e:
    print(f"❌ 数据加载过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 获取类别名称
class_names = train_ds.class_names
print(f"📊 类别数量: {len(class_names)}")
print(f"🏷️  类别名称: {class_names}")

# 检查数据形状
print("\n📈 数据形状信息:")
try:
    print("检查训练集...")
    for i, (images, labels) in enumerate(train_ds.take(2)):
        print(f"   训练集第{i+1}个批次:")
        print(f"     图像形状: {images.shape}")
        print(f"     标签形状: {labels.shape}")
        print(f"     图像值范围: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
        if i >= 0:  # 只检查一个批次
            break

    print("检查验证集...")
    for i, (images, labels) in enumerate(valid_ds.take(2)):
        print(f"   验证集第{i+1}个批次:")
        print(f"     图像形状: {images.shape}")
        print(f"     标签形状: {labels.shape}")
        if i >= 0:  # 只检查一个批次
            break

    print("检查测试集...")
    for i, (images, labels) in enumerate(test_ds.take(2)):
        print(f"   测试集第{i+1}个批次:")
        print(f"     图像形状: {images.shape}")
        print(f"     标签形状: {labels.shape}")
        if i >= 0:  # 只检查一个批次
            break
            
except Exception as e:
    print(f"❌ 获取数据形状时发生错误: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ 数据加载和划分验证完成!")
print("="*50)