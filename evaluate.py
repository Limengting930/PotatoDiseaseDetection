# 修改 evaluate.py 文件
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_model_and_evaluate(model_path, data_path):
    """
    加载已训练的模型并评估
    """
    # 动态导入自定义层
    custom_objects = {}
    
    try:
        # 尝试导入自定义层
        from models.vit_model import Patches, PatchEncoder
        custom_objects = {
            'Patches': Patches,
            'PatchEncoder': PatchEncoder
        }
        print("Successfully imported custom layers from models.vit_model")
    except Exception as e:
        print(f"Warning: Could not import custom layers directly: {e}")
        # 尝试另一种导入方式
        try:
            import importlib.util
            vit_model_path = os.path.join(os.path.dirname(__file__), 'models', 'vit_model.py')
            if os.path.exists(vit_model_path):
                spec = importlib.util.spec_from_file_location("vit_model", vit_model_path)
                vit_model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(vit_model_module)
                custom_objects = {
                    'Patches': vit_model_module.Patches,
                    'PatchEncoder': vit_model_module.PatchEncoder
                }
                print("Successfully imported custom layers using importlib")
            else:
                print(f"Warning: vit_model.py not found at {vit_model_path}")
        except Exception as e2:
            print(f"Warning: Could not import custom layers using importlib: {e2}")

    # 使用custom_object_scope加载包含自定义层的模型
    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")
    
    print(f"Loaded model from {model_path}")
    
    # 加载数据集
    from data_processing.data_loader import load_and_split_datasets
    train_ds, _, test_ds = load_and_split_datasets(data_path)
    
    # 获取类别名称
    class_names = None
    if hasattr(train_ds, 'class_names') and train_ds.class_names is not None:
        class_names = train_ds.class_names
    elif hasattr(test_ds, 'class_names') and test_ds.class_names is not None:
        class_names = test_ds.class_names
    else:
        # 如果无法从数据集获取，尝试从目录结构推断
        try:
            class_names = sorted(os.listdir(data_path))
            # 过滤掉非目录项（如系统文件）
            class_names = [name for name in class_names if os.path.isdir(os.path.join(data_path, name))]
        except:
            # 最后的备用方案
            class_names = [f"Class_{i}" for i in range(10)]  # 假设有10个类别
            print("Warning: Could not determine class names, using default names.")
    
    print(f"Using class names: {class_names}")
    
    # 评估模型
    return evaluate_model(model, test_ds, class_names)

def evaluate_model(model, dataset, class_names):
    """
    计算 Precision, Recall, F1-score, Accuracy 并打印
    绘制混淆矩阵
    """
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    acc = accuracy_score(y_true, y_pred)
    print(f"\n Accuracy: {acc:.4f}")

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return acc, cm

# 如果直接运行此脚本，则执行评估
if __name__ == "__main__":
    # 默认模型路径映射
    model_paths = {
        "resnet": "best_resnet_model.h5",
        "vit": "best_vit_model.h5",
        "hybrid": "best_hybrid_model.h5"
    }
    
    DATA_PATH = r"E:\dataset\PotatoGPT\minidata\potato"
    
    # 获取模型类型参数
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type in model_paths:
            MODEL_PATH = model_paths[model_type]
        else:
            MODEL_PATH = model_type  # 如果传入的是具体路径
    else:
        # 默认使用vit模型
        MODEL_PATH = "best_vit_model.h5"
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file {MODEL_PATH} not found.")
        print("Available models in current directory:")
        available_models = [f for f in os.listdir('.') if f.endswith('.h5')]
        for model in available_models:
            print(f"  - {model}")
        sys.exit(1)
    
    print(f"\n🔎 Loading model from {MODEL_PATH} and evaluating on test set...")
    try:
        load_model_and_evaluate(MODEL_PATH, DATA_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Tips:")
        print("1. Make sure you're running this script from the correct directory")
        print("2. Check if the model file exists")
        print("3. For ViT/Hybrid models, ensure all custom layers are properly defined")