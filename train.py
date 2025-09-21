# 修改 train.py 文件
from data_processing.data_loader import load_and_split_datasets
from data_processing.data_augment import get_data_augmentation
from model_training.train_resnet import train_resnet
from model_training.train_vit import train_vit
from model_training.train_hybrid import train_hybrid
from visualization.training_plot import plot_history


DATA_PATH = r"E:\dataset\PotatoGPT\minidata\potato"

train_ds, valid_ds, test_ds = load_and_split_datasets(DATA_PATH)

class_names = train_ds.class_names  # 用于打印分类报告

data_aug = get_data_augmentation()
train_ds = train_ds.map(lambda x, y: (data_aug(x), y))

# MODEL_TO_TRAIN = "vit"  
# MODEL_TO_TRAIN = "resnet"
MODEL_TO_TRAIN = "hybrid"

if MODEL_TO_TRAIN == "resnet":
    model, history = train_resnet(train_ds, valid_ds, num_classes=len(class_names))
    model_filename = "best_resnet_model.h5"
elif MODEL_TO_TRAIN == "vit":
    model, history = train_vit(train_ds, valid_ds, num_classes=len(class_names))
    model_filename = "best_vit_model.h5"
elif MODEL_TO_TRAIN == "hybrid":
    model, history = train_hybrid(train_ds, valid_ds, num_classes=len(class_names))
    model_filename = "best_hybrid_model.h5"
else:
    raise ValueError("MODEL_TO_TRAIN 必须是 'resnet', 'vit', 或 'hybrid'")

# plot_history(history, "accuracy")
# plot_history(history, "loss")

print(f"\n✅ Training completed. Model saved as {model_filename}")