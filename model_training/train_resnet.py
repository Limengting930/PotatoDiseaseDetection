from tensorflow import keras
from models.resnet_model import build_resnet50_model
from utils.callbacks import get_callbacks

def train_resnet(train_ds, valid_ds, num_classes):
    model = build_resnet50_model(num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = get_callbacks(model_name="best_resnet")
    history = model.fit(
        train_ds,
        epochs=20,
        validation_data=valid_ds,
        callbacks=callbacks
    )

    # 加载最佳权重
    model.load_weights("best_resnet_weights.h5")

    # 保存完整模型 (结构 + 权重)
    model.save("best_resnet_model.h5")

    return model, history
