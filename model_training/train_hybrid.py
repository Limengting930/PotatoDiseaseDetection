from tensorflow import keras
from models.hybrid_model import build_hybrid_model
from utils.callbacks import get_callbacks

def train_hybrid(train_ds, valid_ds, num_classes):
    model = build_hybrid_model(num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = get_callbacks(model_name="best_hybrid")
    history = model.fit(
        train_ds,
        epochs=30,
        validation_data=valid_ds,
        callbacks=callbacks
    )

    # 加载最佳权重
    model.load_weights("best_hybrid_weights.h5")

    # 保存完整模型
    model.save("best_hybrid_model.h5")

    return model, history
