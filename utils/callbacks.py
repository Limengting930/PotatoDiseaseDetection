from tensorflow import keras

def get_callbacks(model_name="best_model"):
    """
    返回回调列表：
    - EarlyStopping: 提前停止
    - ModelCheckpoint: 保存验证集上最优权重
    注意：这里只保存权重，完整模型保存会在训练完成后手动保存
    """
    return [
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_accuracy"
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"{model_name}.weights.h5",  # 仅保存权重
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
