from tensorflow import keras

def build_resnet50_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
