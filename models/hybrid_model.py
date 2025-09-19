from tensorflow import keras
#from tensorflow.keras import layers
from .vit_model import build_vit_model

def build_hybrid_model(input_shape=(224, 224, 3), num_classes=3):
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False
    feature_extractor = keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    resnet_input = keras.Input(shape=input_shape)
    resnet_features = feature_extractor(resnet_input)
    reshaped = keras.layers.Reshape((7, 7, 2048))(resnet_features)
    vit_head = build_vit_model(input_shape=(7, 7, 2048), num_classes=num_classes)
    outputs = vit_head(reshaped)

    return keras.Model(inputs=resnet_input, outputs=outputs)
