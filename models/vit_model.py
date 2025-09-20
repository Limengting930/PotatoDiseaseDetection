# 修改 models/vit_model.py 文件
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        # 使用 tf.image.extract_patches 提取 patches
        batch_size = tf.shape(images)[0]
        patch_height = self.patch_size
        patch_width = self.patch_size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_height, patch_width, 1],
            strides=[1, patch_height, patch_width, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # 计算 patch 数量
        num_patches = (images.shape[1] // self.patch_size) * (images.shape[2] // self.patch_size)
        patch_dim = patch_height * patch_width * images.shape[3]
        # reshape 为 [batch, num_patches, patch_dim]
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dim])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size
        })
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=tf.shape(patches)[1], delta=1)
        positions = tf.expand_dims(positions, axis=0)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config

def build_vit_model(input_shape=(224, 224, 3), num_classes=5, patch_size=16, projection_dim=64, transformer_layers=8):
    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim, activation="gelu")(x3)
        encoded_patches = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)