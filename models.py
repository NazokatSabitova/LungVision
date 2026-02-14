import tensorflow as tf
from tensorflow.keras.applications import (
    DenseNet121,
    ResNet50,
    EfficientNetB0,
    MobileNetV2
)

def build_model(model_name, img_size=224, num_classes=14):

    if model_name == "densenet":
        base = DenseNet121(weights="imagenet", include_top=False,
                           input_shape=(img_size, img_size, 3))

    elif model_name == "resnet":
        base = ResNet50(weights="imagenet", include_top=False,
                        input_shape=(img_size, img_size, 3))

    elif model_name == "efficientnet":
        base = EfficientNetB0(weights="imagenet", include_top=False,
                              input_shape=(img_size, img_size, 3))

    elif model_name == "mobilenet":
        base = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(img_size, img_size, 3))

    else:
        raise ValueError("Unknown model")

    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation="sigmoid")
    ])

    return model
