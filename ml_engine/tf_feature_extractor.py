from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image


IMAGE_SIZE = 256


def build_feature_extractor():
    """
    Build a pretrained ResNet50 feature extractor.

    Extracts:
        conv3_block4_out: lower-level spatial features
        conv4_block6_out: higher-level semantic features

    The deeper feature map is resized to match the
    spatial dimensions of the shallower feature map.
    """

    backbone = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )

    backbone.trainable = False

    shallow_features = backbone.get_layer(
        "conv3_block4_out"
    ).output

    deep_features = backbone.get_layer(
        "conv4_block6_out"
    ).output

    deep_features = layers.Resizing(
        height=32,
        width=32,
        interpolation="bilinear",
        name="resize_deep_features",
    )(deep_features)

    combined_features = layers.Concatenate(
        axis=-1,
        name="combined_patch_features",
    )([
        shallow_features,
        deep_features,
    ])

    feature_extractor = keras.Model(
        inputs=backbone.input,
        outputs=combined_features,
        name="visionguard_resnet50_feature_extractor",
    )

    return feature_extractor


def load_image(image_path):
    """
    Load an image and prepare it for ResNet50.

    Returns:
        Tensor with shape (1, 256, 256, 3)
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return tf.convert_to_tensor(image_array)


def extract_features(feature_extractor, image_path):
    """
    Extract combined CNN features from one image.

    Returns:
        NumPy array with shape:
        (1, 32, 32, 1536)
    """

    image_tensor = load_image(image_path)

    features = feature_extractor(
        image_tensor,
        training=False,
    )

    return features.numpy()


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    sample_image = (
        project_root
        / "datasets"
        / "MVTecAD"
        / "bottle"
        / "train"
        / "good"
        / "000.png"
    )

    print("=" * 60)
    print("VISIONGUARD - TENSORFLOW FEATURE EXTRACTOR")
    print("=" * 60)

    print("\nBuilding ResNet50 feature extractor...")

    extractor = build_feature_extractor()

    print("\nModel summary:")
    extractor.summary()

    print("\nExtracting features from sample image...")

    features = extract_features(
        extractor,
        sample_image,
    )

    print("\nFeature extraction successful.")
    print("Feature shape:", features.shape)
    print("Feature dtype:", features.dtype)
    print("Feature minimum:", float(features.min()))
    print("Feature maximum:", float(features.max()))
    print("Feature mean:", float(features.mean()))

    print("\nExpected feature shape:")
    print("(1, 32, 32, 1536)")
