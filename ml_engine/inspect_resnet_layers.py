
from tensorflow import keras
import tensorflow as tf

IMAGE_SIZE = 256

print("Loading ResNet50...")

backbone = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
)

layer_names = [
    "conv2_block3_out",
    "conv3_block4_out",
    "conv4_block6_out",
    "conv5_block3_out",
]

print("\nRESNET50 FEATURE LAYER ANALYSIS")
print("=" * 60)

for name in layer_names:
    layer = backbone.get_layer(name)

    print(f"{name}: {layer.output.shape}")

print("\nCreating multi-layer feature extractor...")

outputs = [
    backbone.get_layer(name).output
    for name in layer_names
]

diagnostic_model = keras.Model(
    inputs=backbone.input,
    outputs=outputs,
)

print("\nDiagnostic model created successfully.")
diagnostic_model.summary()
