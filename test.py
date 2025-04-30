import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)