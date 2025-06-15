# Example code to add to your Untitled.ipynb notebook
# Copy this into a new cell in your notebook

# Import the conversion functions
from tflite_converter import convert_keras_to_tflite, convert_keras_to_tflite_simple

# Method 1: Convert with quantization (recommended for deployment)
# This uses your validation dataset for quantization calibration
tflite_buffer_quantized = convert_keras_to_tflite(
    model=model,  # Your trained Keras model
    reference_dataset=v,  # Your validation dataset
    output_path="model_quantized.tflite"
)

# Method 2: Simple conversion without quantization (for testing)
tflite_buffer_simple = convert_keras_to_tflite_simple(
    model=model,  # Your trained Keras model
    output_path="model_simple.tflite"
)

# You can also convert without saving to file
# tflite_buffer = convert_keras_to_tflite(model, v)

print("Conversion complete!")
print(f"Quantized model size: {len(tflite_buffer_quantized) / 1024:.2f} KB")
print(f"Simple model size: {len(tflite_buffer_simple) / 1024:.2f} KB")

# Optional: Test the TFLite model
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_buffer_quantized)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nTFLite Model Details:")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input type: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output type: {output_details[0]['dtype']}") 