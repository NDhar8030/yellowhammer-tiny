# Fixed version for Jupyter notebook - no import needed
# Copy this entire cell into your Untitled.ipynb notebook

import tensorflow as tf
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory to Python path (fixes import issues)
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

def _check_int8_quantized(tflite_buf):
    """Utility: returns True if all tensors in the model are int8, else False and prints a summary."""
    interpreter = tf.lite.Interpreter(model_content=tflite_buf)
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    dtypes = [t["dtype"] for t in tensor_details]
    unique_dtypes = set(dtypes)
    print("Tensor dtypes in generated TFLite model:", unique_dtypes)
    all_int8 = unique_dtypes == {np.int8}
    if all_int8:
        print("✅ Model appears to be fully INT8-quantized.")
    else:
        print("⚠️  Model is NOT fully INT8-quantized (some tensors are not int8).")
    return all_int8

def convert_keras_to_tflite(model, reference_dataset, input_shape, output_path=None):
    """
    Convert a Keras model to TensorFlow Lite format with quantization and concrete input shape.
    
    Args:
        model: Keras model to convert
        reference_dataset: tf.data.Dataset for representative data (used for quantization)
        input_shape: Tuple specifying the input shape (batch_size, height, width, channels)
                    e.g., (1, 128, 128, 1) or (None, 128, 128, 1)
        output_path: Optional path to save the .tflite file
    
    Returns:
        bytes: TFLite model buffer
    """
    # Create a concrete function with fixed input shape
    print(f"Creating concrete function with input shape: {input_shape}")
    
    # Create input signature for concrete function
    input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32)]
    
    # Get concrete function
    concrete_func = model.__call__.get_concrete_function(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    )
    
    # Create TFLite converter from concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Set optimization flags for quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set input/output types to int8 for quantization
    converter.inference_input_type = tf.dtypes.int8
    converter.inference_output_type = tf.dtypes.int8
    
    # Specify supported operations for int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Disable per-channel quantization for dense layers (compatibility)
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    
    # Create representative dataset generator for quantization calibration
    def representative_dataset_gen():
        count = 0
        for example_spectrograms, example_labels in reference_dataset.take(10):
            for X, _ in zip(example_spectrograms, example_labels):
                # Reshape to match input_shape if needed
                if input_shape[0] is None or input_shape[0] == 1:
                    # Single batch
                    sample = tf.reshape(X, input_shape)
                    yield [sample[tf.newaxis, ...] if input_shape[0] is None else sample]
                else:
                    # Multiple batch size specified
                    sample = tf.reshape(X, input_shape[1:])  # Remove batch dimension
                    yield [sample[tf.newaxis, ...]]
                
                count += 1
                if count >= 50:  # Limit number of calibration samples
                    break
            if count >= 50:
                break
    
    # Set the representative dataset for quantization
    converter.representative_dataset = representative_dataset_gen
    
    # Convert the model
    print("Converting model to TensorFlow Lite format with concrete input shape...")
    tflite_model_buffer = converter.convert()
    
    # Verify quantization
    _check_int8_quantized(tflite_model_buffer)
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        with output_path.open("wb") as f:
            f.write(tflite_model_buffer)
        print(f"TFLite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model_buffer) / 1024:.2f} KB")
    
    return tflite_model_buffer

def convert_keras_to_tflite_simple(model, input_shape, output_path=None):
    """
    Simple conversion without quantization but with concrete input shape.
    
    Args:
        model: Keras model to convert
        input_shape: Tuple specifying the input shape (batch_size, height, width, channels)
                    e.g., (1, 128, 128, 1) or (None, 128, 128, 1)
        output_path: Optional path to save the .tflite file
    
    Returns:
        bytes: TFLite model buffer
    """
    # Create a concrete function with fixed input shape
    print(f"Creating concrete function with input shape: {input_shape}")
    
    # Get concrete function
    concrete_func = model.__call__.get_concrete_function(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    )
    
    # Create TFLite converter from concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Convert the model (no quantization)
    print("Converting model to TensorFlow Lite format (no quantization) with concrete input shape...")
    tflite_model_buffer = converter.convert()
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        with output_path.open("wb") as f:
            f.write(tflite_model_buffer)
        print(f"TFLite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model_buffer) / 1024:.2f} KB")
    
    return tflite_model_buffer

# Now use the functions directly (no import needed)
print("TFLite conversion functions loaded!")

# Example usage:
# Method 1: Convert with quantization and concrete input shape (recommended for deployment)
# input_shape = (1, 128, 128, 1)  # Example: batch_size=1, height=128, width=128, channels=1
# tflite_buffer_quantized = convert_keras_to_tflite(
#     model=model,  # Your trained Keras model
#     reference_dataset=v,  # Your validation dataset
#     input_shape=input_shape,  # Concrete input shape
#     output_path="model_quantized.tflite"
# )

# Method 2: Simple conversion without quantization but with concrete input shape
# tflite_buffer_simple = convert_keras_to_tflite_simple(
#     model=model,  # Your trained Keras model
#     input_shape=input_shape,  # Concrete input shape
#     output_path="model_simple.tflite"
# )

# Common input shapes for audio models:
# Spectrogram: (1, time_steps, frequency_bins, 1)
# Mel spectrogram: (1, time_frames, mel_channels, 1)
# Raw audio: (1, audio_length, 1) 