# TensorFlow Lite Model Conversion
# Based on BioDCASE-Tiny-2025 implementation

import tensorflow as tf
import numpy as np
from pathlib import Path

def convert_keras_to_tflite(model, reference_dataset, output_path=None):
    """
    Convert a Keras model to TensorFlow Lite format with quantization.
    
    Args:
        model: Keras model to convert
        reference_dataset: tf.data.Dataset for representative data (used for quantization)
        output_path: Optional path to save the .tflite file
    
    Returns:
        bytes: TFLite model buffer
    """
    # Create TFLite converter from Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
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
        for example_spectrograms, example_labels in reference_dataset.take(10):
            for X, _ in zip(example_spectrograms, example_labels):
                # Add batch dimension for single example
                yield [X[tf.newaxis, ...]]
    
    # Set the representative dataset for quantization
    converter.representative_dataset = representative_dataset_gen
    
    # Convert the model
    print("Converting model to TensorFlow Lite format...")
    tflite_model_buffer = converter.convert()
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        with output_path.open("wb") as f:
            f.write(tflite_model_buffer)
        print(f"TFLite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model_buffer) / 1024:.2f} KB")
    
    return tflite_model_buffer

def convert_keras_to_tflite_simple(model, output_path=None):
    """
    Simple conversion without quantization (for testing purposes).
    
    Args:
        model: Keras model to convert
        output_path: Optional path to save the .tflite file
    
    Returns:
        bytes: TFLite model buffer
    """
    # Create TFLite converter from Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Convert the model (no quantization)
    print("Converting model to TensorFlow Lite format (no quantization)...")
    tflite_model_buffer = converter.convert()
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        with output_path.open("wb") as f:
            f.write(tflite_model_buffer)
        print(f"TFLite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model_buffer) / 1024:.2f} KB")
    
    return tflite_model_buffer

# Example usage:
"""
# With quantization (recommended for embedded deployment)
tflite_buffer = convert_keras_to_tflite(model, validation_dataset, "model_quantized.tflite")

# Without quantization (for testing)
tflite_buffer_simple = convert_keras_to_tflite_simple(model, "model_simple.tflite")
"""

if __name__ == "__main__":
    print("TFLite conversion utilities loaded!")
    print("Use convert_keras_to_tflite() for quantized models")
    print("Use convert_keras_to_tflite_simple() for non-quantized models") 