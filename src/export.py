"""Export and deployment pipeline for TinyML models."""

import os
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import tensorflow as tf
from pathlib import Path
import json


class TinyMLExporter:
    """Export TinyML models to various edge deployment formats."""
    
    def __init__(self, output_dir: str = "assets/models"):
        """Initialize the exporter.
        
        Args:
            output_dir: Directory to save exported models.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_tflite(self, model: tf.keras.Model, 
                      quantization_type: str = "int8",
                      representative_data: Optional[np.ndarray] = None,
                      model_name: str = "model") -> Dict[str, Any]:
        """Export model to TensorFlow Lite format.
        
        Args:
            model: Trained TensorFlow model.
            quantization_type: Type of quantization ('float32', 'int8', 'int16').
            representative_data: Representative dataset for quantization.
            model_name: Name for the exported model.
            
        Returns:
            Dictionary with export results and metrics.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Configure quantization
        if quantization_type == "int8":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            if representative_data is not None:
                def representative_data_gen():
                    for i in range(min(100, len(representative_data))):
                        yield [representative_data[i:i+1].astype(np.float32)]
                converter.representative_dataset = representative_data_gen
        
        elif quantization_type == "int16":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert model
        start_time = time.time()
        tflite_model = converter.convert()
        conversion_time = time.time() - start_time
        
        # Save model
        model_path = self.output_dir / f"{model_name}_{quantization_type}.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)
        
        # Calculate metrics
        model_size_mb = len(tflite_model) / (1024 * 1024)
        
        return {
            "format": "tflite",
            "quantization": quantization_type,
            "model_path": str(model_path),
            "model_size_mb": model_size_mb,
            "conversion_time_s": conversion_time,
            "success": True
        }
    
    def export_onnx(self, model: tf.keras.Model, 
                   model_name: str = "model") -> Dict[str, Any]:
        """Export model to ONNX format.
        
        Args:
            model: Trained TensorFlow model.
            model_name: Name for the exported model.
            
        Returns:
            Dictionary with export results and metrics.
        """
        try:
            import tf2onnx
            import onnx
            
            # Convert to ONNX
            start_time = time.time()
            spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            conversion_time = time.time() - start_time
            
            # Save model
            model_path = self.output_dir / f"{model_name}.onnx"
            with open(model_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            # Calculate metrics
            model_size_mb = len(onnx_model.SerializeToString()) / (1024 * 1024)
            
            return {
                "format": "onnx",
                "model_path": str(model_path),
                "model_size_mb": model_size_mb,
                "conversion_time_s": conversion_time,
                "success": True
            }
        except ImportError:
            return {
                "format": "onnx",
                "success": False,
                "error": "tf2onnx not installed"
            }
    
    def benchmark_model(self, model_path: str, 
                       input_shape: Tuple[int, ...] = (1, 28, 28, 1),
                       num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark model inference performance.
        
        Args:
            model_path: Path to the model file.
            input_shape: Input tensor shape.
            num_runs: Number of inference runs for benchmarking.
            
        Returns:
            Dictionary with benchmark results.
        """
        if model_path.endswith('.tflite'):
            return self._benchmark_tflite(model_path, input_shape, num_runs)
        elif model_path.endswith('.onnx'):
            return self._benchmark_onnx(model_path, input_shape, num_runs)
        else:
            return {"error": "Unsupported model format"}
    
    def _benchmark_tflite(self, model_path: str, 
                         input_shape: Tuple[int, ...], 
                         num_runs: int) -> Dict[str, Any]:
        """Benchmark TensorFlow Lite model."""
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Prepare input data
            input_data = np.random.random(input_shape).astype(np.float32)
            
            # Warmup runs
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
            
            # Benchmark runs
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                end_time = time.time()
                times.append(end_time - start_time)
            
            times = np.array(times)
            
            return {
                "format": "tflite",
                "model_path": model_path,
                "latency_ms_mean": float(np.mean(times) * 1000),
                "latency_ms_std": float(np.std(times) * 1000),
                "latency_ms_p50": float(np.percentile(times, 50) * 1000),
                "latency_ms_p95": float(np.percentile(times, 95) * 1000),
                "throughput_fps": float(1.0 / np.mean(times)),
                "success": True
            }
        except Exception as e:
            return {
                "format": "tflite",
                "model_path": model_path,
                "success": False,
                "error": str(e)
            }
    
    def _benchmark_onnx(self, model_path: str, 
                       input_shape: Tuple[int, ...], 
                       num_runs: int) -> Dict[str, Any]:
        """Benchmark ONNX model."""
        try:
            import onnxruntime as ort
            
            # Create inference session
            session = ort.InferenceSession(model_path)
            
            # Prepare input data
            input_name = session.get_inputs()[0].name
            input_data = np.random.random(input_shape).astype(np.float32)
            
            # Warmup runs
            for _ in range(10):
                session.run(None, {input_name: input_data})
            
            # Benchmark runs
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                session.run(None, {input_name: input_data})
                end_time = time.time()
                times.append(end_time - start_time)
            
            times = np.array(times)
            
            return {
                "format": "onnx",
                "model_path": model_path,
                "latency_ms_mean": float(np.mean(times) * 1000),
                "latency_ms_std": float(np.std(times) * 1000),
                "latency_ms_p50": float(np.percentile(times, 50) * 1000),
                "latency_ms_p95": float(np.percentile(times, 95) * 1000),
                "throughput_fps": float(1.0 / np.mean(times)),
                "success": True
            }
        except ImportError:
            return {
                "format": "onnx",
                "model_path": model_path,
                "success": False,
                "error": "onnxruntime not installed"
            }
        except Exception as e:
            return {
                "format": "onnx",
                "model_path": model_path,
                "success": False,
                "error": str(e)
            }
    
    def export_all_formats(self, model: tf.keras.Model, 
                          representative_data: Optional[np.ndarray] = None,
                          model_name: str = "model") -> Dict[str, Any]:
        """Export model to all supported formats.
        
        Args:
            model: Trained TensorFlow model.
            representative_data: Representative dataset for quantization.
            model_name: Name for the exported models.
            
        Returns:
            Dictionary with all export results.
        """
        results = {}
        
        # Export TensorFlow Lite formats
        for quant_type in ["float32", "int8"]:
            result = self.export_tflite(model, quant_type, representative_data, model_name)
            results[f"tflite_{quant_type}"] = result
        
        # Export ONNX
        onnx_result = self.export_onnx(model, model_name)
        results["onnx"] = onnx_result
        
        # Save export summary
        summary_path = self.output_dir / f"{model_name}_export_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
