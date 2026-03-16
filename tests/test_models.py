"""Test suite for TinyML implementations."""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

from src.models import TinyMLModelBuilder, TinyMLTrainer
from src.export import TinyMLExporter
from src.utils import set_deterministic_seed, get_device, get_model_size_mb


class TestTinyMLModelBuilder:
    """Test cases for TinyMLModelBuilder."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = TinyMLModelBuilder(seed=42)
    
    def test_builder_initialization(self):
        """Test model builder initialization."""
        assert self.builder.input_shape == (28, 28, 1)
        assert self.builder.num_classes == 10
        assert self.builder.seed == 42
    
    def test_create_baseline_model(self):
        """Test baseline model creation."""
        model = self.builder.create_baseline_model()
        
        assert isinstance(model, tf.keras.Model)
        assert model.name == 'baseline_model'
        assert model.input_shape == (None, 28, 28, 1)
        assert model.output_shape == (None, 10)
    
    def test_create_tiny_model(self):
        """Test tiny model creation."""
        model = self.builder.create_tiny_model()
        
        assert isinstance(model, tf.keras.Model)
        assert model.name == 'tiny_model'
        assert model.input_shape == (None, 28, 28, 1)
        assert model.output_shape == (None, 10)
    
    def test_model_compilation(self):
        """Test model compilation."""
        model = self.builder.create_baseline_model()
        
        assert model.optimizer is not None
        assert model.loss is not None
        assert model.metrics is not None


class TestTinyMLTrainer:
    """Test cases for TinyMLTrainer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.trainer = TinyMLTrainer(seed=42)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.seed == 42
        assert isinstance(self.trainer.history, dict)
    
    def test_load_data(self):
        """Test data loading."""
        x_train, y_train, x_test, y_test = self.trainer.load_data()
        
        assert x_train.shape == (60000, 28, 28, 1)
        assert y_train.shape == (60000,)
        assert x_test.shape == (10000, 28, 28, 1)
        assert y_test.shape == (10000,)
        
        # Check data normalization
        assert np.all(x_train >= 0.0) and np.all(x_train <= 1.0)
        assert np.all(x_test >= 0.0) and np.all(x_test <= 1.0)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create a simple model for testing
        builder = TinyMLModelBuilder(seed=42)
        model = builder.create_tiny_model()
        
        # Load test data
        _, _, x_test, y_test = self.trainer.load_data()
        
        # Use small subset for testing
        x_test_small = x_test[:100]
        y_test_small = y_test[:100]
        
        # Evaluate model
        metrics = self.trainer.evaluate_model(model, x_test_small, y_test_small)
        
        assert 'accuracy' in metrics
        assert 'loss' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'model_size_mb' in metrics
        
        # Check metric ranges
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
        assert metrics['model_size_mb'] > 0.0


class TestTinyMLExporter:
    """Test cases for TinyMLExporter."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.exporter = TinyMLExporter(output_dir="test_assets/models")
    
    def test_exporter_initialization(self):
        """Test exporter initialization."""
        assert self.exporter.output_dir.name == "models"
        assert self.exporter.output_dir.parent.name == "test_assets"
    
    def test_export_tflite_float32(self):
        """Test TensorFlow Lite float32 export."""
        # Create a simple model
        builder = TinyMLModelBuilder(seed=42)
        model = builder.create_tiny_model()
        
        # Export model
        result = self.exporter.export_tflite(
            model, 
            quantization_type="float32",
            model_name="test_model"
        )
        
        assert result['success'] is True
        assert result['format'] == 'tflite'
        assert result['quantization'] == 'float32'
        assert result['model_size_mb'] > 0.0
        assert result['conversion_time_s'] > 0.0
    
    def test_export_tflite_int8(self):
        """Test TensorFlow Lite int8 export."""
        # Create a simple model
        builder = TinyMLModelBuilder(seed=42)
        model = builder.create_tiny_model()
        
        # Create representative data
        representative_data = np.random.random((10, 28, 28, 1)).astype(np.float32)
        
        # Export model
        result = self.exporter.export_tflite(
            model, 
            quantization_type="int8",
            representative_data=representative_data,
            model_name="test_model_int8"
        )
        
        assert result['success'] is True
        assert result['format'] == 'tflite'
        assert result['quantization'] == 'int8'
        assert result['model_size_mb'] > 0.0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_deterministic_seed(self):
        """Test deterministic seed setting."""
        # This is a basic test - in practice, you'd test actual randomness
        set_deterministic_seed(42)
        assert True  # If no exception is raised, test passes
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in ['cpu', 'cuda', 'mps']
        
        # Test specific device requests
        cpu_device = get_device('cpu')
        assert cpu_device == 'cpu'
    
    def test_get_model_size_mb(self):
        """Test model size calculation."""
        builder = TinyMLModelBuilder(seed=42)
        model = builder.create_tiny_model()
        
        size_mb = get_model_size_mb(model)
        assert size_mb > 0.0
        assert isinstance(size_mb, float)


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test complete training and export pipeline."""
        # Initialize components
        builder = TinyMLModelBuilder(seed=42)
        trainer = TinyMLTrainer(seed=42)
        exporter = TinyMLExporter(output_dir="test_assets/models")
        
        # Load data
        x_train, y_train, x_test, y_test = trainer.load_data()
        
        # Use small subset for testing
        x_train_small = x_train[:1000]
        y_train_small = y_train[:1000]
        x_test_small = x_test[:100]
        y_test_small = y_test[:100]
        
        # Create and train model
        model = builder.create_tiny_model()
        trained_model = trainer.train_model(
            model, x_train_small, y_train_small, 
            x_test_small, y_test_small,
            epochs=1, batch_size=32, verbose=0
        )
        
        # Evaluate model
        metrics = trainer.evaluate_model(trained_model, x_test_small, y_test_small)
        assert metrics['accuracy'] >= 0.0
        
        # Export model
        export_result = exporter.export_tflite(
            trained_model,
            quantization_type="float32",
            model_name="integration_test"
        )
        
        assert export_result['success'] is True


# Pytest configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Set deterministic seed for tests
    set_deterministic_seed(42)
    
    # Configure TensorFlow for testing
    tf.config.experimental.enable_op_determinism()
    
    yield
    
    # Cleanup after tests
    import shutil
    import os
    if os.path.exists("test_assets"):
        shutil.rmtree("test_assets")
