"""Modernized TinyML model implementations with comprehensive compression techniques."""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.datasets import mnist
import tensorflow_model_optimization as tfmot
from src.utils import set_deterministic_seed, get_model_size_mb


class TinyMLModelBuilder:
    """Builder class for creating TinyML-optimized models."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (28, 28, 1), 
                 num_classes: int = 10, seed: int = 42):
        """Initialize the model builder.
        
        Args:
            input_shape: Input image shape (height, width, channels).
            num_classes: Number of output classes.
            seed: Random seed for reproducibility.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.seed = seed
        set_deterministic_seed(seed)
    
    def create_baseline_model(self) -> tf.keras.Model:
        """Create a baseline CNN model for MNIST.
        
        Returns:
            Compiled TensorFlow model.
        """
        model = models.Sequential([
            layers.Conv2D(8, kernel_size=3, activation='relu', 
                          input_shape=self.input_shape, name='conv1'),
            layers.MaxPooling2D(pool_size=2, name='pool1'),
            layers.Conv2D(16, kernel_size=3, activation='relu', name='conv2'),
            layers.MaxPooling2D(pool_size=2, name='pool2'),
            layers.Flatten(name='flatten'),
            layers.Dense(32, activation='relu', name='dense1'),
            layers.Dropout(0.2, name='dropout'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='baseline_model')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_tiny_model(self) -> tf.keras.Model:
        """Create an ultra-small CNN suitable for microcontrollers.
        
        Returns:
            Compiled TensorFlow model optimized for TinyML.
        """
        model = models.Sequential([
            layers.Conv2D(4, kernel_size=3, activation='relu', 
                          input_shape=self.input_shape, name='conv1'),
            layers.MaxPooling2D(pool_size=2, name='pool1'),
            layers.Flatten(name='flatten'),
            layers.Dense(8, activation='relu', name='dense1'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='tiny_model')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_pruned_model(self, pruning_params: Optional[Dict[str, Any]] = None) -> tf.keras.Model:
        """Create a pruned model using magnitude-based pruning.
        
        Args:
            pruning_params: Pruning configuration parameters.
            
        Returns:
            Pruned TensorFlow model.
        """
        if pruning_params is None:
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=0.5,
                    begin_step=0,
                    end_step=1000
                )
            }
        
        base_model = self.create_baseline_model()
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
        
        pruned_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return pruned_model
    
    def create_quantized_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Create a quantized version of the model.
        
        Args:
            model: Base model to quantize.
            
        Returns:
            Quantized TensorFlow model.
        """
        quantized_model = tfmot.quantization.keras.quantize_model(model)
        
        quantized_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return quantized_model


class TinyMLTrainer:
    """Trainer class for TinyML models with comprehensive evaluation."""
    
    def __init__(self, seed: int = 42):
        """Initialize the trainer.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        set_deterministic_seed(seed)
        self.history: Dict[str, Any] = {}
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess MNIST dataset.
        
        Returns:
            Tuple of (x_train, y_train, x_test, y_test).
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Add channel dimension
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        
        return x_train, y_train, x_test, y_test
    
    def train_model(self, model: tf.keras.Model, 
                   x_train: np.ndarray, y_train: np.ndarray,
                   x_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 10, batch_size: int = 32,
                   verbose: int = 1) -> tf.keras.Model:
        """Train a model with callbacks and evaluation.
        
        Args:
            model: Model to train.
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            verbose: Verbosity level.
            
        Returns:
            Trained model.
        """
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        self.history[model.name] = history.history
        return model
    
    def evaluate_model(self, model: tf.keras.Model, 
                      x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            model: Model to evaluate.
            x_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Standard evaluation
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Predictions for additional metrics
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        
        # Model size
        model_size_mb = get_model_size_mb(model)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'model_size_mb': float(model_size_mb)
        }
