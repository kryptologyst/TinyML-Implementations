"""Main training script for TinyML implementations."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import tensorflow as tf

from src.models import TinyMLModelBuilder, TinyMLTrainer
from src.export import TinyMLExporter
from src.utils import set_deterministic_seed, configure_tensorflow_memory_growth


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='TinyML Model Training Pipeline')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-type', type=str, default='baseline',
                       choices=['baseline', 'tiny', 'pruned', 'quantized'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--export', action='store_true',
                       help='Export models after training')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark exported models')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_deterministic_seed(args.seed)
    configure_tensorflow_memory_growth()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting TinyML training with model type: {args.model_type}")
    
    # Initialize components
    model_builder = TinyMLModelBuilder(seed=args.seed)
    trainer = TinyMLTrainer(seed=args.seed)
    exporter = TinyMLExporter()
    
    # Load data
    logger.info("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = trainer.load_data()
    
    # Split training data for validation
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]
    
    logger.info(f"Training samples: {len(x_train)}")
    logger.info(f"Validation samples: {len(x_val)}")
    logger.info(f"Test samples: {len(x_test)}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    if args.model_type == 'baseline':
        model = model_builder.create_baseline_model()
    elif args.model_type == 'tiny':
        model = model_builder.create_tiny_model()
    elif args.model_type == 'pruned':
        model = model_builder.create_pruned_model()
    elif args.model_type == 'quantized':
        base_model = model_builder.create_baseline_model()
        model = model_builder.create_quantized_model(base_model)
    
    # Train model
    logger.info(f"Training model for {args.epochs} epochs...")
    trained_model = trainer.train_model(
        model, x_train, y_train, x_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = trainer.evaluate_model(trained_model, x_test, y_test)
    
    logger.info("Model Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save model
    model_path = f"assets/models/{args.model_type}_model.h5"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    trained_model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Export models if requested
    if args.export:
        logger.info("Exporting models...")
        export_results = exporter.export_all_formats(
            trained_model, 
            representative_data=x_train[:100],
            model_name=f"{args.model_type}_model"
        )
        
        logger.info("Export Results:")
        for format_name, result in export_results.items():
            if result.get('success', False):
                logger.info(f"  {format_name}: {result['model_size_mb']:.2f} MB")
            else:
                logger.info(f"  {format_name}: Failed - {result.get('error', 'Unknown error')}")
        
        # Benchmark models if requested
        if args.benchmark:
            logger.info("Benchmarking exported models...")
            for format_name, result in export_results.items():
                if result.get('success', False) and 'model_path' in result:
                    benchmark_result = exporter.benchmark_model(result['model_path'])
                    if benchmark_result.get('success', False):
                        logger.info(f"  {format_name} Benchmark:")
                        logger.info(f"    Latency (mean): {benchmark_result['latency_ms_mean']:.2f} ms")
                        logger.info(f"    Throughput: {benchmark_result['throughput_fps']:.2f} FPS")
    
    # Save results
    results = {
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'metrics': metrics,
        'export_results': export_results if args.export else None
    }
    
    results_path = f"assets/results/{args.model_type}_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
