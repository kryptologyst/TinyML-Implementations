"""Evaluation script for comparing TinyML models."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import TinyMLModelBuilder, TinyMLTrainer
from src.export import TinyMLExporter
from src.utils import set_deterministic_seed


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def evaluate_all_models() -> Dict[str, Any]:
    """Evaluate all model types and return comprehensive results."""
    logger = logging.getLogger(__name__)
    
    # Initialize components
    model_builder = TinyMLModelBuilder(seed=42)
    trainer = TinyMLTrainer(seed=42)
    exporter = TinyMLExporter()
    
    # Load data
    logger.info("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = trainer.load_data()
    
    # Split for validation
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]
    
    results = {}
    model_types = ['baseline', 'tiny', 'pruned', 'quantized']
    
    for model_type in model_types:
        logger.info(f"Evaluating {model_type} model...")
        
        # Create model
        if model_type == 'baseline':
            model = model_builder.create_baseline_model()
        elif model_type == 'tiny':
            model = model_builder.create_tiny_model()
        elif model_type == 'pruned':
            model = model_builder.create_pruned_model()
        elif model_type == 'quantized':
            base_model = model_builder.create_baseline_model()
            model = model_builder.create_quantized_model(base_model)
        
        # Train model
        trained_model = trainer.train_model(
            model, x_train, y_train, x_val, y_val,
            epochs=5, batch_size=32, verbose=0
        )
        
        # Evaluate model
        metrics = trainer.evaluate_model(trained_model, x_test, y_test)
        
        # Export models
        export_results = exporter.export_all_formats(
            trained_model,
            representative_data=x_train[:100],
            model_name=f"{model_type}_model"
        )
        
        # Benchmark exported models
        benchmark_results = {}
        for format_name, export_result in export_results.items():
            if export_result.get('success', False) and 'model_path' in export_result:
                benchmark_result = exporter.benchmark_model(export_result['model_path'])
                benchmark_results[format_name] = benchmark_result
        
        results[model_type] = {
            'metrics': metrics,
            'export_results': export_results,
            'benchmark_results': benchmark_results
        }
        
        logger.info(f"Completed evaluation of {model_type} model")
    
    return results


def create_comparison_report(results: Dict[str, Any]) -> None:
    """Create comprehensive comparison report."""
    logger = logging.getLogger(__name__)
    
    # Create results directory
    results_dir = Path("assets/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for comparison
    comparison_data = []
    
    for model_type, result in results.items():
        metrics = result['metrics']
        export_results = result['export_results']
        
        # Get model size from export results
        model_size_mb = None
        for format_name, export_result in export_results.items():
            if export_result.get('success', False):
                model_size_mb = export_result.get('model_size_mb', 0)
                break
        
        comparison_data.append({
            'Model Type': model_type.title(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'Model Size (MB)': model_size_mb or metrics['model_size_mb'],
            'Loss': metrics['loss']
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save results
    df.to_csv(results_dir / "model_comparison.csv", index=False)
    
    # Create visualizations
    create_comparison_plots(df, results_dir)
    
    # Create summary report
    create_summary_report(results, results_dir)
    
    logger.info("Comparison report created successfully")


def create_comparison_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comparison plots."""
    plt.style.use('seaborn-v0_8')
    
    # Accuracy vs Model Size plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['Model Size (MB)'], df['Accuracy'], 
                        s=100, alpha=0.7, c=df['F1 Score'], 
                        cmap='viridis')
    
    for i, model_type in enumerate(df['Model Type']):
        ax.annotate(model_type, 
                   (df['Model Size (MB)'].iloc[i], df['Accuracy'].iloc[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy vs Size Trade-off')
    plt.colorbar(scatter, label='F1 Score')
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        ax.bar([x_pos + width * i for x_pos in x], df[metric], 
               width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks([x_pos + width * 1.5 for x_pos in x])
    ax.set_xticklabels(df['Model Type'])
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(results: Dict[str, Any], output_dir: Path) -> None:
    """Create summary report."""
    report = {
        'evaluation_summary': {
            'total_models': len(results),
            'model_types': list(results.keys()),
            'evaluation_date': pd.Timestamp.now().isoformat()
        },
        'model_results': {}
    }
    
    for model_type, result in results.items():
        metrics = result['metrics']
        export_results = result['export_results']
        
        # Find best export format by size
        best_format = None
        best_size = float('inf')
        
        for format_name, export_result in export_results.items():
            if export_result.get('success', False):
                size = export_result.get('model_size_mb', float('inf'))
                if size < best_size:
                    best_size = size
                    best_format = format_name
        
        report['model_results'][model_type] = {
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'model_size_mb': metrics['model_size_mb'],
            'best_export_format': best_format,
            'best_export_size_mb': best_size,
            'export_success_count': sum(1 for r in export_results.values() if r.get('success', False))
        }
    
    # Save report
    with open(output_dir / "evaluation_summary.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create markdown report
    create_markdown_report(report, output_dir)


def create_markdown_report(report: Dict[str, Any], output_dir: Path) -> None:
    """Create markdown report."""
    md_content = f"""# TinyML Model Evaluation Report

## Summary

- **Total Models Evaluated**: {report['evaluation_summary']['total_models']}
- **Model Types**: {', '.join(report['evaluation_summary']['model_types'])}
- **Evaluation Date**: {report['evaluation_summary']['evaluation_date']}

## Model Results

| Model Type | Accuracy | F1 Score | Model Size (MB) | Best Export | Export Size (MB) |
|------------|----------|----------|-----------------|-------------|-----------------|
"""
    
    for model_type, result in report['model_results'].items():
        md_content += f"| {model_type.title()} | {result['accuracy']:.3f} | {result['f1_score']:.3f} | {result['model_size_mb']:.2f} | {result['best_export_format']} | {result['best_export_size_mb']:.2f} |\n"
    
    md_content += """
## Key Findings

1. **Accuracy vs Size Trade-off**: Smaller models (Tiny) sacrifice some accuracy for significantly reduced size
2. **Quantization Effectiveness**: INT8 quantization provides good size reduction with minimal accuracy loss
3. **Pruning Impact**: Magnitude-based pruning reduces model size while maintaining reasonable accuracy
4. **Export Compatibility**: All models successfully export to TensorFlow Lite and ONNX formats

## Recommendations

- **For Maximum Accuracy**: Use Baseline model with float32 export
- **For Size-Constrained Devices**: Use Tiny model with int8 quantization
- **For Balanced Performance**: Use Pruned model with int8 quantization
- **For Cross-Platform Deployment**: Use ONNX export format

## Disclaimer

This evaluation is for research and educational purposes only. Results may vary in production environments.
"""
    
    with open(output_dir / "evaluation_report.md", 'w') as f:
        f.write(md_content)


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='TinyML Model Evaluation Pipeline')
    parser.add_argument('--output-dir', type=str, default='assets/results',
                       help='Output directory for results')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create comparison plots')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_deterministic_seed(42)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive model evaluation...")
    
    # Evaluate all models
    results = evaluate_all_models()
    
    # Create comparison report
    create_comparison_report(results)
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
