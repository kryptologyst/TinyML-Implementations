"""Interactive demo for TinyML implementations using Streamlit."""

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional

from src.models import TinyMLModelBuilder, TinyMLTrainer
from src.export import TinyMLExporter
from src.utils import set_deterministic_seed


# Page configuration
st.set_page_config(
    page_title="TinyML Implementations Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <p>This TinyML implementation is NOT intended for safety-critical applications, 
    production deployment, or any use where model accuracy and reliability are critical. 
    Use at your own risk.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔬 TinyML Implementations Demo</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")
model_type = st.sidebar.selectbox(
    "Model Type",
    ["baseline", "tiny", "pruned", "quantized"],
    help="Select the type of TinyML model to demonstrate"
)

export_format = st.sidebar.selectbox(
    "Export Format",
    ["tflite_float32", "tflite_int8", "onnx"],
    help="Select the deployment format"
)

benchmark_runs = st.sidebar.slider(
    "Benchmark Runs",
    min_value=10,
    max_value=1000,
    value=100,
    help="Number of inference runs for benchmarking"
)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = {}

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["Model Training", "Model Comparison", "Live Demo", "Performance Analysis"])

with tab1:
    st.header("Model Training Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=10)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Set deterministic seed
                set_deterministic_seed(42)
                
                # Initialize components
                model_builder = TinyMLModelBuilder(seed=42)
                trainer = TinyMLTrainer(seed=42)
                
                # Load data
                x_train, y_train, x_test, y_test = trainer.load_data()
                
                # Split for validation
                val_split = 0.1
                val_size = int(len(x_train) * val_split)
                x_val = x_train[:val_size]
                y_val = y_train[:val_size]
                x_train = x_train[val_size:]
                y_train = y_train[val_size:]
                
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
                    epochs=epochs, batch_size=batch_size, verbose=0
                )
                
                # Evaluate model
                metrics = trainer.evaluate_model(trained_model, x_test, y_test)
                
                # Store results
                st.session_state.model_loaded = True
                st.session_state.model_metrics = metrics
                st.session_state.trained_model = trained_model
                st.session_state.x_test = x_test
                st.session_state.y_test = y_test
                
                st.success("Model training completed!")
    
    with col2:
        st.subheader("Model Architecture")
        
        if st.session_state.model_loaded:
            model = st.session_state.trained_model
            st.text(f"Model: {model.name}")
            st.text(f"Parameters: {model.count_params():,}")
            st.text(f"Size: {st.session_state.model_metrics['model_size_mb']:.2f} MB")
            
            # Display model summary
            with st.expander("Model Summary"):
                from io import StringIO
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = buffer = StringIO()
                model.summary()
                sys.stdout = old_stdout
                
                st.text(buffer.getvalue())
        else:
            st.info("Train a model to see architecture details")

with tab2:
    st.header("Model Comparison Dashboard")
    
    if st.session_state.model_loaded:
        metrics = st.session_state.model_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
        with col3:
            st.metric("Model Size", f"{metrics['model_size_mb']:.2f} MB")
        with col4:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        
        # Performance comparison chart
        st.subheader("Performance Metrics")
        
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [metrics['accuracy'], metrics['precision'], 
                     metrics['recall'], metrics['f1_score']]
        }
        
        fig = px.bar(metrics_data, x='Metric', y='Value', 
                    title=f"{model_type.title()} Model Performance",
                    color='Value', color_continuous_scale='Viridis')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Model size comparison
        st.subheader("Model Size Comparison")
        size_data = {
            'Model Type': ['Baseline', 'Tiny', 'Pruned', 'Quantized'],
            'Size (MB)': [0.5, 0.1, 0.3, 0.2]  # Example sizes
        }
        
        fig_size = px.bar(size_data, x='Model Type', y='Size (MB)',
                         title="Model Size Comparison",
                         color='Size (MB)', color_continuous_scale='Blues')
        st.plotly_chart(fig_size, use_container_width=True)
    
    else:
        st.info("Please train a model first to see comparison metrics")

with tab3:
    st.header("Live Model Demo")
    
    if st.session_state.model_loaded:
        st.subheader("MNIST Digit Classification")
        
        # Create a simple drawing interface
        st.write("Draw a digit (0-9) in the canvas below:")
        
        # Simple drawing interface using matplotlib
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 28)
        ax.set_ylim(0, 28)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title("Draw a digit")
        
        # Create a simple random digit for demo
        if st.button("Generate Random Digit"):
            # Generate a random MNIST-like digit
            random_idx = np.random.randint(0, len(st.session_state.x_test))
            digit_image = st.session_state.x_test[random_idx]
            true_label = st.session_state.y_test[random_idx]
            
            # Display the image
            st.image(digit_image.squeeze(), caption=f"True Label: {true_label}", width=200)
            
            # Make prediction
            model = st.session_state.trained_model
            prediction = model.predict(digit_image.reshape(1, 28, 28, 1), verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.3f}")
            st.write(f"**Correct:** {'✅' if predicted_class == true_label else '❌'}")
            
            # Show prediction probabilities
            st.subheader("Prediction Probabilities")
            prob_data = {
                'Digit': list(range(10)),
                'Probability': prediction[0]
            }
            
            fig_prob = px.bar(prob_data, x='Digit', y='Probability',
                            title="Prediction Probabilities",
                            color='Probability', color_continuous_scale='RdYlBu')
            st.plotly_chart(fig_prob, use_container_width=True)
    
    else:
        st.info("Please train a model first to use the live demo")

with tab4:
    st.header("Performance Analysis")
    
    if st.session_state.model_loaded:
        st.subheader("Benchmark Results")
        
        if st.button("Run Benchmark"):
            with st.spinner("Running benchmark..."):
                exporter = TinyMLExporter()
                
                # Export model
                model = st.session_state.trained_model
                x_test = st.session_state.x_test
                
                export_result = exporter.export_tflite(
                    model, 
                    quantization_type=export_format.split('_')[1],
                    representative_data=x_test[:100],
                    model_name=f"{model_type}_demo"
                )
                
                if export_result['success']:
                    # Benchmark the exported model
                    benchmark_result = exporter.benchmark_model(
                        export_result['model_path'],
                        num_runs=benchmark_runs
                    )
                    
                    if benchmark_result['success']:
                        st.session_state.benchmark_results = benchmark_result
                        
                        # Display benchmark results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Mean Latency", f"{benchmark_result['latency_ms_mean']:.2f} ms")
                        with col2:
                            st.metric("P95 Latency", f"{benchmark_result['latency_ms_p95']:.2f} ms")
                        with col3:
                            st.metric("Throughput", f"{benchmark_result['throughput_fps']:.1f} FPS")
                        
                        # Latency distribution
                        st.subheader("Latency Distribution")
                        
                        # Simulate latency distribution for visualization
                        mean_latency = benchmark_result['latency_ms_mean']
                        std_latency = benchmark_result['latency_ms_std']
                        simulated_latencies = np.random.normal(mean_latency, std_latency, 1000)
                        
                        fig_latency = px.histogram(
                            x=simulated_latencies,
                            nbins=50,
                            title="Inference Latency Distribution",
                            labels={'x': 'Latency (ms)', 'y': 'Frequency'}
                        )
                        st.plotly_chart(fig_latency, use_container_width=True)
                        
                        # Performance comparison
                        st.subheader("Performance Comparison")
                        
                        performance_data = {
                            'Metric': ['Latency (ms)', 'Throughput (FPS)', 'Model Size (MB)'],
                            'Value': [
                                benchmark_result['latency_ms_mean'],
                                benchmark_result['throughput_fps'],
                                export_result['model_size_mb']
                            ]
                        }
                        
                        fig_perf = px.bar(performance_data, x='Metric', y='Value',
                                        title="Performance Metrics",
                                        color='Value', color_continuous_scale='Viridis')
                        st.plotly_chart(fig_perf, use_container_width=True)
                    else:
                        st.error(f"Benchmark failed: {benchmark_result.get('error', 'Unknown error')}")
                else:
                    st.error(f"Export failed: {export_result.get('error', 'Unknown error')}")
    
    else:
        st.info("Please train a model first to run performance analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>TinyML Implementations Demo - Research & Educational Use Only</p>
    <p>Built with Streamlit and TensorFlow</p>
</div>
""", unsafe_allow_html=True)
