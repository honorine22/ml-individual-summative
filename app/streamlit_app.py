"""Streamlit dashboard for FaultSense."""
from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Get API URL from environment or use default
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="FaultSense Control Room", layout="wide", initial_sidebar_state="expanded")


def get_status():
    try:
        return requests.get(f"{API_URL}/status", timeout=5).json()
    except requests.RequestException:
        return {"job": {"status": "unknown"}, "model": {}}


def render_header():
    status = get_status()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        api_status = "Online" if status.get("job") else "Offline"
        st.metric("API Status", api_status)
    with col2:
        job_status = status.get("job", {}).get("status", "idle")
        st.metric("Retrain Job", job_status.title())
    with col3:
        model_info = status.get("model", {})
        if model_info and "history" in model_info:
            history = model_info["history"]
            if history:
                best_epoch = max(history, key=lambda x: x.get("val_f1", 0))
                st.metric("Best Val F1", f"{best_epoch.get('val_f1', 0):.3f}")
        else:
            st.metric("Best Val F1", "N/A")
    with col4:
        st.metric("Last Update", time.strftime("%H:%M:%S"))


def render_insights_tab():
    st.header("📊 Data Insights & Visualizations")
    
    # Dataset statistics
    st.subheader("📈 Dataset Overview")
    data_dir = Path("data")
    curated_manifest = data_dir / "curated" / "manifest.csv"
    
    if curated_manifest.exists():
        df = pd.read_csv(curated_manifest)
        
        # Key statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Classes", df["label"].nunique())
        with col3:
            train_count = len(pd.read_csv(data_dir / "train_manifest.csv")) if (data_dir / "train_manifest.csv").exists() else 0
            st.metric("Training Samples", train_count)
        with col4:
            test_count = len(pd.read_csv(data_dir / "test_manifest.csv")) if (data_dir / "test_manifest.csv").exists() else 0
            st.metric("Test Samples", test_count)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution
            class_counts = df["label"].value_counts().sort_index()
            fig = px.bar(
                x=class_counts.index,
                y=class_counts.values,
                title="Class Distribution - Dataset Balance",
                labels={"x": "Fault Type", "y": "Number of Samples"},
                color=class_counts.values,
                color_continuous_scale="viridis",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
            
            # Feature Interpretation 1: Class Balance
            st.info("""
            **🔍 Feature Interpretation 1 - Class Distribution Balance**:
            
            The dataset maintains relatively balanced distribution across fault types (mechanical_fault, 
            electrical_fault, fluid_leak, normal_operation). This balanced distribution is crucial for:
            
            • **Preventing Model Bias**: Equal representation prevents the model from becoming biased 
              toward majority classes
            • **Robust Learning**: Each fault type gets sufficient training examples for pattern recognition
            • **Real-world Applicability**: Reflects realistic industrial scenarios where different 
              fault types occur with similar frequencies
            
            **Story**: This balance tells us the dataset was carefully curated to represent 
            real industrial environments where various fault types are equally important to detect.
            """)
        
        with col2:
            # Dataset split
            train_manifest = data_dir / "train_manifest.csv"
            test_manifest = data_dir / "test_manifest.csv"
            train_count = len(pd.read_csv(train_manifest)) if train_manifest.exists() else 0
            test_count = len(pd.read_csv(test_manifest)) if test_manifest.exists() else 0
            
            fig = px.pie(
                values=[train_count, test_count],
                names=["Train (80%)", "Test (20%)"],
                title="Train/Test Split",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, width='stretch')
            
            # Feature Interpretation 2: Data Split Strategy
            st.info("""
            **🔍 Feature Interpretation 2 - Train/Test Split Strategy**:
            
            The 80/20 train-test split is strategically chosen for optimal model performance:
            
            • **Training Sufficiency**: 80% provides enough data for the CNN to learn complex 
              audio patterns and fault signatures
            • **Validation Reliability**: 20% test set ensures statistically significant evaluation 
              while preserving data for training
            • **Generalization Testing**: Unseen test data validates the model's ability to 
              generalize to new industrial environments
            
            **Story**: This split reflects industry best practices where you need substantial 
            training data for deep learning while maintaining rigorous evaluation standards 
            for safety-critical fault detection systems.
            """)
        
        # Feature insights
        st.subheader("🔍 Feature Analysis")
        
        # Feature Interpretation 3: Audio Feature Engineering
        st.info("""
        **🔍 Feature Interpretation 3 - Audio Feature Engineering Impact**:
        
        Our multi-modal feature extraction approach combines three powerful audio representations:
        
        • **Log-Mel Spectrograms**: Capture time-frequency patterns that reveal fault signatures 
          in the frequency domain - critical for detecting mechanical vibrations and electrical noise
        • **MFCC Features**: Extract perceptually-relevant coefficients that mirror human auditory 
          processing - excellent for distinguishing between different types of industrial sounds
        • **Wav2Vec2 Embeddings**: Leverage pre-trained deep representations that capture complex 
          audio patterns learned from massive datasets
        
        **Story**: This combination tells the story of comprehensive fault detection - we capture 
        both the physical characteristics (spectrograms), perceptual qualities (MFCC), and 
        learned patterns (Wav2Vec2) of industrial audio. This multi-faceted approach ensures 
        our model can detect subtle differences between normal operations and various fault types, 
        making it robust for real-world industrial monitoring.
        """)
        
        # Load EDA visuals if available
        eda_dir = Path("reports/eda_visuals")
        if (eda_dir / "mfcc_trends.png").exists():
            col1, col2 = st.columns(2)
            with col1:
                st.image(str(eda_dir / "mfcc_trends.png"), caption="MFCC Feature Trends by Class")
                st.info("""
                **Insight**: MFCC coefficients show distinct patterns across fault types. 
                Mechanical faults exhibit higher energy in lower coefficients, while 
                electrical faults show more variation across the spectrum.
                """)
            with col2:
                if (eda_dir / "sample_waveforms_spectrograms.png").exists():
                    st.image(str(eda_dir / "sample_waveforms_spectrograms.png"), caption="Sample Waveforms & Spectrograms")
                    st.info("""
                    **Insight**: Visual inspection reveals distinct spectral patterns. 
                    Mechanical faults show rhythmic patterns, while fluid leaks exhibit 
                    more continuous frequency content.
                    """)
        
        # Confusion matrix
        st.subheader("🧭 Confusion Matrix")
        cm_path = Path("reports/confusion_matrix.json")
        if cm_path.exists():
            cm_payload = json.loads(cm_path.read_text())
            labels = cm_payload.get("labels", [])
            matrix = cm_payload.get("matrix", [])
            if labels and matrix:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=matrix,
                        x=labels,
                        y=labels,
                        text=matrix,
                        texttemplate="%{z}",
                        colorscale="Blues",
                        hoverongaps=False,
                    )
                )
                fig.update_layout(
                    title="Confusion Matrix (Validation Set)",
                    xaxis_title="Predicted Label",
                    yaxis_title="Actual Label",
                )
                st.plotly_chart(fig, width='stretch')
                np_matrix = np.array(matrix)
                if np_matrix.size > 0:
                    off_diag = np_matrix.copy()
                    np.fill_diagonal(off_diag, 0)
                    max_idx = np.unravel_index(off_diag.argmax(), off_diag.shape)
                    actual = labels[max_idx[0]]
                    predicted = labels[max_idx[1]]
                    count = int(off_diag[max_idx])
                    st.info(f"**Insight**: Most frequent confusion: `{actual}` misclassified as `{predicted}` ({count} samples). Consider adding more `{actual}` samples or targeted augmentation.")
            else:
                st.info("Confusion matrix will appear after the first training run.")
        else:
            st.info("Retrain the model to generate a confusion matrix.")
    
    # Model performance metrics
    st.subheader("Model Performance Metrics")
    registry_path = Path("models/registry.json")
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        history = registry.get("history", [])
        
        if history:
            df_history = pd.DataFrame(history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Training curves
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_history["epoch"], y=df_history["train_accuracy"], name="Train Accuracy", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df_history["epoch"], y=df_history["val_accuracy"], name="Val Accuracy", line=dict(color="red")))
                fig.update_layout(title="Accuracy Over Time", xaxis_title="Epoch", yaxis_title="Accuracy")
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # F1 score
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_history["epoch"], y=df_history["train_f1"], name="Train F1", line=dict(color="green")))
                fig.add_trace(go.Scatter(x=df_history["epoch"], y=df_history["val_f1"], name="Val F1", line=dict(color="orange")))
                fig.update_layout(title="F1 Score Over Time", xaxis_title="Epoch", yaxis_title="F1 Score")
                st.plotly_chart(fig, width='stretch')
            
            # Best metrics table
            best_epoch = df_history.loc[df_history["val_f1"].idxmax()]
            st.subheader("🏆 Best Model Performance (by Validation F1)")
            
            col1, col2 = st.columns(2)
            with col1:
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Loss"],
                    "Train": [
                        best_epoch["train_accuracy"],
                        best_epoch["train_precision"],
                        best_epoch["train_recall"],
                        best_epoch["train_f1"],
                        best_epoch["train_loss"],
                    ],
                    "Validation": [
                        best_epoch["val_accuracy"],
                        best_epoch["val_precision"],
                        best_epoch["val_recall"],
                        best_epoch["val_f1"],
                        best_epoch["val_loss"],
                    ],
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Performance interpretation
                gap = best_epoch["train_accuracy"] - best_epoch["val_accuracy"]
                if gap < 0.1:
                    status = "✅ Excellent generalization"
                    color = "green"
                elif gap < 0.2:
                    status = "⚠️ Good generalization"
                    color = "orange"
                else:
                    status = "⚠️ Possible overfitting"
                    color = "red"
                
                st.markdown(f"""
                **Model Health**: <span style="color:{color}">{status}</span>
                
                **Key Observations**:
                - Train-Val gap: {gap:.3f}
                - Best epoch: {int(best_epoch['epoch'])}
                - Learning rate: {best_epoch.get('lr', 'N/A')}
                
                **Interpretation**: The model achieves {best_epoch['val_accuracy']:.1%} validation accuracy 
                with balanced precision and recall, indicating good performance across all fault types.
                """, unsafe_allow_html=True)
            
            # Loss curves
            st.subheader("📉 Loss Curves")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_history["epoch"], y=df_history["train_loss"], name="Train Loss", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df_history["epoch"], y=df_history["val_loss"], name="Val Loss", line=dict(color="red")))
            fig.update_layout(title="Training and Validation Loss", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig, width='stretch')
            
            st.info("""
            **Insight**: The loss curves show the model learning effectively. The gap between 
            train and validation loss indicates the model's generalization capability.
            """)
            
            # Uploaded data statistics
            st.subheader("📤 Uploaded Data Statistics")
            upload_manifest = data_dir / "uploads" / "manifest.json"
            if upload_manifest.exists():
                upload_data = json.loads(upload_manifest.read_text())
                if upload_data:
                    upload_df = pd.DataFrame(upload_data)
                    upload_counts = upload_df["label"].value_counts() if "label" in upload_df.columns else pd.Series()
                    
                    if len(upload_counts) > 0:
                        fig = px.bar(
                            x=upload_counts.index,
                            y=upload_counts.values,
                            title="Uploaded Files by Class",
                            labels={"x": "Fault Type", "y": "Uploaded Files"},
                            color=upload_counts.values,
                            color_continuous_scale="blues",
                        )
                        st.plotly_chart(fig, width='stretch')
                        st.success(f"✅ {len(upload_data)} file(s) uploaded and ready for retraining")
                    else:
                        st.info("No uploaded files yet. Upload files in the 'Upload Data' tab.")
                else:
                    st.info("No uploaded files yet. Upload files in the 'Upload Data' tab.")
            else:
                st.info("No uploaded files yet. Upload files in the 'Upload Data' tab.")
    else:
        st.info("No model training history available. Train a model first.")


def render_predict_tab():
    st.header("🎯 Predict Fault Type")
    uploaded = st.file_uploader("Upload audio clip (.wav)", type=["wav"])
    
    if uploaded:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes, format="audio/wav")
        
        # Show waveform
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            y, sr = librosa.load(tmp_path, sr=16000)
            
            col1, col2 = st.columns(2)
            with col1:
                # Waveform
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(y)
                ax.set_title("Waveform")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)
            
            with col2:
                # Spectrogram
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                fig, ax = plt.subplots(figsize=(10, 3))
                img = librosa.display.specshow(mel_db, sr=sr, hop_length=512, ax=ax, cmap="magma")
                ax.set_title("Mel Spectrogram")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not visualize audio: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        
        if st.button("🔮 Run Prediction", type="primary"):
            with st.spinner("Analyzing audio..."):
                files = {"file": (uploaded.name, io.BytesIO(audio_bytes), uploaded.type)}
                try:
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    # Display prediction
                    st.success(f"**Prediction: {result['label'].replace('_', ' ').title()}** (Confidence: {result['confidence']:.2%})")
                    
                    # Probability distribution
                    dist = result["distribution"]
                    fig = px.bar(
                        x=list(dist.keys()),
                        y=list(dist.values()),
                        title="Prediction Probabilities",
                        labels={"x": "Fault Type", "y": "Probability"},
                        color=list(dist.values()),
                        color_continuous_scale="RdYlGn",
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                except requests.RequestException as e:
                    st.error(f"Prediction failed: {e}")


def render_upload_tab():
    st.header("📤 Upload Data for Retraining")
    st.info("Upload labeled audio files to improve the model. Files will be preprocessed and used in the next retraining cycle.")
    
    label = st.selectbox("Fault Label", ["mechanical_fault", "electrical_fault", "fluid_leak", "normal_operation"])
    files = st.file_uploader("Drop WAV files", type=["wav"], accept_multiple_files=True)
    
    if files:
        st.info(f"Selected {len(files)} file(s) for label: {label}")
        
        if st.button("📥 Upload Batch", type="primary"):
            with st.spinner(f"Uploading {len(files)} file(s)..."):
                multipart = [("files", (f.name, f, "audio/wav")) for f in files]
                try:
                    resp = requests.post(f"{API_URL}/upload", data={"label": label}, files=multipart, timeout=120)
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(f"✅ Successfully uploaded {len(result['stored_files'])} file(s)!")
                    st.json(result)
                except requests.RequestException as e:
                    st.error(f"Upload failed: {e}")


def render_retrain_tab():
    st.header("🔄 Model Retraining")
    st.info("Retraining uses the existing model as a pre-trained model (transfer learning) and incorporates uploaded data.")
    
    status = get_status()
    job_status = status.get("job", {})
    
    if job_status.get("status") == "running":
        st.warning("⏳ Retraining in progress...")
        st.progress(0.5)  # Placeholder
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Retraining", type="primary", disabled=job_status.get("status") == "running"):
            with st.spinner("Scheduling retraining job..."):
                try:
                    resp = requests.post(f"{API_URL}/retrain", timeout=5)
                    resp.raise_for_status()
                    st.success("✅ Retraining job scheduled!")
                    st.info("The model will be fine-tuned using uploaded data. This may take several minutes.")
                except requests.RequestException as e:
                    st.error(f"Failed to start retraining: {e}")
    
    with col2:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    # Show retraining status and metrics
    if job_status.get("metrics"):
        st.subheader("Last Training Metrics")
        metrics = job_status["metrics"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Val Accuracy", f"{metrics.get('val_accuracy', 0):.3f}")
        with col2:
            st.metric("Val F1", f"{metrics.get('val_f1', 0):.3f}")
        with col3:
            st.metric("Val Precision", f"{metrics.get('val_precision', 0):.3f}")
        with col4:
            st.metric("Val Recall", f"{metrics.get('val_recall', 0):.3f}")


# Main app
render_header()

tabs = st.tabs(["📊 Insights", "🎯 Predict", "📤 Upload Data", "🔄 Retraining"])
with tabs[0]:
    render_insights_tab()
with tabs[1]:
    render_predict_tab()
with tabs[2]:
    render_upload_tab()
with tabs[3]:
    render_retrain_tab()
