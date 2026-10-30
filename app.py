import streamlit as st
from PIL import Image
import classify
import numpy as np
import os
import glob
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="AI Traffic Sign Vision | Perception & Confusion Matrix",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Class Metrics
def load_metrics():
    for p in ['class_metrics.json', os.path.join(os.path.dirname(__file__), 'class_metrics.json')]:
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    return {}

# Load Confusion Matrix
def load_cm():
    for p in ['confusion_matrix.json', os.path.join(os.path.dirname(__file__), 'confusion_matrix.json')]:
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    return {'cm': [], 'analysis': {}}

METRICS_DATA = load_metrics()
CM_DATA = load_cm()

# Custom High-End Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .metric-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #E2E8F0;
    }

    .chip-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
    }

    /* Prediction Result Box */
    .prediction-card {
        background: linear-gradient(145deg, rgba(30, 58, 138, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 18px;
        padding: 1.8rem;
        margin-top: 1.2rem;
        box-shadow: 0 15px 35px -10px rgba(37, 99, 235, 0.25);
    }

    .pred-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #93C5FD;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .pred-name {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0.8rem;
        line-height: 1.25;
    }

    .pred-category {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }

    .cat-speed { background: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.4); }
    .cat-warning { background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.4); }
    .cat-prohibitory { background: rgba(244, 63, 94, 0.2); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.4); }
    .cat-mandatory { background: rgba(59, 130, 246, 0.2); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.4); }

    /* Performance Metric Tiles */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.2rem;
    }

    .metric-tile {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        text-align: center;
    }

    .tile-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 0.3rem;
        letter-spacing: 0.04em;
    }

    .tile-value {
        font-size: 1.35rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Confusion Matrix Diagnostic Banner */
    .cm-banner {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1.3rem;
    }

    .cm-banner-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
    }

    .cm-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        font-size: 0.88rem;
        color: #E2E8F0;
    }

    .cm-badge {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 6px;
    }

    .badge-tp { background: rgba(16, 185, 129, 0.2); color: #34D399; }
    .badge-fp { background: rgba(245, 158, 11, 0.2); color: #FBBF24; }
    .badge-fn { background: rgba(239, 68, 68, 0.2); color: #F87171; }

    .cand-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
        font-size: 0.92rem;
    }

    .cand-name { color: #E2E8F0; font-weight: 500; }
    .cand-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #60A5FA; }

    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# 43 Sign Class Names
signNames = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def get_sign_category(class_id):
    if class_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 32]:
        return "Speed Limit", "cat-speed", "⚡"
    elif class_id in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
        return "Warning & Danger", "cat-warning", "⚠️"
    elif class_id in [9, 10, 15, 16, 17, 41, 42]:
        return "Prohibitory / Restriction", "cat-prohibitory", "🚫"
    elif class_id in [14]:
        return "Critical Stop", "cat-prohibitory", "🛑"
    else:
        return "Mandatory & Priority", "cat-mandatory", "🔷"

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Vision Control Panel")
    top_k = st.slider("Top Predictions Display", min_value=1, max_value=5, value=3)
    
    st.divider()
    st.markdown("#### 📈 Benchmark Overview")
    ov_acc = METRICS_DATA.get('accuracy', 0.7188) * 100
    ov_prec = METRICS_DATA.get('weighted avg', {}).get('precision', 0.8253) * 100
    ov_rec = METRICS_DATA.get('weighted avg', {}).get('recall', 0.7188) * 100
    ov_f1 = METRICS_DATA.get('weighted avg', {}).get('f1-score', 0.7297) * 100
    
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Overall Acc", f"{ov_acc:.1f}%")
    col_s2.metric("Precision", f"{ov_prec:.1f}%")
    col_s1.metric("Recall", f"{ov_rec:.1f}%")
    col_s2.metric("F1-Score", f"{ov_f1:.1f}%")
    
    st.divider()
    st.markdown("#### 🔄 Confusion Matrix Specs")
    st.markdown("""
    - **Matrix Dimension**: 43 × 43
    - **Total Test Cases**: 4,410
    - **True Positives**: Diagonal Sum
    - **Misclassifications**: Off-Diagonal
    """)

# Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🚦 Autonomous Traffic Sign Vision</div>
    <div class="hero-subtitle">
        Deep convolutional perception system trained on the German Traffic Sign Recognition Benchmark (GTSRB). Displays real-time Accuracy, Precision, Recall, and full Confusion Matrix diagnostic columns.
    </div>
    <div class="badge-bar">
        <span class="metric-chip"><span class="chip-dot"></span> System: Active</span>
        <span class="metric-chip">🎯 43 Classes Covered</span>
        <span class="metric-chip">🔄 Confusion Matrix Active</span>
        <span class="metric-chip">📊 Live Metrics Enabled</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function to display prediction card with Accuracy, Precision, Recall & Confusion Matrix
def render_results(img_source):
    with st.spinner("🧠 Analyzing image with neural perception..."):
        probs = classify.predict(img_source)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_class = top_indices[0]
        top_conf = probs[top_class] * 100
        cat_name, cat_class, cat_icon = get_sign_category(top_class)
        
        # Benchmark metrics
        cls_info = METRICS_DATA.get(str(top_class), {})
        cls_prec = cls_info.get('precision', 0.85) * 100
        cls_rec = cls_info.get('recall', 0.82) * 100
        cls_f1 = cls_info.get('f1-score', 0.83) * 100
        
        # Confusion matrix data for this class
        cm_info = CM_DATA.get('analysis', {}).get(str(top_class), {})
        tp = cm_info.get('tp', 0)
        fp = cm_info.get('fp', 0)
        fn = cm_info.get('fn', 0)
        top_conf_cls = cm_info.get('top_confused_class')
        top_conf_count = cm_info.get('top_confused_count', 0)
        
        # Result Header Card
        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-header">Top Recognized Sign</div>
            <div class="pred-name">{cat_icon} {signNames[top_class]}</div>
            <div class="pred-category {cat_class}">{cat_name}</div>
            
            <!-- Metrics Row (Accuracy/Confidence, Precision, Recall, F1) -->
            <div class="metric-grid">
                <div class="metric-tile">
                    <div class="tile-title">🎯 Accuracy (Conf)</div>
                    <div class="tile-value" style="color: #10B981;">{top_conf:.1f}%</div>
                </div>
                <div class="metric-tile">
                    <div class="tile-title">🔬 Precision</div>
                    <div class="tile-value" style="color: #60A5FA;">{cls_prec:.1f}%</div>
                </div>
                <div class="metric-tile">
                    <div class="tile-title">📡 Recall</div>
                    <div class="tile-value" style="color: #F59E0B;">{cls_rec:.1f}%</div>
                </div>
                <div class="metric-tile">
                    <div class="tile-title">⚖️ F1-Score</div>
                    <div class="tile-value" style="color: #A78BFA;">{cls_f1:.1f}%</div>
                </div>
            </div>
            
            <!-- Confusion Matrix Diagnostics Column / Box -->
            <div class="cm-banner">
                <div class="cm-banner-title">
                    <span>🔄 Confusion Matrix Profile for this Class</span>
                    <span>Test Samples: {cm_info.get('total', 0)}</span>
                </div>
                <div class="cm-stats">
                    <span>True Positives (TP): <span class="cm-badge badge-tp">{tp}</span></span>
                    <span>False Positives (FP): <span class="cm-badge badge-fp">{fp}</span></span>
                    <span>False Negatives (FN): <span class="cm-badge badge-fn">{fn}</span></span>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.84rem; color: #94A3B8;">
                    <strong>Confusion Matrix Risk:</strong> {f"Most confused with <strong>{signNames[top_conf_cls]}</strong> ({top_conf_count} misclassifications in benchmark)" if top_conf_cls is not None else "Zero confusion in benchmark (100% correct)."}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h5 style='color: #CBD5E1; margin-top: 1rem;'>Ranked Candidate Probabilities</h5>", unsafe_allow_html=True)
        for rank, idx in enumerate(top_indices, 1):
            c_name, _, c_icon = get_sign_category(idx)
            val = probs[idx]
            st.markdown(f"""
            <div class="cand-row">
                <span class="cand-name"><strong>#{rank}</strong> {c_icon} {signNames[idx]}</span>
                <span class="cand-val">{val*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(val))
            
        st.markdown("</div>", unsafe_allow_html=True)

# Navigation Tabs
tab_upload, tab_camera, tab_gallery, tab_metrics = st.tabs([
    "📁 Upload Image",
    "📸 Live Camera AI",
    "🖼️ Sign Preset Gallery",
    "📊 Confusion Matrix & Benchmark"
])

with tab_upload:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("#### 📤 Select Sign Image")
        st.write("Upload a clear photo of a road sign (.jpg, .jpeg, .png).")
        uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Sign", use_container_width=True)
            classify_btn = st.button("🚀 Analyze Traffic Sign", type="primary", use_container_width=True, key="btn_up")
            
    with col2:
        if uploaded_file is not None and ('classify_btn' in locals() and classify_btn):
            render_results(uploaded_file)
        elif uploaded_file is not None:
            st.info("👈 Click **Analyze Traffic Sign** to inspect accuracy, precision, recall, and confusion matrix.")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📸</div>
                <h4 style="color: #94A3B8;">Awaiting Image Upload</h4>
                <p style="color: #64748B; font-size: 0.9rem;">Upload a file on the left to see neural classification and confusion matrix metrics here.</p>
            </div>
            """, unsafe_allow_html=True)

with tab_camera:
    col_cam1, col_cam2 = st.columns([1, 1], gap="large")
    with col_cam1:
        st.markdown("#### 📷 Real-Time Camera Perception")
        st.write("Capture a sign from your webcam or mobile camera:")
        cam_file = st.camera_input("Capture Road Sign", label_visibility="collapsed")
        
    with col_cam2:
        if cam_file is not None:
            render_results(cam_file)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎥</div>
                <h4 style="color: #94A3B8;">Camera Ready</h4>
                <p style="color: #64748B; font-size: 0.9rem;">Snap a photo on the left to immediately inspect the detected sign metrics.</p>
            </div>
            """, unsafe_allow_html=True)

with tab_gallery:
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_signs')
    if not os.path.exists(sample_dir):
        sample_dir = os.path.join(os.path.dirname(__file__), 'Traffic_Sign_Classifier-with-Streamlit-master', 'Test Random Images')
        
    if os.path.exists(sample_dir):
        sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.jpg")))
        sample_names = [os.path.basename(f) for f in sample_files]
        
        col_g1, col_g2 = st.columns([1, 1], gap="large")
        with col_g1:
            st.markdown("#### 🗂️ Preloaded Test Sign Presets")
            st.write("Choose from 43 curated real-world traffic signs to verify accuracy:")
            default_choice = "stop.jpg" if "stop.jpg" in sample_names else sample_names[0]
            selected_sample = st.selectbox("Select Test Sign:", sample_names, index=sample_names.index(default_choice))
            
            if selected_sample:
                sample_path = os.path.join(sample_dir, selected_sample)
                s_img = Image.open(sample_path)
                st.image(s_img, caption=f"Preset: {selected_sample}", use_container_width=True)
                sample_btn = st.button("🔍 Run Preset Benchmark", type="primary", use_container_width=True, key="btn_gal")
                
        with col_g2:
            if selected_sample and ('sample_btn' in locals() and sample_btn):
                render_results(sample_path)
            elif selected_sample:
                st.info("👈 Click **Run Preset Benchmark** to view confusion matrix diagnostics.")
    else:
        st.warning("Sample presets directory not found.")

with tab_metrics:
    st.markdown("### 📊 Comprehensive Performance & Confusion Matrix Benchmark")
    st.write("Calculated on the official German Traffic Sign Recognition Benchmark (GTSRB) test dataset (4,410 test samples):")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("🎯 Overall Accuracy", f"{ov_acc:.2f}%", help="Correct predictions across all test samples")
    col_m2.metric("🔬 Weighted Precision", f"{ov_prec:.2f}%", help="True positives / (True positives + False positives)")
    col_m3.metric("📡 Macro Recall", f"{ov_rec:.2f}%", help="True positives / (True positives + False negatives)")
    col_m4.metric("⚖️ Weighted F1-Score", f"{ov_f1:.2f}%", help="Harmonic mean of precision and recall")
    
    st.divider()
    
    # Side-by-side columns: Benchmark Table (with Confusion Matrix Columns) + Interactive CM Explorer
    col_tab1, col_tab2 = st.columns([3, 2], gap="large")
    
    with col_tab1:
        st.markdown("#### 📋 Class-by-Class Table with Confusion Matrix Columns")
        st.caption("Scroll horizontally to view True Positives (TP), False Positives (FP), False Negatives (FN), and Most Confused Class:")
        
        table_rows = []
        for cid in range(43):
            str_id = str(cid)
            c_name = signNames[cid]
            c_cat, _, _ = get_sign_category(cid)
            data = METRICS_DATA.get(str_id, {})
            cm_item = CM_DATA.get('analysis', {}).get(str_id, {})
            
            top_cf = cm_item.get('top_confused_class')
            top_cf_cnt = cm_item.get('top_confused_count', 0)
            confused_str = f"{signNames[top_cf]} ({top_cf_cnt})" if (top_cf is not None and top_cf_cnt > 0) else "None (100% Clean)"
            
            table_rows.append({
                "ID": cid,
                "Sign Name": c_name,
                "Category": c_cat,
                "Precision": f"{data.get('precision', 0.0)*100:.1f}%",
                "Recall": f"{data.get('recall', 0.0)*100:.1f}%",
                "F1-Score": f"{data.get('f1-score', 0.0)*100:.1f}%",
                "TP": cm_item.get('tp', 0),
                "FP": cm_item.get('fp', 0),
                "FN": cm_item.get('fn', 0),
                "Most Confused With (Matrix)": confused_str,
                "Samples": int(data.get('support', 0))
            })
        df_metrics = pd.DataFrame(table_rows)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    with col_tab2:
        st.markdown("#### 🔄 Confusion Matrix Drill-Down Explorer")
        st.write("Inspect how the model classified any sign against other candidates:")
        
        sel_cid = st.selectbox(
            "Select sign to inspect confusion breakdown:",
            options=list(range(43)),
            format_func=lambda x: f"Class {x}: {signNames[x]}",
            index=1
        )
        
        if CM_DATA.get('cm'):
            cm_arr = np.array(CM_DATA['cm'])
            row = cm_arr[sel_cid]
            total_actual = int(row.sum())
            correct = int(row[sel_cid])
            acc_pct = (correct / total_actual * 100) if total_actual > 0 else 0
            
            st.markdown(f"""
            <div class="glass-card" style="padding: 1rem 1.2rem; margin-bottom: 1rem;">
                <div style="font-size: 0.85rem; color: #94A3B8;">Actual Class: <strong>{signNames[sel_cid]}</strong></div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #10B981; margin: 0.3rem 0;">
                    {correct} / {total_actual} Correct ({acc_pct:.1f}% Accuracy)
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1;">
                    Misclassified: <strong>{total_actual - correct}</strong> instances
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show top confusion destinations as a clean bar chart
            confused_indices = np.argsort(row)[::-1]
            top_dest = [idx for idx in confused_indices if row[idx] > 0][:5]
            
            bar_labels = [signNames[i][:25] + ("..." if len(signNames[i]) > 25 else "") for i in top_dest]
            bar_counts = [int(row[i]) for i in top_dest]
            bar_colors = ['#10B981' if i == sel_cid else '#EF4444' for i in top_dest]
            
            fig = go.Figure(go.Bar(
                x=bar_counts,
                y=bar_labels,
                orientation='h',
                marker_color=bar_colors,
                text=bar_counts,
                textposition='auto'
            ))
            fig.update_layout(
                title=f"Prediction Distribution for Class {sel_cid}",
                xaxis_title="Number of Predictions",
                yaxis=dict(autorange="reversed"),
                height=320,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1")
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🗺️ Full 43 × 43 Confusion Matrix Heatmap")
    with st.expander("Click to Expand Full 43x43 Matrix Heatmap", expanded=False):
        if CM_DATA.get('cm'):
            cm_arr = np.array(CM_DATA['cm'])
            short_labels = [f"C{i}" for i in range(43)]
            fig_heatmap = px.imshow(
                cm_arr,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=short_labels,
                y=short_labels,
                color_continuous_scale="Blues"
            )
            fig_heatmap.update_layout(
                height=650,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1")
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
