import streamlit as st
from PIL import Image
import classify
import numpy as np
import os
import glob

# Page Configuration
st.set_page_config(
    page_title="AI Traffic Sign Vision | Autonomous Perception",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern CSS Styling
st.markdown("""
<style>
    /* Google Fonts */
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
        letter-spacing: 0.01em;
    }

    .chip-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
    }

    /* Modern Glass Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 1.2rem;
    }

    /* Prediction Result Box */
    .prediction-card {
        background: linear-gradient(145deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
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
        margin-bottom: 1rem;
    }

    .cat-speed { background: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.4); }
    .cat-warning { background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.4); }
    .cat-prohibitory { background: rgba(244, 63, 94, 0.2); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.4); }
    .cat-mandatory { background: rgba(59, 130, 246, 0.2); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.4); }

    /* Candidate Progress Item */
    .cand-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
        font-size: 0.92rem;
    }

    .cand-name {
        color: #E2E8F0;
        font-weight: 500;
    }

    .cand-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #60A5FA;
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
    st.write("Tune detection parameters and explore model architecture.")
    
    top_k = st.slider("Top Predictions Display", min_value=1, max_value=5, value=3)
    min_confidence = st.slider("Confidence Highlight Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.0f%%")
    
    st.divider()
    
    st.markdown("#### 🧠 Model Specifications")
    st.markdown("""
    - **Architecture**: Multi-Stage Deep CNN
    - **Dataset**: GTSRB (German Traffic Signs)
    - **Classes**: 43 Distinct Sign Categories
    - **Input Shape**: 32 × 32 × 1 (Grayscale)
    - **Feature Layers**: 3 Conv2D + MaxPool Blocks
    - **Dense Units**: 106 Hidden (tanh) $\\to$ 43 Output (softmax)
    """)
    
    st.divider()
    st.markdown("Developed with ❤️ using Streamlit & Deep Learning Vision.")

# Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🚦 Autonomous Traffic Sign Vision</div>
    <div class="hero-subtitle">
        Deep convolutional perception system trained on the German Traffic Sign Recognition Benchmark (GTSRB). Detect, classify, and verify road signs instantly using uploaded images, live cameras, or preloaded presets.
    </div>
    <div class="badge-bar">
        <span class="metric-chip"><span class="chip-dot"></span> Model Status: Active</span>
        <span class="metric-chip">🎯 43 Classes Covered</span>
        <span class="metric-chip">⚡ 32x32 Fast-Inference</span>
        <span class="metric-chip">🔒 Zero-Data Retention</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function to display modern prediction card
def render_results(img_source):
    with st.spinner("🧠 Analyzing image with neural perception..."):
        probs = classify.predict(img_source)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_class = top_indices[0]
        top_conf = probs[top_class] * 100
        cat_name, cat_class, cat_icon = get_sign_category(top_class)
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-header">Top Recognized Sign</div>
            <div class="pred-name">{cat_icon} {signNames[top_class]}</div>
            <div class="pred-category {cat_class}">{cat_name}</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #10B981; margin-bottom: 1.2rem;">
                Confidence: {top_conf:.1f}%
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
tab_upload, tab_camera, tab_gallery, tab_about = st.tabs([
    "📁 Upload Image",
    "📸 Live Camera AI",
    "🖼️ Sign Preset Gallery",
    "📊 Dataset & Architecture"
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
            st.info("👈 Click **Analyze Traffic Sign** to run inference.")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📸</div>
                <h4 style="color: #94A3B8;">Awaiting Image Upload</h4>
                <p style="color: #64748B; font-size: 0.9rem;">Upload a file on the left to see neural classification results here.</p>
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
                <p style="color: #64748B; font-size: 0.9rem;">Snap a photo on the left to immediately inspect the detected sign.</p>
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
                st.info("👈 Click **Run Preset Benchmark** to test this traffic sign.")
    else:
        st.warning("Sample presets directory not found.")

with tab_about:
    st.markdown("### 📊 Dataset & System Architecture")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        #### 🏛️ Neural Network Layout
        ```
        Input: (32 × 32 × 1 Grayscale)
          ├── Conv2D (35 filters, 3x3, ReLU, Same)
          │    └── MaxPool2D (2x2) + Dropout(0.05)
          ├── Conv2D (77 filters, 3x3, ReLU, Same)
          │    └── MaxPool2D (2x2) + Dropout(0.25)
          ├── Conv2D (68 filters, 5x5, ReLU, Same)
          │    └── MaxPool2D (2x2) + Dropout(0.20)
          ├── Flatten (1088 units)
          ├── Dense (106 units, Tanh) + Dropout(0.25)
          └── Dense Output (43 units, Softmax)
        ```
        """)
    with col_a2:
        st.markdown("""
        #### 📈 Dataset Information
        * **Origin**: German Traffic Sign Recognition Benchmark (GTSRB)
        * **Training Set**: 34,799 samples
        * **Validation Set**: 4,410 samples
        * **Test Set**: 12,630 samples
        * **Classes**: 43 distinct European traffic signs
        * **Inference Latency**: < 15ms (Fast edge deployment ready)
        """)
