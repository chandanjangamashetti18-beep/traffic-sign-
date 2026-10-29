import streamlit as st
from PIL import Image
import classify
import numpy as np
import os
import glob

st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦", layout="centered")

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

st.title("🚦 Traffic Sign Classifier")
st.markdown("Classify traffic signs in real-time using deep learning (CNN) trained on the German Traffic Sign Recognition Benchmark (GTSRB).")

def show_prediction_results(img_source):
    probs = classify.predict(img_source)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    top_class = top_indices[0]
    confidence = probs[top_class] * 100

    st.subheader("🎯 Prediction Result")
    st.success(f"**{signNames[top_class]}** ({confidence:.1f}% confidence)")

    st.write("##### Top 3 Candidates")
    for rank, idx in enumerate(top_indices, 1):
        col_name, col_prob = st.columns([3, 1])
        with col_name:
            st.write(f"**{rank}. {signNames[idx]}**")
        with col_prob:
            st.write(f"`{probs[idx]*100:.2f}%`")
        st.progress(float(probs[idx]))

tab_upload, tab_camera, tab_gallery = st.tabs(["📁 Upload Image", "📸 Live Camera", "🖼️ Sample Gallery"])

with tab_upload:
    uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "png", "jpeg"], key="uploader")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Sign", use_container_width=True)
        if st.button("Classify Uploaded Image", type="primary", key="btn_upload"):
            show_prediction_results(uploaded_file)

with tab_camera:
    st.info("Take a photo of a traffic sign using your device's camera:")
    camera_file = st.camera_input("Capture Traffic Sign", key="camera")
    if camera_file is not None:
        img = Image.open(camera_file)
        st.image(img, caption="Captured Image", use_container_width=True)
        show_prediction_results(camera_file)

with tab_gallery:
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_signs')
    if os.path.exists(sample_dir):
        sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.jpg")))
        sample_names = [os.path.basename(f) for f in sample_files]

        selected_sample_name = st.selectbox("Choose a test sign to classify:", sample_names, index=sample_names.index("stop.jpg") if "stop.jpg" in sample_names else 0)
        if selected_sample_name:
            selected_path = os.path.join(sample_dir, selected_sample_name)
            img = Image.open(selected_path)
            st.image(img, caption=f"Selected: {selected_sample_name}", use_container_width=True)
            if st.button("Classify Selected Sign", type="primary", key="btn_sample"):
                show_prediction_results(selected_path)
    else:
        st.warning("Sample directory not found.")
