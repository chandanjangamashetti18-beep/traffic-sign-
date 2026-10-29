# 🚦 Traffic Sign Recognition System

An AI-powered web application that classifies 43 classes of German Traffic Signs (GTSRB) in real-time using deep Convolutional Neural Networks (CNNs).

## Features
- 📁 **File Upload**: Classify custom traffic sign images (.jpg, .png, .jpeg).
- 📸 **Live Camera**: Capture and classify road signs on-the-fly using your webcam or phone camera.
- 🖼️ **Sample Gallery**: 43 preloaded real-world test images for 1-click evaluation.
- 🎯 **Confidence Scores**: Displays top prediction and candidate ranking with probabilities.

## Quickstart Locally
```bash
git clone <repo-url>
cd traffic_sign_app
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Compatible out-of-the-box with **Streamlit Community Cloud** and **Hugging Face Spaces**.
Entrypoint file: `app.py`.
