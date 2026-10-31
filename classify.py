import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import cv2

def _find_model_path():
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), 'Model', 'Traffic_Sign_Classifier_CNN.hdf5'),
        os.path.join(os.path.dirname(__file__), 'Traffic_Sign_Classifier-with-Streamlit-master', 'Model', 'Traffic_Sign_Classifier_CNN.hdf5'),
        'Model/Traffic_Sign_Classifier_CNN.hdf5',
        'Traffic_Sign_Classifier-with-Streamlit-master/Model/Traffic_Sign_Classifier_CNN.hdf5'
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find Traffic_Sign_Classifier_CNN.hdf5")

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    HAS_TF = True
except ImportError:
    HAS_TF = False
    import h5py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _TrafficSignCNN(nn.Module):
        def __init__(self, h5_path):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 35, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(35, 77, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(77, 68, kernel_size=5, padding=2)
            self.fc1 = nn.Linear(1088, 106)
            self.fc2 = nn.Linear(106, 43)

            with h5py.File(h5_path, 'r') as f:
                w = f['model_weights']
                # Conv weights: Keras (H, W, in_c, out_c) -> PyTorch (out_c, in_c, H, W)
                k1 = np.transpose(w['conv2d']['conv2d']['kernel:0'][:], (3, 2, 0, 1))
                b1 = w['conv2d']['conv2d']['bias:0'][:]
                self.conv1.weight.data = torch.from_numpy(k1).float()
                self.conv1.bias.data = torch.from_numpy(b1).float()

                k2 = np.transpose(w['conv2d_1']['conv2d_1']['kernel:0'][:], (3, 2, 0, 1))
                b2 = w['conv2d_1']['conv2d_1']['bias:0'][:]
                self.conv2.weight.data = torch.from_numpy(k2).float()
                self.conv2.bias.data = torch.from_numpy(b2).float()

                k3 = np.transpose(w['conv2d_2']['conv2d_2']['kernel:0'][:], (3, 2, 0, 1))
                b3 = w['conv2d_2']['conv2d_2']['bias:0'][:]
                self.conv3.weight.data = torch.from_numpy(k3).float()
                self.conv3.bias.data = torch.from_numpy(b3).float()

                k_fc1 = w['dense']['dense']['kernel:0'][:]
                b_fc1 = w['dense']['dense']['bias:0'][:]
                self.fc1.weight.data = torch.from_numpy(k_fc1.T).float()
                self.fc1.bias.data = torch.from_numpy(b_fc1).float()

                k_fc2 = w['dense_1']['dense_1']['kernel:0'][:]
                b_fc2 = w['dense_1']['bias:0'][:] if 'bias:0' in w['dense_1'] else w['dense_1']['dense_1']['bias:0'][:]
                self.fc2.weight.data = torch.from_numpy(k_fc2.T).float()
                self.fc2.bias.data = torch.from_numpy(b_fc2).float()

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
            x = torch.tanh(self.fc1(x))
            x = F.softmax(self.fc2(x), dim=-1)
            return x

@st.cache_resource
def get_model():
    model_path = _find_model_path()
    if HAS_TF:
        model = load_model(model_path)
    else:
        model = _TrafficSignCNN(model_path)
        model.eval()
    print('Model Loaded from:', model_path)
    return model

def preprocess_image(image_input, auto_crop=True):
    """
    Complete GTSRB neural preprocessing pipeline:
    1. EXIF orientation correction (phone cameras)
    2. RGBA transparency alpha composition (web PNGs)
    3. Optional Auto-ROI detection & square crop (scene photos)
    4. Resizing to (32, 32)
    5. Grayscale conversion + Histogram Equalization (cv2.equalizeHist)
    6. Normalization to [0, 1]
    Returns:
        normalized_32x32: (32, 32) float32 numpy array
        cropped_visual: PIL Image of the cropped/centered sign
    """
    # 1. Load Image into PIL
    if isinstance(image_input, (str, os.PathLike)):
        pil_img = Image.open(image_input)
    elif hasattr(image_input, 'read'):
        image_input.seek(0)
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input)
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    # 2. Fix EXIF orientation
    pil_img = ImageOps.exif_transpose(pil_img)

    # 3. Handle RGBA alpha transparency
    if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
        bg = Image.new('RGB', pil_img.size, (255, 255, 255))
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    else:
        pil_img = pil_img.convert('RGB')

    np_rgb = np.array(pil_img)
    h, w, _ = np_rgb.shape

    # 4. Auto-Crop ROI for images with large margins / scenes
    cropped_pil = pil_img
    if auto_crop and (w > 64 or h > 64):
        hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
        # Red signs (Stop, Speed limit, No Entry, Warnings, Prohibitory)
        mask_r1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([12, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([155, 50, 40]), np.array([180, 255, 255]))
        # Blue signs (Mandatory, Roundabout, Turn)
        mask_b = cv2.inRange(hsv, np.array([95, 50, 40]), np.array([135, 255, 255]))
        # Yellow signs (Priority road)
        mask_y = cv2.inRange(hsv, np.array([18, 60, 60]), np.array([38, 255, 255]))
        
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_y))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        max_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (h * w * 0.015):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh)
            if 0.55 <= aspect <= 1.8:
                cx = x + bw / 2
                cy = y + bh / 2
                dist_center = np.hypot(cx - w/2, cy - h/2) / np.hypot(w/2, h/2)
                score = area * (1.0 - 0.4 * dist_center)
                if score > max_score:
                    max_score = score
                    best_box = (x, y, bw, bh)

        if best_box is not None:
            x, y, bw, bh = best_box
            pad = int(max(bw, bh) * 0.08)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
            side = max(x2 - x1, y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            sx1, sy1 = max(0, cx - side // 2), max(0, cy - side // 2)
            sx2, sy2 = min(w, sx1 + side), min(h, sy1 + side)
            cropped_pil = pil_img.crop((sx1, sy1, sx2, sy2))
        else:
            # Aspect-ratio preserving square center-crop
            min_dim = min(h, w)
            cx, cy = w // 2, h // 2
            cropped_pil = pil_img.crop((cx - min_dim // 2, cy - min_dim // 2, cx + min_dim // 2, cy + min_dim // 2))

    # 5. Resize to 32x32
    np_crop = np.array(cropped_pil)
    resized_32 = cv2.resize(np_crop, (32, 32), interpolation=cv2.INTER_AREA)

    # 6. Grayscale + Histogram Equalization (GTSRB training requirement)
    gray = cv2.cvtColor(resized_32, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    normalized = equalized.astype(np.float32) / 255.0

    return normalized, cropped_pil

def predict(image_file, auto_crop=True):
    model = get_model()
    arr, _ = preprocess_image(image_file, auto_crop=auto_crop)

    if HAS_TF:
        inp = np.reshape(arr, [1, 32, 32, 1])
        classes = model.predict(inp)
        return classes
    else:
        inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            classes = model(inp).cpu().numpy()
        return classes

def generate_gradcam(image_file, auto_crop=True):
    """Generates Grad-CAM visual explanation overlay and heatmap for input image."""
    model = get_model()
    arr, orig_img = preprocess_image(image_file, auto_crop=auto_crop)

    inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    inp.requires_grad = True

    activations, gradients = [], []
    def f_hook(m, i, o): activations.append(o)
    def b_hook(m, gi, go): gradients.append(go[0])

    hf = model.conv3.register_forward_hook(f_hook)
    hb = model.conv3.register_full_backward_hook(b_hook)

    out = model(inp)
    pred_idx = out.argmax().item()
    confidence = float(out[0, pred_idx].item())
    out[0, pred_idx].backward()

    hf.remove()
    hb.remove()

    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, orig_img.size)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros_like(cam)

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * np.array(orig_img) + 0.45 * heatmap_rgb)

    return Image.fromarray(overlay), Image.fromarray(heatmap_rgb), pred_idx, confidence
