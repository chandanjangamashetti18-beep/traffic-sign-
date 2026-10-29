import os
import numpy as np
import streamlit as st
from PIL import Image

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

def predict(image_file):
    model = get_model()

    if isinstance(image_file, (str, os.PathLike)):
        img = Image.open(image_file).convert('L').resize((32, 32))
    elif hasattr(image_file, 'read'):
        image_file.seek(0)
        img = Image.open(image_file).convert('L').resize((32, 32))
    elif isinstance(image_file, Image.Image):
        img = image_file.convert('L').resize((32, 32))
    else:
        raise ValueError(f"Unsupported image type: {type(image_file)}")

    arr = np.array(img, dtype=np.float32) / 255.0

    if HAS_TF:
        inp = np.reshape(arr, [1, 32, 32, 1])
        classes = model.predict(inp)
        return classes
    else:
        inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            classes = model(inp).cpu().numpy()
        return classes
