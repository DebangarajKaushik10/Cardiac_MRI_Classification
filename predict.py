import torch
import numpy as np
import nibabel as nib
from model import get_model
from preprocess import preprocess_slice
print("PREDICT FILE STARTED")

# Class labels
classes = ["NOR", "DCM", "HCM", "MINF", "RV"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 Load trained model
model = get_model().to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

# 🔹 Load real MRI file
path = "data/patient001/patient001_frame01.nii.gz"   # you can change patient

data = nib.load(path).get_fdata()

# 🔹 Take middle slice
slice_index = data.shape[2] // 2
sample = data[:, :, slice_index]

# 🔹 Preprocess
sample = preprocess_slice(sample)

# 🔹 Convert to tensor
sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

# 🔹 Predict
with torch.no_grad():
    output = model(sample)
    pred = torch.argmax(output, 1).item()

# 🔹 Print result
print("Predicted Class:", classes[pred])