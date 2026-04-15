import os
import nibabel as nib
import numpy as np
import cv2

IMG_SIZE = 224

def load_nifti(path):
    return nib.load(path).get_fdata()

def preprocess_slice(slice_):
    slice_ = cv2.resize(slice_, (IMG_SIZE, IMG_SIZE))
    if np.max(slice_) != 0:
        slice_ = slice_ / np.max(slice_)
    return slice_

def get_label(info_path):
    with open(info_path) as f:
        for line in f:
            if "Group" in line:
                return line.split(":")[1].strip()

label_map = {
    "NOR": 0,
    "DCM": 1,
    "HCM": 2,
    "MINF": 3,
    "RV": 4
}