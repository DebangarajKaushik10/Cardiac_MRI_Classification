import os
import torch
from torch.utils.data import Dataset
import preprocess  # safer import

class ACDCDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        print("Loading dataset...")

        for patient in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient)

            if not os.path.isdir(patient_path):
                continue

            info_path = os.path.join(patient_path, "Info.cfg")

            # skip if label missing
            if not os.path.exists(info_path):
                print(f"Skipping (no label): {patient}")
                continue

            try:
                label_str = preprocess.get_label(info_path)
                label = preprocess.label_map[label_str]
            except Exception as e:
                print(f"Skipping label error in {patient}: {e}")
                continue

            for file in os.listdir(patient_path):
                if "frame" in file and "_gt" not in file:

                    file_path = os.path.join(patient_path, file)

                    # 🔥 skip corrupted files
                    try:
                        data = preprocess.load_nifti(file_path)
                    except Exception as e:
                        print(f"Skipping corrupted file: {file_path}")
                        continue

                    for i in range(data.shape[2]):
                        slice_ = data[:, :, i]

                        # skip empty slices
                        if slice_.sum() == 0:
                            continue

                        try:
                            slice_ = preprocess.preprocess_slice(slice_)
                        except:
                            continue

                        self.samples.append((slice_, label))

        print(f"Dataset loaded with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return img, label