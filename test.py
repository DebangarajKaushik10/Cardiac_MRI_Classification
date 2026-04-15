import torch
from sklearn.metrics import classification_report
from model import get_model
from dataset import ACDCDataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ACDCDataset("data")
loader = DataLoader(dataset, batch_size=8)

model = get_model().to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))

model.eval()

preds, actual = [], []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        preds.extend(predicted.cpu().numpy())
        actual.extend(labels.numpy())

print(classification_report(actual, preds))