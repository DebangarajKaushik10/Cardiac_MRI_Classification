import torch
from torch.utils.data import DataLoader, random_split
from dataset import ACDCDataset
from model import get_model
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")

    dataset = ACDCDataset("data")

    print("Dataset loaded:", len(dataset))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    model = get_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10

    # 🔥 for graph
    loss_history = []

    print("Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # ✅ store loss
        loss_history.append(total_loss)

    # 💾 save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")

    # 📈 plot graph
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epochs")
    plt.savefig("loss.png")   # saves image
    plt.show()                # shows graph


if __name__ == "__main__":
    main()