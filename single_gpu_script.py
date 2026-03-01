import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import ToyDataset, NeuralNetwork, create_datasets


def prepare_dataset():
    train_ds, test_ds = create_datasets()

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True, # Standard shuffle for single device
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


def main(num_epochs):
    # Determine device: Use GPU 0 if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(device) # Move model to device
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:

            features, labels = features.to(device), labels.to(device) # Move data to device
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    train_acc = compute_accuracy(model, train_loader, device=device)
    print(f"Training accuracy: {train_acc:.4f}")
    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f"Test accuracy: {test_acc:.4f}")


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    torch.manual_seed(123)
    num_epochs = 3
    main(num_epochs)