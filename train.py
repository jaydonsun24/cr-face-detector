import torch
import torch.nn as nn
from data_setup import train_dataloader, test_dataloader, device, class_names, train_data
from model import model
from tqdm import tqdm
from collections import Counter

weighted = []
for i in range(len(class_names)):
    weighted.append(len(train_data.targets) / (len(class_names) * Counter(train_data.targets)[i]))

weighted = torch.tensor(weighted, dtype = torch.float32).to(device)



loss_fn = nn.CrossEntropyLoss(weight = weighted)
optimizer = torch.optim.AdamW(params = filter(lambda p: p.requires_grad, model.parameters()),
                               lr=0.0001, weight_decay = 0.01)

print(f"Weight Tensor: {weighted}")
print(class_names)


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(tqdm(dataloader, desc = "Training...")):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        accuracy = (y_pred.argmax(dim = 1) == y).sum().item() / len(y)
        train_acc += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, optimizer, device):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(dataloader, desc = "Testing...")):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            accuracy = ((test_pred.argmax(dim = 1) == y).sum().item() / len(y))
            test_acc += accuracy
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc
    
def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs):
    best_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, optimizer, device)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Best Accuracy: {best_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "models/emotion_model.pth")


train(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs=10)