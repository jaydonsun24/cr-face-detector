import torch
import torch.nn as nn
from data_setup import train_dataloader, test_dataloader, device, class_to_idx
from model import model

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=0.001)

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        accuracy = (y_pred.argmax(dim = 1) == y).sum().item() / len(y)
        train_acc += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_step(model, dataloade,r loss_fn, optimizer, device):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()