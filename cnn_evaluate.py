import torch
import torch.nn as nn
import cnn_config

def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    total_pred = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in loader:

                imgs, labels = imgs.to(cnn_config.DEVICE), labels.to(cnn_config.DEVICE)
                
                outputs = model(imgs)

                loss = criterion(outputs, labels)

                # Loss and accuracy calculations
                total_loss += loss.item() * imgs.size(0)
                _ , pred = outputs.max(1)
                total += labels.size(0)
                total_pred += pred.eq(labels).sum().item()

    # Saving loss and accuracy
    return (float(total_loss)/total), (float(total_pred)/total)