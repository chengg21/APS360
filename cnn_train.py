import torch
import torch.nn as nn
import numpy as np
import os
import time
import cnn_config
from tqdm import tqdm
from cnn_evaluate import evaluate
from cnn_helper import EarlyStopping, get_model_name, plot_curve, get_data_loader
from cnn_model import EMNISTCNN


#ls = learning rate, bs = batch size
def train(debug = False):
    num_epochs = cnn_config.NUM_EPOCHS
    torch.manual_seed(1000)

    train_loader, val_loader, _ = get_data_loader(bs=cnn_config.BS)

    model = EMNISTCNN(num_classes=62).to(cnn_config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cnn_config.LR, weight_decay=cnn_config.DECAY)

    # loss and error tracking
    train_char_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_char_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    early_stopper = EarlyStopping(patience=15)

    #Creating folder
    folder_name = get_model_name(model, cnn_config.LR, cnn_config.BS, cnn_config.DECAY, num_epochs, debug)
    save_dir = "./cnn_models/"+folder_name
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total = 0
        total_pred = 0
        #Train
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):

                imgs, labels = imgs.to(cnn_config.DEVICE), labels.to(cnn_config.DEVICE)
                
                optimizer.zero_grad()
                outputs = model(imgs)

                loss = criterion(outputs, labels)

                loss.backward()

                #nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                optimizer.step()

                # Loss and accuracy calculations
                total_loss += loss.item() * imgs.size(0)
                _ , pred = outputs.max(1)
                total += labels.size(0)
                total_pred += pred.eq(labels).sum().item()

        # Saving loss and accuracy
        train_loss[epoch] = float(total_loss)/total
        train_char_acc[epoch] = float(total_pred)/total
        val_loss[epoch], val_char_acc[epoch] = evaluate(model, val_loader)

        # Printing loss and acc
        if (epoch % 10 == 0):
            print("Train Loss: {:.4f} | Train Char Acc {:.4f}% | Val Loss: {:.4f} | Val Char Acc {:.4f}%".format(
                train_loss[epoch], train_char_acc[epoch]*100, val_loss[epoch], val_char_acc[epoch]*100))
        # Saving model (set to 10 times)
        if (epoch % (num_epochs/20) == 0) or (epoch == num_epochs - 1):
            model_name = f"{model.name}_epoch_{epoch}"
            torch.save({
                'epoch': epoch,
                'model_classifier_state_dict': model.classifier.state_dict(),
                'model_cnn_state_dict': model.cnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_dir}/{model_name}.pth")

        early_stopper(val_char_acc[epoch])
        if early_stopper.early_stop:
            print("Early stopping triggered")
            train_char_acc = train_char_acc[:epoch]
            train_loss = train_loss[:epoch]
            val_char_acc = val_char_acc[:epoch]
            val_loss = val_loss[:epoch]
            model_name = f"{model.name}_epoch_{epoch}"
            torch.save({
                'epoch': epoch,
                'model_classifier_state_dict': model.classifier.state_dict(),
                'model_cnn_state_dict': model.cnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_dir}/{model_name}.pth")
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken: {total_time}")
    model_dir = save_dir+"/"+model_name

    # Saving plotting values
    np.savetxt(f"{model_dir}_train_loss.csv", train_loss)
    np.savetxt(f"{model_dir}_train_char_err.csv", train_char_acc)
    np.savetxt(f"{model_dir}_val_loss.csv", val_loss)
    np.savetxt(f"{model_dir}_val_char_err.csv", val_char_acc)
    plot_curve(model_dir)

if __name__ == "__main__":
    train()