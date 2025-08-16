import torch
import torch.nn as nn
import numpy as np
import os
import time
import config
import torch.nn.functional as F
from features import get_features
from tqdm import tqdm
from evaluate import evaluate
from helper import EarlyStopping, get_model_name, levenshtein, plot_curve, get_data_loader
from datasets import custom_loaded_collate_fn, my_collate_fn
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from model import CRNN_WRN


#ls = learning rate, bs = batch size
def train(debug = False):
    num_epochs = config.NUM_EPOCHS
    torch.manual_seed(1000)

    model = CRNN_WRN(hidden_size=config.HIDDEN_SIZE,num_layers=config.NUM_LAYERS).to(config.DEVICE)

    criterion = nn.CTCLoss(zero_infinity=True)    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.DECAY)
    
    #checkpoint = torch.load('models/test5_hidden256_layers2_lr0.0001_bs64_decay1e-10_epoch200/test5_epoch_199.pth')
    #model.load_state_dict(checkpoint)

    train_dataset, val_dataset, _ = get_features()
    train_loader = DataLoader(train_dataset, batch_size=config.BS, collate_fn=custom_loaded_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BS, collate_fn=custom_loaded_collate_fn)

    # loss and error tracking
    train_word_err = np.zeros(num_epochs)
    train_char_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_word_err = np.zeros(num_epochs)
    val_char_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    tokenizer = Tokenizer()

    test_print = True

    early_stopper = EarlyStopping(patience=50)

    #Creating folder
    folder_name = get_model_name(model, config.LR, config.BS, config.DECAY, num_epochs, debug)
    save_dir = "./models/"+folder_name
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_char_errs = 0.0
        total_train_word_errs = 0.0
        total_chars = 0
        total_words = 0
        total_items = 0
        #Train
        model.train()
        for _, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
                # return sample, label, target_length, target
                imgs, labels, target_lengths, targets = data
                imgs = imgs.to(config.DEVICE) # ex. [148, 1, 2048] = [batch size, seq len, channels]
                target_lengths = target_lengths.to(dtype=torch.long, device=config.DEVICE)
                targets = targets.to(config.DEVICE)
                bs = imgs.size(0) #just in case for last batch size
                
                optimizer.zero_grad()
                output = model(imgs)
                input_lengths = torch.full(size=(bs,), fill_value=output.size(0),
                                                dtype=torch.long, device = config.DEVICE)

                if debug and test_print:
                    #print(imgs)
                    print(f"img: {imgs.shape}")
                    #print(labels)
                    print(len(labels))
                    #print(target_lengths)
                    print(f"target_lengths: {target_lengths.shape}")
                    #print(targets)
                    print(f"bs: {bs}")
                    #print(output)
                    print(f"output: {output.shape}")
                    #print(input_lengths)
                    print(f"input_lengths: {input_lengths.shape}")
                    test_print = False

                log_probs = F.log_softmax(output, dim=2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                optimizer.step()

                # Loss, CER, and WER calculations
                #print(log_probs.shape)
                total_items += bs
                total_train_loss += loss.item() * bs
                pred_sentences = tokenizer.decode(log_probs, input_lengths)

                for pred, truth in zip(pred_sentences, labels):
                        pred  = pred.strip()
                        truth = truth.strip()

                        # Character level (ignoring spaces)
                        pred_c  = pred.replace(" ", "")
                        truth_c = truth.replace(" ", "")
                        total_train_char_errs += levenshtein(pred_c, truth_c)
                        total_chars += len(truth_c)

                        # Word level
                        pred_words  = pred.split()
                        truth_words = truth.split()
                        total_train_word_errs += levenshtein(pred_words, truth_words)
                        total_words += len(truth_words)

                del imgs, labels, targets, output, log_probs, input_lengths




        # Saving loss, CER, and WER
        train_loss[epoch] = float(total_train_loss)/total_items
        train_char_err[epoch] = min(1.0, float(total_train_char_errs)/ max(1, total_chars))
        train_word_err[epoch] = min(1.0, float(total_train_word_errs)/ max(1, total_words))
        val_loss[epoch], val_char_err[epoch], val_word_err[epoch] = evaluate(model, val_loader, criterion, tokenizer)

        # Printing loss and acc
        if (epoch % 10 == 0):
            print("Train Loss: {:.4f} | Train Char Err {:.4f}% | Train Word Err {:.4f}% | Val Loss: {:.4f} | Val Char Err {:.4f}% | Val Word Err {:.4f}% ".format(
                train_loss[epoch], train_char_err[epoch]*100, train_word_err[epoch]*100, val_loss[epoch], val_char_err[epoch]*100, val_word_err[epoch]*100))
        # Saving model (set to 10 times)
        if (epoch % (num_epochs/10) == 0) or (epoch == num_epochs - 1):
            model_name = f"{model.name}_epoch_{epoch}"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_dir}/{model_name}.pth")

        early_stopper(val_char_err[epoch])
        if early_stopper.early_stop:
            print("Early stopping triggered")
            train_char_err = train_char_err[:epoch]
            train_word_err = train_word_err[:epoch]
            train_loss = train_loss[:epoch]
            val_word_err = val_word_err[:epoch]
            val_char_err = val_char_err[:epoch]
            val_loss = val_loss[:epoch]
            model_name = f"{model.name}_epoch_{epoch}"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_dir}/{model_name}.pth")
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken: {total_time}")
    model_dir = save_dir+"/"+model_name

    # Saving plotting values
    np.savetxt(f"{model_dir}_train_loss.csv", train_loss)
    np.savetxt(f"{model_dir}_train_char_err.csv", train_char_err)
    np.savetxt(f"{model_dir}_train_word_err.csv", train_word_err)
    np.savetxt(f"{model_dir}_val_loss.csv", val_loss)
    np.savetxt(f"{model_dir}_val_char_err.csv", val_char_err)
    np.savetxt(f"{model_dir}_val_word_err.csv", val_word_err)

    plot_curve(model_dir)

if __name__ == "__main__":
    train()