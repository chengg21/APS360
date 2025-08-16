from evaluate import evaluate
import re
from model import CRNN_WRN
import config
import torch
import torch.nn as nn
from tokenizer import Tokenizer
from features import get_test_features
from torch.utils.data import DataLoader
from datasets import custom_loaded_collate_fn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_model(path):
    hidden_match = re.search(r"hidden(\d+)", path)
    layers_match = re.search(r"layers(\d+)", path)

    hidden_size = int(hidden_match.group(1)) if hidden_match else None
    num_layers = int(layers_match.group(1)) if layers_match else None

    model = CRNN_WRN(hidden_size=hidden_size,num_layers=num_layers).to(config.DEVICE)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
def predict(model):
    criterion = nn.CTCLoss(zero_infinity=True)
    tokenizer = Tokenizer()

    test_dataset = get_test_features()

    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=custom_loaded_collate_fn)

    loss, cer, wer = evaluate(model, test_loader, criterion, tokenizer)

    print(f"Loss: {loss}, Character Error Rate: {cer}, Word Error Rate: {wer}")

def single_predict(model, times):
    test_dataset = get_test_features()

    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_loaded_collate_fn)

    single_loader = iter(test_loader)

    tokenizer = Tokenizer()
    for n in range(times):
        img, label, target_length, target = next(single_loader)
        img = img.to(config.DEVICE)
        target_length = target_length.to(dtype=torch.long, device=config.DEVICE)
        target = target.to(config.DEVICE)

        output = model(img)
        log_probs = F.log_softmax(output, dim=2)

        input_lengths = torch.full(size=(1,), fill_value=output.size(0),
                                        dtype=torch.long, device = config.DEVICE)
        
        pred_sentence = tokenizer.decode(log_probs, input_lengths)

        print(f"Label: {label} \n")
        print(f"Prediction: {pred_sentence} \n")

        plot_embeddings(output)

def plot_embeddings(output):
    out = output.detach().cpu()

    if out.dim() == 3:
        if out.shape[1] == 1:
            out = out.squeeze(1)   # (T, C)
        elif out.shape[0] == 1:
            out = out.squeeze(0)   # (T, C)

    plt.imshow(out.T, aspect='auto', cmap='viridis')  # (C,T) due to transpose
    plt.colorbar()
    plt.xlabel('Time Steps')
    plt.ylabel('Embedding dimension')
    plt.show()


if __name__ == "__main__":
    model = get_model("models/test12_frozen_cnn_hidden256_layers2_lr0.0001_bs128_decay1e-10_epoch300/test12_frozen_cnn_epoch_150.pth")
    predict(model)