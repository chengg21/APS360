import torch
import torch.nn.functional as F
import config
from helper import levenshtein

def evaluate(model, loader, criterion, tokenizer):
    torch.manual_seed(1000)
    model.eval()
    total_loss = 0.0
    total_char_errs = 0.0
    total_word_errs = 0.0
    total_items  = 0
    total_chars = 0
    total_words = 0
    with torch.no_grad():
        for imgs, labels, target_lengths, targets in loader:
            # sample, label, target_length, target
            imgs = imgs.to(config.DEVICE)
            target_lengths = target_lengths.to(dtype=torch.long, device=config.DEVICE)
            targets = targets.to(config.DEVICE)
            batch_size = imgs.size(0)

            ################## Model output
            output = model(imgs)
            log_probs = F.log_softmax(output, dim=2)
            input_lengths = torch.full(size=(batch_size,), fill_value=output.size(0),
                                        dtype=torch.long, device = config.DEVICE)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
            total_loss += loss.item() * batch_size
            total_items += batch_size

            pred_sentences = tokenizer.decode(log_probs, input_lengths)

            for pred, truth in zip(pred_sentences, labels):
                        pred  = pred.strip()
                        truth = truth.strip()

                        # Character level (ignoring spaces)
                        pred_c  = pred.replace(" ", "")
                        truth_c = truth.replace(" ", "")
                        total_char_errs += levenshtein(pred_c, truth_c)
                        total_chars += len(truth_c)

                        # Word level
                        pred_words  = pred.split()
                        truth_words = truth.split()
                        total_word_errs += levenshtein(pred_words, truth_words)
                        total_words += len(truth_words)

            del imgs, labels, targets, output, log_probs, input_lengths
            
    avg_loss = total_loss / total_items
    cer = min(1.0, float(total_char_errs) / max(1, total_chars))
    wer = min(1.0, total_word_errs / max(1, total_words))
    return avg_loss, cer, wer