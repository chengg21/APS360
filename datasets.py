import torch
import glob
import os
from tokenizer import Tokenizer
from torchvision.datasets import ImageFolder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class Handwriting_Images(ImageFolder):
    def __init__(self, root, labels, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.labels = labels
        self.tokenizer = Tokenizer()

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)

        label = self.labels[index]
        target_length = torch.tensor(len(label), dtype=torch.long)
        target = torch.tensor(self.tokenizer.encode(label), dtype=torch.long)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, target_length, target
    

def my_collate_fn(batch):
    images, labels, target_lengths, targets = zip(*batch)
    # convert images to batch tensor (all same shape, so stack is fine)
    images = torch.stack(images)
    # keep lists of targets (no stack)

    target_lengths = torch.tensor([tl for tl in target_lengths], dtype=torch.long)

    # Convert each target (sequence) to a tensor
    targets = [t.clone().detach().long() for t in targets]

    # Pad the variable-length targets to the max length in the batch
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return images, labels, target_lengths, targets


class Handwriting_Tensor(Dataset):
    def __init__(self, feature_path, targets, labels, target_lengths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_paths = sorted(glob.glob(os.path.join(feature_path, "features_*.pt")))
        self.index_map = []
        self.targets = targets
        self.labels = labels
        self.target_lengths = target_lengths

        # Build index map
        for file_idx, fpath in enumerate(self.file_paths):
            batch = torch.load(fpath, map_location="cpu")
            for sample_idx in range(batch.shape[0]):
                self.index_map.append((file_idx, sample_idx))

        assert len(self.targets) == len(self.labels) == len(self.target_lengths), \
            "All input lists must have the same length"

        # Preload all batches into memory once
        self.cache = {}
        for file_idx, fpath in enumerate(self.file_paths):
            self.cache[file_idx] = torch.load(fpath, map_location="cpu")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        file_idx, sample_idx = self.index_map[index]
        batch = self.cache[file_idx]
        target = self.targets[index]
        label = self.labels[index]
        target_length = self.target_lengths[index]
        return batch[sample_idx], label, target_length, target
    
def custom_loaded_collate_fn(batch):
    # Unpack batch
    images, labels, target_lengths, targets = zip(*batch)

    # Stack image tensors (stay on CPU)
    images = torch.stack([img.cpu() for img in images], dim=0)  # (B, 148, 2048)

    # Keep labels as list of strings
    labels = list(labels)

    # Convert target_lengths to CPU tensor
    target_lengths = torch.stack([tl.cpu() for tl in target_lengths]).long()  # (B,)

    # Convert targets to CPU tensor and pad
    targets = [t.long().cpu() for t in targets]
    targets = pad_sequence(targets, batch_first=True, padding_value=0)  # (B, max_seq_len)

    return images, labels, target_lengths, targets