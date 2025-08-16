import config
import os
import torch
import torchvision
from tqdm import tqdm
from datasets import Handwriting_Tensor
from helper import get_data_loader
from torchvision.models import Wide_ResNet50_2_Weights, resnet18, ResNet18_Weights

# ── pretrained Wide-ResNet-50-2 trunk (no avg-pool, no FC) ─────────────────────
# wrn = torchvision.models.wide_resnet50_2(
#     weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1
# ).eval().to(config.DEVICE)

resnet = resnet18(weights=ResNet18_Weights.DEFAULT).eval().to(config.DEVICE)

feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-3]).eval().to(config.DEVICE)

for p in feature_extractor.parameters():
    p.requires_grad = False

# ── cache CNN features for an entire loader ───────────────────────────────────
def save_features(loader, save_dir, saved=False):
    os.makedirs(save_dir ,exist_ok = True)  
    feats_labels = []
    feats_target_lengths = []
    feats_targets = []
    idx = 0
    with torch.no_grad():
        for imgs, labels, target_lengths, targets in tqdm(loader, desc="Loading"):            # works with your my_collate_fn
            targets = targets.to(config.DEVICE)
            if not saved:
                imgs = imgs.to(config.DEVICE)                # (B, C, 32, W)
                
    
                if imgs.shape[1] == 1:                # replicate if grayscale
                    imgs = imgs.repeat(1, 3, 1, 1)
                f = feature_extractor(imgs)           # (B, 2048, 1, W′)
                #f = f.flatten(2).permute(0, 2, 1)  # (B, W′, 2048)
                #f = (f - f.mean(dim=2, keepdim=True)) / (f.std(dim=2, keepdim=True) + 1e-8)
                
                torch.save(f, os.path.join(save_dir, f"features_{idx}.pt"))
            feats_labels.extend(labels)
            feats_target_lengths.extend(target_lengths)
            feats_targets.extend([t.to(config.DEVICE) for t in targets])

            idx += 1

    return feats_targets, feats_labels, feats_target_lengths

# ── extract & wrap in TensorDatasets ──────────────────────────────────────────
def get_features():
    train_loader, val_loader, test_loader = get_data_loader(bs=128)

    train_path = "./features/train_features"
    val_path = "./features/val_features"
    test_path = "./features/test_features"

    train_targets, train_labels, train_target_lengths = save_features(train_loader, train_path, saved=config.SAVED_FEATURES)
    val_targets, val_labels, val_target_lengths = save_features(val_loader, val_path, saved=config.SAVED_FEATURES)
    test_targets, test_labels, test_target_lengths = save_features(test_loader, test_path, saved=config.SAVED_FEATURES)

    train_dataset = Handwriting_Tensor(train_path, train_targets, labels=train_labels, target_lengths=train_target_lengths)
    val_dataset = Handwriting_Tensor(val_path, val_targets, labels=val_labels, target_lengths=val_target_lengths)
    test_dataset = Handwriting_Tensor(test_path, test_targets, labels=test_labels, target_lengths=test_target_lengths)

    return train_dataset, val_dataset, test_dataset

def get_test_features():
    _, _, test_loader = get_data_loader(bs=128)

    test_path = "./features/test_features"

    test_targets, test_labels, test_target_lengths = save_features(test_loader, test_path, saved=config.SAVED_FEATURES)

    test_dataset = Handwriting_Tensor(test_path, test_targets, labels=test_labels, target_lengths=test_target_lengths)

    return test_dataset