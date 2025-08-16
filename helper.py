import numpy as np
import torch
import matplotlib.pyplot as plt
import config
from preprocess import preprocess_data, preprocess_testing_data
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import Handwriting_Images, my_collate_fn

class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_score):
        score = -val_score  # assuming lower CER is better
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def get_model_name(model, lr, bs, decay, num_epochs, debug):
  path = "{0}_hidden{1}_layers{2}_lr{3}_bs{4}_decay{5}_epoch{6}".format(model.name, model.hidden_size, model.num_layers, lr, bs, decay, num_epochs)
  if (debug):
    path += "_debug"
  return path

def plot_curve(path, show = True, axes = None):
  train_loss = np.loadtxt(f"{path}_train_loss.csv")
  train_char_err = np.loadtxt(f"{path}_train_char_err.csv")
  train_word_err = np.loadtxt(f"{path}_train_word_err.csv")
  val_loss = np.loadtxt(f"{path}_val_loss.csv")
  val_char_err = np.loadtxt(f"{path}_val_char_err.csv")
  val_word_err = np.loadtxt(f"{path}_val_word_err.csv")

  num_epoch = len(train_loss)

  if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))

  axes[0].set_title("Train vs Validation Loss")
  axes[0].plot(range(1, num_epoch+1), train_loss, label="Train")
  axes[0].plot(range(1, num_epoch+1), val_loss, label="Validation")
  axes[0].set(xlabel="Epoch", ylabel=("Loss"))
  axes[0].legend(loc="best")

  axes[1].set_title("Train vs Validation Character Error")
  axes[1].plot(range(1, num_epoch+1), train_char_err, label="Train")
  axes[1].plot(range(1, num_epoch+1), val_char_err, label="Validation")
  axes[1].set(xlabel="Epoch", ylabel=("Character Error"))
  axes[1].legend(loc="best")

  axes[2].set_title("Train vs Validation Word Error")
  axes[2].plot(range(1, num_epoch+1), train_word_err, label="Train")
  axes[2].plot(range(1, num_epoch+1), val_word_err, label="Validation")
  axes[2].set(xlabel="Epoch", ylabel=("Word Error"))
  axes[2].legend(loc="best")

  if show:
    plt.show()

def plot_curves(path_list):
  n = len(path_list)
  fig = plt.figure(constrained_layout=True, figsize=(20, 6 * n))
  subfigs = fig.subfigures(n, 1)

  for i, path in enumerate(path_list):
      axes = subfigs[i].subplots(1, 3, sharex=True)
      subfigs[i].suptitle(path, fontsize=12)
      plot_curve(path, show=False, axes=axes)

  plt.show()

#rescales tensor img to 224 x 9408
class RescaleTransform:
  def __init__(self):
    self.fixed_height = 56
    self.fixed_width = 2352

  def __call__(self,img):
    old_width, old_height = img.size
    new_width = int(old_width * (self.fixed_height / old_height))
    rescale_transform = transforms.Compose(
        [transforms.Resize((self.fixed_height, new_width)),
        transforms.Pad((0, 0, self.fixed_width - new_width, 0), 255),
        ])
    return rescale_transform(img)

def get_data_loader(bs, debug = False):

  np.random.seed(1000)
  
  smhd_labels, iam_labels = preprocess_data()

  train_transforms = transforms.Compose([
      transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.5),
      transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05)),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
      RescaleTransform(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  normal_transform = transforms.Compose([
      RescaleTransform(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  # Initializes all datasets
  augmented_iam_dataset = Handwriting_Images(root=config.IAM_SAVE_PATH, labels=iam_labels, transform=train_transforms)
  standard_iam_dataset = Handwriting_Images(root=config.IAM_SAVE_PATH, labels=iam_labels, transform=normal_transform)
  #print(len(iam_dataset))
  augmented_smhd_dataset = Handwriting_Images(root=config.SMHD_SAVE_PATH, labels=smhd_labels, transform=train_transforms)
  standard_smhd_dataset = Handwriting_Images(root=config.SMHD_SAVE_PATH, labels=smhd_labels, transform=normal_transform)
  #print(len(smhd_dataset))
  #emist_dataset = Handwriting_Images(root=emnist_save_path, labels=emnist_labels, transform=normal_transform)
  test_labels = preprocess_testing_data()
  standard_test_dataset = Handwriting_Images(root=config.TEST_SAVE_PATH, labels=test_labels, transform=normal_transform)

  # Combines them in a list
  augmented_combined_dataset = []
  augmented_combined_dataset.append(augmented_iam_dataset)
  augmented_combined_dataset.append(augmented_smhd_dataset)
  
  standard_combined_dataset = []
  standard_combined_dataset.append(standard_iam_dataset)
  standard_combined_dataset.append(standard_smhd_dataset)

  test_combined_dataset = []
  test_combined_dataset.append(standard_test_dataset)
  #combined_dataset.append(emist_dataset)

  # Merges list as one dataset
  total_augmented_dataset = torch.utils.data.ConcatDataset(augmented_combined_dataset)
  total_standard_dataset = torch.utils.data.ConcatDataset(standard_combined_dataset)
  total_testing_dataset = torch.utils.data.ConcatDataset(test_combined_dataset)

  # Marks all indices
  indices = []
  if debug:
    for i in range(bs):
      indices.append(i)
  else:
    for i in range(len(total_augmented_dataset)):
      indices.append(i)

  # print(len(indices))
  # Shuffles indices
  np.random.shuffle(indices)

  # Split indices (train:val:test = 80:10:10)
  train_val_split = int(len(indices) * 0.8) #split at 80%
  val_test_split = train_val_split + int(len(indices) * 0.1) #split at 90%

  # Split into training and validation indices
  train_indices, val_indices, test_indices = indices[:train_val_split], indices[train_val_split:val_test_split], indices[val_test_split:]

  test_indices = [i for i in range(len(total_testing_dataset))]

  # Create sampler objects
  train_sampler = SubsetRandomSampler(train_indices)
  train_loader = torch.utils.data.DataLoader(total_augmented_dataset, batch_size=bs, collate_fn=my_collate_fn,
                                               sampler=train_sampler)
  val_sampler = SubsetRandomSampler(val_indices)
  val_loader = torch.utils.data.DataLoader(total_standard_dataset, batch_size=bs, collate_fn=my_collate_fn,
                                              sampler=val_sampler)

  test_sampler = SubsetRandomSampler(test_indices)
  test_loader = torch.utils.data.DataLoader(total_testing_dataset, batch_size=bs, collate_fn=my_collate_fn,
                                             sampler=test_sampler)

  return train_loader, val_loader, test_loader

def levenshtein(a, b):
    # Returns edit-distance between two strings
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    if len(a) > len(b):
        a, b = b, a

    prev_row = range(len(a) + 1)
    for i, cb in enumerate(b, start=1):
        cur_row = [i]
        for j, ca in enumerate(a, start=1):
            insert_cost = cur_row[j-1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j-1] + (ca != cb)
            cur_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = cur_row
    return prev_row[-1]

if __name__ == "__main__":
    plot_curve("models/test12_frozen_cnn_hidden256_layers2_lr0.0001_bs128_decay1e-10_epoch300/test12_frozen_cnn_epoch_203")
