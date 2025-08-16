import numpy as np
import torch
import matplotlib.pyplot as plt
import cnn_config
from cnn_preprocess import preprocess
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_score):
        score = val_score  # assuming lower CER is better
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
  path = "{0}_lr{1}_bs{2}_decay{3}_epoch{4}".format(model.name, lr, bs, decay, num_epochs)
  if (debug):
    path += "_debug"
  return path

def plot_curve(path, show = True, axes = None):
  train_loss = np.loadtxt(f"{path}_train_loss.csv")
  train_char_err = np.loadtxt(f"{path}_train_char_err.csv")
  val_loss = np.loadtxt(f"{path}_val_loss.csv")
  val_char_err = np.loadtxt(f"{path}_val_char_err.csv")

  num_epoch = len(train_loss)

  if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))

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

  if show:
    plt.show()

def plot_curves(path_list):
  n = len(path_list)
  fig = plt.figure(constrained_layout=True, figsize=(20, 6 * n))
  subfigs = fig.subfigures(n, 1)

  for i, path in enumerate(path_list):
      axes = subfigs[i].subplots(1, 2, sharex=True)
      subfigs[i].suptitle(path, fontsize=12)
      plot_curve(path, show=False, axes=axes)

  plt.show()

#rescales tensor img to 224 x 9408
class RescaleTransform:
  def __init__(self):
    self.fixed_height = 56

  def __call__(self,img):
    rescale_transform = transforms.Resize((self.fixed_height, self.fixed_height))
    return rescale_transform(img)

def get_data_loader(bs, download = False, limit = 1000, debug = False):

  np.random.seed(1000)

  preprocess(download = download, limit = limit)

  train_transforms = transforms.Compose([
      transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.5),
      transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05)),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
      RescaleTransform(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
  ])

  normal_transform = transforms.Compose([
      RescaleTransform(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
  ])

  # Initializes all datasets
  augmented_emnist_dataset = ImageFolder(root=cnn_config.EMNIST_SAVE_PATH, transform=train_transforms)
  standard_emnist_dataset = ImageFolder(root=cnn_config.EMNIST_SAVE_PATH, transform=normal_transform)

  # Marks all indices
  indices = []
  if debug:
    for i in range(bs):
      indices.append(i)
  else:
    for i in range(len(standard_emnist_dataset)):
      indices.append(i)

  # print(len(indices))
  # Shuffles indices
  np.random.shuffle(indices)

  # Split indices (train:val:test = 80:10:10)
  train_val_split = int(len(indices) * 0.8) #split at 80%
  val_test_split = train_val_split + int(len(indices) * 0.1) #split at 90%

  # Split into training and validation indices
  train_indices, val_indices, test_indices = indices[:train_val_split], indices[train_val_split:val_test_split], indices[val_test_split:]

  # Create sampler objects
  train_sampler = SubsetRandomSampler(train_indices)
  train_loader = torch.utils.data.DataLoader(augmented_emnist_dataset, batch_size=bs, sampler=train_sampler)

  val_sampler = SubsetRandomSampler(val_indices)
  val_loader = torch.utils.data.DataLoader(standard_emnist_dataset, batch_size=bs, sampler=val_sampler)

  test_sampler = SubsetRandomSampler(test_indices)
  test_loader = torch.utils.data.DataLoader(standard_emnist_dataset, batch_size=bs, sampler=test_sampler)

  return train_loader, val_loader, test_loader

if __name__ == "__main__":
    plot_curves([
       "models/test2_hidden256_layers2_lr0.0001_bs64_decay1e-10_epoch250/test2_epoch_249",
       "models/test2_hidden256_layers2_lr0.0001_bs256_decay1e-10_epoch1000/test2_epoch_999",
       "models/test5_hidden256_layers2_lr0.0001_bs64_decay1e-10_epoch400/test5_epoch_399",
    ])
