import torchvision
import os
import cnn_config

def preprocess(download=False, limit = 1000):
    if download:
        emnist_dataset = torchvision.datasets.EMNIST(root='.',
                                            split="byclass",
                                            download=download)

        emnist_save_path = cnn_config.EMNIST_SAVE_PATH

        # Limiter to balance how many each character is produced
        emnist_count = {char: 0 for char in emnist_dataset.classes}

        filled = False

        for idx, (img, label) in enumerate(emnist_dataset):
            # Note: label is the indice of classes
            char = emnist_dataset.classes[label] # Gets the character of the image

            os.makedirs(f"{emnist_save_path}/{char}", exist_ok=True)
            # Creates a directory with the character

            # Adds the image to the dataset
            if (emnist_count[char] < limit):
                img.save(f"{emnist_save_path}/{char}/{idx}.png")
            emnist_count[char] += 1
            if ((emnist_count[x] for x in emnist_dataset.classes) == limit):
                filled = True
            if filled:
                break

