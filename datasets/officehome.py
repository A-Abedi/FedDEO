from os import path
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class OfficeHomeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

def _read_txt_list(txt_path, base_path=None):
    images = []
    labels = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, label = line.split()
            if base_path is not None and not path.isabs(img_path):
                img_path = path.join(base_path, img_path)
            images.append(img_path)
            labels.append(int(label))
    return images, labels

def get_officehome_dataset(list_dir, domain, transform=None, base_path=None):
    txt_file = path.join(list_dir, f"{domain}.txt")
    imgs, labels = _read_txt_list(txt_file, base_path)
    return OfficeHomeDataset(imgs, labels, transform)

def get_officehome_multi(list_dir, domains, transform=None, base_path=None):
    all_imgs, all_labels = [], []
    for d in domains:
        imgs, labels = _read_txt_list(path.join(list_dir, f"{d}.txt"), base_path)
        all_imgs.extend(imgs)
        all_labels.extend(labels)
    return OfficeHomeDataset(all_imgs, all_labels, transform)
