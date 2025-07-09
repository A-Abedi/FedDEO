from os import path
from PIL import Image
from torch.utils.data import Dataset

class OfficeCaltechDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
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
        return img, self.labels[index]


def _read_txt_list(txt_file, base_path=None):
    imgs, labels = [], []
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, label = line.split()
            if base_path is not None and not path.isabs(img_path):
                img_path = path.join(base_path, img_path)
            imgs.append(img_path)
            labels.append(int(label))
    return imgs, labels


def get_officecaltech_dataset(list_dir, domain, transform=None, base_path=None):
    txt_file = path.join(list_dir, f"{domain}.txt")
    imgs, labels = _read_txt_list(txt_file, base_path)
    return OfficeCaltechDataset(imgs, labels, transform)


def get_officecaltech_multi(list_dir, domains, transform=None, base_path=None):
    all_imgs, all_labels = [], []
    for d in domains:
        imgs, labels = _read_txt_list(path.join(list_dir, f"{d}.txt"), base_path)
        all_imgs.extend(imgs)
        all_labels.extend(labels)
    return OfficeCaltechDataset(all_imgs, all_labels, transform)
