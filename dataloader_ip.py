import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class IntenPhaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_info = self.load_dataset()

    def load_dataset(self):
        data_info = []

        for label, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_dir = os.path.join(self.root_dir, class_name)
            inten_dir = os.path.join(class_dir, "inten")
            inten_images = glob.glob(os.path.join(inten_dir, "*.tiff"))

            for inten_path in inten_images:
                pha_path = inten_path.replace("inten", "pha")
                data_info.append((inten_path, pha_path, label))
        
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        inten_path, pha_path, label = self.data_info[idx]

        inten_image = Image.open(inten_path).convert("RGB")
        pha_image = Image.open(pha_path).convert("RGB")

        if self.transform:
            inten_image = self.transform(inten_image)
            pha_image = self.transform(pha_image)
        
        return inten_image, pha_image, label, inten_path, pha_path


class IntenPhaTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.inten_files = sorted(glob.glob(os.path.join(root_dir, 'inten', '*.tiff')))
   
        self.paired_files = [
            (inten_file, inten_file.replace("inten", "pha") if os.path.exists(inten_file.replace("inten", "pha")) else inten_file)
            for inten_file in self.inten_files
        ]

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        inten_path, pha_path = self.paired_files[idx]

        inten_image = Image.open(inten_path).convert("RGB")
        pha_image = Image.open(pha_path).convert("RGB")

        if self.transform:
            inten_image = self.transform(inten_image)
            pha_image = self.transform(pha_image)

        return inten_image, pha_image, inten_path, pha_path