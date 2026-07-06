import os
import re

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config as cfg


class MyDataset(Dataset):
    """
    Read images and labels from filenames.

    Required filename format:
      <variant>_digit_X_NNNNNN.png

    For regular dataset folders, the accepted variant prefix is defined by
    cfg.DATASET_FILENAME_VARIANTS. For the special training folder, variants
    are bw/rbw/bwr.
    """

    FILENAME_PATTERN = re.compile(
        r"^(?P<variant>.+)_digit_(?P<label>[0-9])_(?P<index>[0-9]+)\.png$",
        re.IGNORECASE,
    )

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.expected_variants = self._expected_variants_for_dir(root_dir)
        self.samples = []

        for filename in sorted(os.listdir(root_dir)):
            if not filename.lower().endswith(".png"):
                continue
            label = self._parse_label(filename)
            self.samples.append((filename, label))
            random.shuffle(self.samples)

    @staticmethod
    def _expected_variants_for_dir(root_dir):
        dataset_name = os.path.basename(os.path.normpath(root_dir)).lower()
        return {
            variant.lower()
            for variant in cfg.DATASET_FILENAME_VARIANTS.get(dataset_name, (dataset_name,))
        }

    def _parse_label(self, filename):
        match = self.FILENAME_PATTERN.match(filename)
        if not match:
            raise ValueError(
                f"Unexpected filename format '{filename}'. "
                "Expected '<variant>_digit_X_NNNNNN.png'."
            )

        variant = match.group("variant").lower()
        if variant not in self.expected_variants:
            expected = ", ".join(sorted(self.expected_variants))
            raise ValueError(
                f"Unexpected filename variant '{variant}' in '{filename}'. "
                f"Expected one of: {expected}."
            )

        return int(match.group("label"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label


# Shared transforms for training/evaluation images.
train_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(cfg.MEAN, cfg.STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(cfg.MEAN, cfg.STD),
])
