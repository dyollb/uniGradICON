import glob
import random
from pathlib import Path

import itk
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unigradicon import quantile
from my_config import DATASET_DIR



class MRIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Path,
        data_num=1000,
        desired_shape=None,
        labels_path: Path | None =None,
        request_labels: bool = False,
        device="cpu"
    ):
        self.request_labels = (labels_path is not None) and request_labels
        if not self.request_labels:
            t1_files, labels_files = list(data_path.glob("*.nii.gz")), []
        else:
            t1_files, labels_files = [], []
            for f in data_path.glob("*.nii.gz"):
                if (labels_path / f.name).exists():
                    t1_files.append(f)
                    labels_files.append(labels_path/f.name)

        print(f"Loading {data_path} data.")
        self.imgs = [torch.Tensor(np.asarray(itk.imread(f), dtype=float)).unsqueeze(0).unsqueeze(0) for f in tqdm(t1_files)]
        self.labels = [torch.Tensor(np.asarray(itk.imread(f), dtype=np.uint8)).unsqueeze(0).unsqueeze(0) for f in tqdm(labels_files)]

        print(f"Processing {data_path} data.")
        self.imgs = list(map(lambda x: self.process(x, desired_shape, modality="mri", device=device)[0], tqdm(self.imgs)))
        self.labels = list(map(lambda x: self.process(x, desired_shape, modality="seg", device=device)[0], tqdm(self.labels)))

        self.img_num = len(self.imgs)
        self.data_num = data_num
    
    def process(self, img, desired_shape=None, modality="mri", device="cpu"):
        img = img.to(device)
        if modality == "mri":
            im_min, im_max = torch.min(img), quantile(img.view(-1), 0.99)
            img = torch.clip(img, im_min, im_max)
            img = (img-im_min) / (im_max-im_min)
        
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear" if modality != "seg" else "nearest")
        return img.cpu()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        idx_a = random.randint(0, self.img_num - 1)
        idx_b = random.randint(0, self.img_num - 1)
        img_a = self.imgs[idx_a]
        img_b = self.imgs[idx_b]
        if self.request_labels:
            return img_a, img_b, self.labels[idx_a], self.labels[idx_b]
        return img_a, img_b


class IXIDataset(MRIDataset):
    def __init__(
        self,
        data_path=DATASET_DIR / "IXI" / "IXI-T1",
        data_num=1000,
        desired_shape=None,
        device="cpu"
    ):
        super().__init__(data_path=data_path, data_num=data_num, desired_shape=desired_shape, device=device)


class CCDataset(MRIDataset):
    def __init__(
        self,
        data_path=DATASET_DIR / "CC" / "t1w_n4_1mm",
        data_num=1000,
        desired_shape=None,
        device="cpu"
    ):
        super().__init__(data_path=data_path, data_num=data_num, desired_shape=desired_shape, device=device)


class DRCMRDataset(MRIDataset):
    def __init__(
        self,
        data_path=DATASET_DIR / "thielscher" / "coreg_0.85" / "t1w",
        data_num=1000,
        desired_shape=None,
        labels_path=DATASET_DIR / "thielscher" / "coreg_0.85" / "head40",
        request_labels=False,
        device="cpu"
    ):
        super().__init__(data_path=data_path, data_num=data_num, desired_shape=desired_shape, labels_path=labels_path, device=device, request_labels=request_labels)
