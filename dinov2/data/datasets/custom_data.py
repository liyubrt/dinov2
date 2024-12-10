from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, TiffImagePlugin
from torchvision.datasets import DatasetFolder

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)
logging.getLogger("PIL.TiffImagePlugin").setLevel(51)
TiffImagePlugin.DEBUG = False

def pil_loader(p):
    return Image.open(p).convert("RGB")


class CustomData(ExtendedVisionDataset):

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.data = DatasetFolder(root, loader=pil_loader, extensions=["jpg"])

    def get_image_data(self, index: int) -> bytes:
        return self.data[index][0]

    def get_target(self, index: int) -> Optional[int]:
        return 0

    def __len__(self) -> int:
        return len(self.data)


class CustomLargeDataset(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        # load OpenImages data df
        self.oi_data_path = '/data2/jupiter/datasets/OpenImages'
        oi_label_df = pd.read_csv(os.path.join(self.oi_data_path, 'annotations', f'train_600classes.csv'))
        oi_label_df = oi_label_df[(oi_label_df.corrupted == False)]
        oi_label_df['data'] = 'openimages'
        # load COYO300M data df
        self.coyo_data_path = '/data2/jupiter/datasets/coyo-700m-webdataset'
        coyo_label_df = pd.read_parquet(os.path.join(self.coyo_data_path, 'matches_downloaded', 'label_02p.parquet'))
        coyo_label_df = coyo_label_df[(coyo_label_df.corrupted == False)]
        coyo_label_df['data'] = 'coyo300m'
        # combine two dfs
        self.df = pd.concat([oi_label_df[['data', 'ImageID']], coyo_label_df[['data', 'part1m_dir', 'key']]], ignore_index=True)
        
    def __len__(self):
        return len(self.df)
    
    def get_image_data(self, index):
        row = self.df.iloc[index]
        if row['data'] == 'openimages':
            img_path = os.path.join(self.oi_data_path, 'train', row['ImageID']+'.jpg')
        else:
            img_path = os.path.join(self.coyo_data_path, row['part1m_dir'], f'{str(int(row.key)).zfill(9)}.jpg')
        return Image.open(img_path).convert('RGB')

    def get_target(self, index: int) -> Optional[int]:
        return 0