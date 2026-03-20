import os
import numpy as np
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset



class CMGroupDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_gids = np.load(os.path.join(root, 'npy', 'train_id.npy')).tolist()
            selected_gids = train_gids 
        else:
            test_gids = np.load(os.path.join(root, 'npy', 'test_id.npy')).tolist()
            selected_gids = test_gids

        img_paths = glob(os.path.join(root, '**/*.jpeg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2].split('_')[0]) in selected_gids]

        if mode == 'gallery':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 3)]
        elif mode == 'query':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (4, 5, 6)]

        img_paths = sorted(img_paths)
        gp_ids = [path.split('/')[-2] for path in img_paths]
        gp_ids = np.unique(gp_ids)
        num_ids = len(gp_ids)

        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]

        self.num_ids = num_ids
        self.transform = transform
          
        ## relabel

        if mode == 'train':
            id_map = dict(zip(gp_ids, range(num_ids)))
            self.ids = [id_map[path.split('/')[-2]] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2].split('_')[-1]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]

        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item


