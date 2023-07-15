import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class ScanNet(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):

        self.npoints = config.N_POINTS
        self.DATASET_PATH = config.DATASET_PATH
        self.category= config.category
        self.subset = config.subset

        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',  # random permutate points
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial']
            },{
                'callback': 'ToTensor',
                'objects': ['partial']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial']
            }])

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = {
            'taxonomy_id': [],
            'model_id': [],
            'partial_path': [],
        }
        if self.category == 'chair':
            taxonomy_id = '03001627'
        elif self.category == 'table':
            taxonomy_id = '04379243'
        else:
            raise NotImplementedError()
        data_path = self.DATASET_PATH % (self.subset, self.category)
        for file_name in os.listdir(data_path):
            file_list['taxonomy_id'].append(taxonomy_id)
            file_list['model_id'].append(file_name[:-8])
            file_list['partial_path'].append(data_path + file_name)

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list['partial_path']), logger='SCANNETDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = {}
        data = {}

        sample['taxonomy_id'] = self.file_list['taxonomy_id'][idx]
        sample['model_id'] = self.file_list['model_id'][idx]

        data['partial'] = IO.get(self.file_list['partial_path'][idx]).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'])

    def __len__(self):
        return len(self.file_list['partial_path'])