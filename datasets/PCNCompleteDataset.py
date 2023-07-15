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
class PCNComplete(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.categories = config.categories

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in self.categories]

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['sample_path']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['sample_path']
            },{
                'callback': 'ToTensor',
                'objects': ['sample_path']
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
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = {
            'taxonomy_id': [],
            'model_id': [],
            'sample_path': [],
        }
        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNCompleteDATASET')
            samples = dc[subset]

            for s in samples:
                # for i in range(n_renderings):
                    # file_list['taxonomy_id'].append(dc['taxonomy_id'])
                    # file_list['model_id'].append(s)
                    # file_list['sample_path'].append(self.partial_points_path % (subset, dc['taxonomy_id'], s, i))

                file_list['taxonomy_id'].append(dc['taxonomy_id'])
                file_list['model_id'].append(s)
                file_list['sample_path'].append(self.complete_points_path % (subset, dc['taxonomy_id'], s))

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNCompleteDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = {}
        data = {}

        sample['taxonomy_id'] = self.file_list['taxonomy_id'][idx]
        sample['model_id'] = self.file_list['model_id'][idx]

        data['sample_path'] = IO.get(self.file_list['sample_path'][idx]).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data['sample_path']

    def __len__(self):
        return len(self.file_list['sample_path'])