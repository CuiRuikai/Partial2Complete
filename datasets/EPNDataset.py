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
class EPN3D(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):

        self.npoints = config.N_POINTS
        self.category_file = config.CATEGORY_FILE_PATH
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.class_choice = config.class_choice
        self.subset = config.subset

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in self.class_choice]

        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',  # random permutate points
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['partial', 'complete']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'complete']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'complete']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'complete']
            }])

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = {
            'taxonomy_id': [],
            'model_id': [],
            'partial_path': [],
            'gt_path': []
        }


        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='EPN3DNDATASET')
            category_name = dc['taxonomy_name']
            partial_samples = dc[subset]['partial']
            complete_samples = dc[subset]['complete']

            for (partial_file, complete_file) in zip(partial_samples, complete_samples):
                file_list['taxonomy_id'].append(dc['taxonomy_id'])
                file_list['model_id'].append(complete_file)
                file_list['partial_path'].append(self.partial_points_path % (category_name, partial_file))
                file_list['gt_path'].append(self.complete_points_path % (category_name, complete_file))

        shuffled_gt = file_list['gt_path'].copy()
        # random.shuffle(shuffled_gt)
        file_list['shuffled_gt_path'] = shuffled_gt


        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list['partial_path']), logger='EPN3DDATASET')
        return file_list

    def shuffle_gt(self):
        random.shuffle(self.file_list['shuffled_gt_path'])

    def __getitem__(self, idx):
        sample = {}
        data = {}

        sample['taxonomy_id'] = self.file_list['taxonomy_id'][idx]
        sample['model_id'] = self.file_list['model_id'][idx]

        data['partial'] = IO.get(self.file_list['partial_path'][idx]).astype(np.float32)
        if self.subset == 'train':
            data['complete'] = IO.get(self.file_list['shuffled_gt_path'][idx]).astype(np.float32)
        else:  # test/val
            data['complete'] = IO.get(self.file_list['gt_path'][idx]).astype(np.float32)


        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['complete'])

    def __len__(self):
        return len(self.file_list['partial_path'])