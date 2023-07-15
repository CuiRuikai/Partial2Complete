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
class EPN3DComplete(data.Dataset):
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
        return data_transforms.Compose([{
            'callback': 'RandomSamplePoints',  # random permutate points
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


    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = {
            'taxonomy_id': [],
            'model_id': [],
            'sample_path': [],
        }


        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='EPN3DNDATASET')
            category_name = dc['taxonomy_name']
            partial_samples = dc[subset]['partial']
            complete_samples = dc[subset]['complete']

            for (partial_file, complete_file) in zip(partial_samples, complete_samples):
                file_list['taxonomy_id'].append(dc['taxonomy_id'])
                file_list['model_id'].append(complete_file)
                file_list['sample_path'].append(self.partial_points_path % (category_name, partial_file))
                # if self.complete_points_path % (category_name, complete_file) not in file_list['sample_path']:
                #     file_list['taxonomy_id'].append(dc['taxonomy_id'])
                #     file_list['model_id'].append(complete_file)
                #     file_list['sample_path'].append(self.complete_points_path % (category_name, complete_file))
                file_list['taxonomy_id'].append(dc['taxonomy_id'])
                file_list['model_id'].append(complete_file)
                file_list['sample_path'].append(self.complete_points_path % (category_name, complete_file))

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list['sample_path']), logger='EPN3DDATASET')
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