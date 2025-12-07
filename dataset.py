#  -*- coding: utf-8 -*-
'''
author: xuechao.wang@ugent.be
'''
import numpy as np
from torch.utils.data import Dataset



class MyDataset_test(Dataset):
    def __init__(self, dataset, name, transform=None):
        self.name = name
        self.transform = transform
        name2cls = {'KT':0, 'PD':1}
        test_files_list = self.read_list_file(dataset, name2cls)
        self.files_list = test_files_list
        self.caches = {}

    def read_list_file(self, files, name2cls):
        files_list = []
        for patch_pc in files:
            name_type = self.name
            cur = patch_pc
            files_list.append([cur, name2cls[name_type]])
        return files_list

    def __getitem__(self, index):
        if index in self.caches:
            return self.caches[index]
        file, label = self.files_list[index]
        data = np.float32(file)
        self.caches[index] = [data, label]
        return data, label

    def __len__(self):
        return len(self.files_list)