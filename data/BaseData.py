import copy
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler


class BaseData:
    def __init__(self, args):
        self.args = args
        self.label_list = self._read_labels()
        self.id2label, self.label2id = [], {}
        self.label2task_id = {}
        self.train_data, self.val_data, self.test_data = None, None, None

    def _read_labels(self):
        """
        :return: only return the label name, in order to set label index from 0 more conveniently.
        """
        id2label = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'id2label.json')))
        return id2label

    def read_and_preprocess(self, **kwargs):
        raise NotImplementedError

    def add_labels(self, cur_labels, task_id):
        for c in cur_labels:
            if c not in self.id2label:
                self.id2label.append(c)
                self.label2id[c] = len(self.label2id)
                self.label2task_id[self.label2id[c]] = task_id

    def filter(self, labels, split='train'):
        if not isinstance(labels, list):
            labels = [labels]
        split = split.lower()
        res = []
        for label in labels:
            if split == 'train':
                if self.args.debug:
                    res += copy.deepcopy(self.train_data[label])[:10]
                else:
                    res += copy.deepcopy(self.train_data[label])
            elif split in ['dev', 'val']:
                if self.args.debug:
                    res += copy.deepcopy(self.val_data[label])[:10]
                else:
                    res += copy.deepcopy(self.val_data[label])
            elif split == 'test':
                if self.args.debug:
                    res += copy.deepcopy(self.test_data[label])[:10]
                else:
                    res += copy.deepcopy(self.test_data[label])
        for idx in range(len(res)):
            res[idx]["labels"] = self.label2id[res[idx]["labels"]]
        return res

    def filter_and_add_desciption(self, labels, descriptions):
        if not isinstance(labels, list):
            labels = [labels]
        # labels_label2id = [self.label2id[label_] for label_ in labels]
        print(labels)
        res = []
        for label in labels:
            pools = {}
            if label in descriptions.keys():
                pools = descriptions[label]
            sub_res = []
            for anchor in self.train_data[label]:
                # print(label)
                # if idxxx:
                #     print("-------")
                # for key, value in anchor.items():
                #     print(f"  {key}: {value}")
                    
                cur_label = anchor["labels"]
                if cur_label in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spouse',
                                        'per:alternate_names', 'per:other_family']:
                    continue
                
                ins = {
                    'input_ids': anchor['input_ids'],  # default: add marker to the head entity and tail entity
                    'subject_marker_st': anchor['subject_marker_st'],
                    'object_marker_st': anchor['object_marker_st'],
                    'labels': anchor['labels'],
                    'input_ids_without_marker': anchor['input_ids_without_marker'],
                    'subject_st': anchor['subject_st'],
                    'subject_ed': anchor['subject_ed'],
                    'object_st': anchor['object_st'],
                    'object_ed': anchor['object_ed'],
                }
                
                for idx, pool in enumerate(pools):
                    ins.update({
                        f'description_ids_{idx}': pool
                    })
                sub_res.append(ins)
            res += sub_res
        for idx in range(len(res)):
            res[idx]["labels"] = self.label2id[res[idx]["labels"]]
        return res

    def filter_and_add_desciption_and_old_description(self, labels, descriptions, seen_labels, old_descriptions):
        if not isinstance(labels, list):
            labels = [labels]
        print(labels)
        res = []
        lenght_seen_labels = len(seen_labels)
        count_negative_label = 0
        for label in labels:
            pools = {}
            if label in descriptions.keys():
                pools = descriptions[label]
            sub_res = []
            for anchor in self.train_data[label]:
                # cur_label = anchor["labels"]
                # if cur_label in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spouse',
                #                         'per:alternate_names', 'per:other_family']:
                #     continue
                
                ins = {
                    'input_ids': anchor['input_ids'],  # default: add marker to the head entity and tail entity
                    'subject_marker_st': anchor['subject_marker_st'],
                    'object_marker_st': anchor['object_marker_st'],
                    'labels': anchor['labels'],
                    'input_ids_without_marker': anchor['input_ids_without_marker'],
                    # 'mask_marker': anchor['mask_marker'],
                    'subject_st': anchor['subject_st'],
                    'subject_ed': anchor['subject_ed'],
                    'object_st': anchor['object_st'],
                    'object_ed': anchor['object_ed'],
                }
                
                for idx, pool in enumerate(pools):
                    ins.update({
                        f'description_ids_{idx}': pool
                    })
                if lenght_seen_labels != 0:
                    old_labels = seen_labels[count_negative_label%lenght_seen_labels]
                    # ins.update({
                    #         'old_labels': self.label2id[old_labels]
                    #     })
                    old_pools = {}
                    if old_labels in old_descriptions.keys():
                        old_pools = old_descriptions[old_labels]
                        
                    for idx, pool in enumerate(old_pools):
                        ins.update({
                            f'old_description_ids_{idx}': pool
                        })
                
                count_negative_label += 1
                sub_res.append(ins)
            res += sub_res
        for idx in range(len(res)):
            res[idx]["labels"] = self.label2id[res[idx]["labels"]]
        return res


class BaseDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, dict):
            res = []
            for key in data.keys():
                res += data[key]
            data = res
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # cur_data = self.data[idx]
        # cur_data["idx"] = idx
        # mask_head = True if random.random() > 0.5 else False
        # input_ids, attention_mask, subject_start_pos, object_start_pos = mask_entity(cur_data["input_ids"], mask_head)
        # augment_data = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "subject_start_pos": subject_start_pos,
        #     "object_start_pos": object_start_pos,
        #     "labels": cur_data["labels"],
        #     "idx": idx
        # }
        # return [cur_data, augment_data]
        return self.data[idx]

