import os
import sys
import json
import glob
import numpy as np
import random
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial import KDTree

from dataset.data_utils import *
from dataset.buildin import DATASET_META

############################################################################################
############################################################################################
############################################################################################

def synthetic_map_data_processor(anno_data, thre=1):    
    out_anno_data = []
    if dataset_name == 'SynthMap_train':
        for anno in anno_data:
            if len(anno['groups']) > thre:
                out_anno_data.append(anno)
    elif dataset_name == 'SynthMap_test':
        out_anno_data = anno_data[1000:1200] # this is just for demoing
    return out_anno_data

    
class LinkingDataset():
    def __init__(self, dataset_name, anno_path, img_dir, return_labels=False):
        self.dataset_name = dataset_name
        self.anno_path = anno_path
        self.img_dir = img_dir
        self.return_labels = return_labels

        if dataset_name in ["MapText_train", "MapText_test", "MapText_val", "SynthMap_train"]:
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)    
            
            if 'SynthMap' in dataset_name:    
                anno_data = synthetic_map_data_processor(anno_data)
                
            self.anno_data = anno_data            
            self.generate_group_labels()
            self.length = len(self.anno_data)
            print(f"{dataset_name} contains {self.length} samples.")
        else:
            self.anno_files = glob.glob(os.path.join(anno_path, '*.json'))
            self.length = len(self.anno_files)
            print(f"{dataset_name} contains {len(self.anno_files)} samples.")

        self.unused_indices = list(range(self.length))
        
    def reset(self):
        self.unused_indices = list(range(self.length))

    def generate_group_labels(self):
        group_id_list = [i for i in range(1, 10000) if '0' not in str(i)]
        for anno in self.anno_data:
            for i, group in enumerate(anno['groups']):
                group_id = group_id_list[i]
                seq_id = 1
                for item in group:
                    if item.get('illegible', False) or item.get('truncated', False):                        
                        continue    
                    if self.return_labels: 
                        item['label'] = int(f"{group_id}000{seq_id}")  
                    else:
                        item['label'] = 0
                    seq_id += 1

    def get_length(self):
        return self.length

    def get_item(self, idx, is_random=False):
        if is_random:
            idx = random.choice(self.unused_indices)
            self.unused_indices.remove(idx)

        if self.dataset_name in ["MapText_train", "MapText_test", "MapText_val", "SynthMap_train"]:
            item = self.anno_data[idx]
            image_name = item['image'].split('/')[-1]
            image_path = os.path.join(self.img_dir, image_name)
            anno = item['groups']
        else:
            anno_file = self.anno_files[idx]
            with open(anno_file, 'r') as f:
                anno = json.load(f)
                anno = anno['annotations']

            image_name = os.path.basename(anno_file).replace('json', 'jpg')
            image_path = os.path.join(self.img_dir, image_name)
            
        image = Image.open(image_path).convert("RGB")
        return image, anno, image_name

############################################################################################
############################################################################################
############################################################################################
def mask_coordinates(sequence, mask_token_id, mask_ratio=0.5):
    """
    Mask coordinates in a sequence following two strategies:
    1. Mask both (x, y) of a point.
    2. Mask either x or y independently.
    """
    if mask_ratio <= 0.01:
        return sequence, [], sequence
        
    gt_sequence = sequence.copy()  # Ground truth values
    seq_len = len(sequence)  # Sequence length (should be <= 32)
    num_masked = max(1, int(mask_ratio * (seq_len // 2)))  # Number of tokens to mask
    
    indices = np.array([i for i in range(seq_len) if i % 2 == 0])
    mask_indices = np.random.choice(indices, num_masked, replace=False)
    
    masked_positions = []
    for idx in mask_indices:
        if False: # np.random.rand() < 0.5:
            # Strategy 1: Mask both x and y
            masked_positions.extend([idx, idx + 1])
            sequence[idx] = mask_token_id
            sequence[idx + 1] = mask_token_id
        else:
            # Strategy 2: Mask either x or y
            if np.random.rand() < 0.5:
                masked_positions.append(idx)  # Mask x
                sequence[idx] = mask_token_id
            else:
                masked_positions.append(idx + 1)  # Mask y
                sequence[idx + 1] = mask_token_id

    return sequence, masked_positions, gt_sequence


############################################################################################
############################################################################################
############################################################################################
def process_one_sample(
    anno,
    image,
    dataset_name,
    args,
    num_samples_per_image
):
    
    image_width, image_height = image.size
    padding_token_id = args.padding_token_id
    mask_token_id = args.mask_token_id
    max_len = args.max_len
    
    # prepare raw data
    sequence_list, masked_positions_list, gt_sequence_list = [], [], []
    polygon_list, center_x_list, center_y_list, char_height_list, angle_list = [], [], [], [], []
    word_list = []
    
    def helper(polygon):
        poly = compute_polygon(polygon, image_width, image_height, sampling=True)
        try:
            poly_flatten = poly.copy()
            poly_flatten = poly_flatten.reshape(-1)
            sequence, masked_positions, gt = mask_coordinates(poly_flatten, 
                                                              mask_token_id, 
                                                              mask_ratio=args.mask_ratio)
        except Exception as e:
            print(e)
            return

        polygon_list.append(poly)
        center = get_center(poly)
        center_x_list.append(center[0])
        center_y_list.append(center[1])
        angle = get_angle(poly) / 180 # 0 to 180
        angle_list.append(angle)
        char_height = get_char_height(poly)
        char_height_list.append(char_height)

        pid = padding_token_id
        sequence_padded = pad_sequence(sequence, max_len, pad_value=pid)
        masked_positions_padded = pad_sequence(masked_positions, max_len, pad_value=pid)
        gt_padded = pad_sequence(gt, max_len, pad_value=pid)
        sequence_list.append(sequence_padded)
        masked_positions_list.append(masked_positions_padded)
        gt_sequence_list.append(gt_padded)
        
        
        
    if dataset_name in ["MapText_train", "MapText_test", "MapText_val", "SynthMap_train"]:
        for group in anno:
            for item in group:
                if item.get('label') is None:
                    continue
                polygon = np.array(item['vertices']).reshape(-1, 2).astype(float)
                helper(polygon)
                word_list.append(item['text'])
    else:
        for item in anno:
            polygon = np.array(item['pts']).reshape(-1, 2).astype(float)
            helper(polygon)
            word_list.append(item['text'])
            
    #######################################################################
    #######################################################################
    indices = list(range(len(sequence_list)))
    random.shuffle(indices)

    sequence_list = [sequence_list[i] for i in indices]
    masked_positions_list = [masked_positions_list[i] for i in indices]
    gt_sequence_list = [gt_sequence_list[i] for i in indices]
    center_x_list = [center_x_list[i] for i in indices]
    center_y_list = [center_y_list[i] for i in indices]
    char_height_list = [char_height_list[i] for i in indices]
    angle_list = [angle_list[i] for i in indices]
    polygon_list = [polygon_list[i] for i in indices]    
    word_list = [word_list[i] for i in indices]    
    
    n = num_samples_per_image
    pid = padding_token_id
    if len(sequence_list) >= n:
        sequence_list = sequence_list[:n]
        masked_positions_list = masked_positions_list[:n]
        gt_sequence_list = gt_sequence_list[:n]
        center_x_list = center_x_list[:n]
        center_y_list = center_y_list[:n]
        char_height_list = char_height_list[:n]
        angle_list = angle_list[:n]     
        polygon_list = polygon_list[:n]
        word_list = polygon_list[:n]
    else:
        sequence_list += [[pid] * max_len] * (n - len(sequence_list))
        masked_positions_list += [[pid] * max_len] * (n - len(masked_positions_list))
        gt_sequence_list += [[pid] * max_len] * (n - len(gt_sequence_list))
        center_x_list += [pid] * (n - len(center_x_list))
        center_y_list += [pid] * (n - len(center_y_list))
        char_height_list += [pid] * (n - len(char_height_list))
        angle_list += [pid] * (n - len(angle_list))

    sequences = np.stack(sequence_list, axis=0)
    masked_positions = np.stack(masked_positions_list, axis=0)
    gt_sequences = np.stack(gt_sequence_list, axis=0)
    center_x = np.array(center_x_list)
    center_y = np.array(center_y_list)
    char_heights = np.array(char_height_list)
    angles = np.array(angle_list)

    sequences = torch.tensor(sequences, dtype=torch.float)
    masked_positions = torch.tensor(masked_positions, dtype=torch.long)
    gt_sequences = torch.tensor(gt_sequences, dtype=torch.float)
    center_x = torch.tensor(center_x, dtype=torch.float)
    center_y = torch.tensor(center_y, dtype=torch.float)
    char_heights = torch.tensor(char_heights, dtype=torch.float)
    angles = torch.tensor(angles, dtype=torch.float)
            
    #######################################################################
    #######################################################################
    m = len(polygon_list)
    dist_matrix = np.zeros((m, m))
    kd_trees = [KDTree(poly) for poly in polygon_list]
    for i in range(m):
        for j in range(i + 1, m):
            poly1 = polygon_list[i]
            poly2 = polygon_list[j]
            tree1 = kd_trees[i]
            tree2 = kd_trees[j]
            min_dist_1_to_2, _ = tree1.query(poly2)
            min_dist_2_to_1, _ = tree2.query(poly1)
            min_dist = min(min_dist_1_to_2.min(), min_dist_2_to_1.min())
            dist_matrix[i, j] = min_dist
            dist_matrix[j, i] = min_dist

    np.fill_diagonal(dist_matrix, np.inf)
    closest_indices = np.argmin(dist_matrix, axis=1)
    padded_indices = np.pad(closest_indices, (0, num_samples_per_image - m), constant_values=padding_token_id)
    padded_indices = torch.tensor(padded_indices, dtype=torch.long)
    
    return {'sequences': sequences, # 200 x 32
            'attn_mask': masked_positions,
            'gt_sequences': gt_sequences,
            'center_x': center_x,
            'center_y': center_y,
            'char_heights': char_heights,
            'angles': angles,
            'dist_matrix': padded_indices,
            'm': torch.tensor([m]),
           }

############################################################################################
############################################################################################
############################################################################################
class PolyTrainDataset(Dataset):
    def __init__(self, args):
        assert len(args.train_datasets) == len(args.train_data_probabilities)
        self.args = args
        self.probabilities = args.train_data_probabilities
        self.dataset_names = args.train_datasets

        self.datasets = {}
        for i, name in enumerate(self.dataset_names):
            print(f"Load training data: {name}")
            assert os.path.exists(DATASET_META[name]['anno_path']), f"Annotation data {name} must exist."
            assert os.path.exists(DATASET_META[name]['img_dir']), f"Image data {name} must exist."
            dataset = LinkingDataset(dataset_name=name,
                                     anno_path=DATASET_META[name]['anno_path'], 
                                     img_dir=DATASET_META[name]['img_dir'],
                                     return_labels=True)
            self.datasets[name] = dataset
    
    def __len__(self):
        return self.args.num_samples_per_epoch 

    def reset(self):
        for _, dataset in self.datasets.items():
            dataset.reset()

    def __getitem__(self, idx):
        dataset_name = random.choices(self.dataset_names, self.probabilities)[0]
        image, anno, image_name = self.datasets[dataset_name].get_item(idx, is_random=True)
        if len(anno) == 0:
            dataset_name = random.choices(self.dataset_names, self.probabilities)[0]
            image, anno, image_name = self.datasets[dataset_name].get_item(idx, is_random=True)
        output = process_one_sample(anno, image, dataset_name, self.args, num_samples_per_image=100)
        return output


class PolyTestDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.dataset_name = args.val_dataset
        self.dataset = LinkingDataset(dataset_name=self.dataset_name,
                                      anno_path=DATASET_META[self.dataset_name]['anno_path'], 
                                      img_dir=DATASET_META[self.dataset_name]['img_dir'],
                                      return_labels=True)
    def __len__(self):
        return self.dataset.get_length()
        
    def __getitem__(self, idx):
        image, anno, image_name = self.dataset.get_item(idx)
        output = process_one_sample(anno, image, self.dataset_name, self.args, num_samples_per_image=500)
        return output



