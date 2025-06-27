import os
import sys
import json
import numpy as np
import random
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.data_utils import *
from dataset.buildin import DATASET_META
from scipy.spatial import KDTree


def synthetic_map_data_processor(anno_data, thre=3):    
    out_anno_data = []
    if dataset_name == 'SynthMap_train':
        for anno in anno_data:
            if len(anno['groups']) > thre:
                out_anno_data.append(anno)
    elif dataset_name == 'SynthMap_test':
        out_anno_data = anno_data[1000:1200] # this is just for demoing
    return out_anno_data

def hiertext_data_processor(anno_data):
    out_anno_data = []
    for anno in anno_data:
        if anno['image'] not in ['62aec264ad248f1e', 'd5e1b07d3bf588d7', 'dc30c1e1f7bd87d1']:
            anno['image'] += '.jpg'
            out_anno_data.append(anno)
    return out_anno_data
            
    
class LinkingDataset():
    def __init__(self, dataset_name, anno_path, img_dir, return_labels=False):
        self.dataset_name = dataset_name
        self.anno_path = anno_path
        self.img_dir = img_dir
        self.return_labels = return_labels
        with open(anno_path, 'r') as f:
            anno_data = json.load(f)

        if "SynthMap" in dataset_name:
            anno_data = synthetic_map_data_processor(anno_data)
        if "HierText" in dataset_name:
            anno_data = hiertext_data_processor(anno_data)
            
        self.anno_data = anno_data
        self.generate_group_labels()
        print(f"Dataset `{dataset_name}` contains {len(self.anno_data)} samples.")
        self.unused_indices = list(range(len(self.anno_data)))
        
    def reset(self):
        self.unused_indices = list(range(len(self.anno_data)))

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
        return len(self.anno_data)

    def get_item(self, idx, is_random=False):
        if is_random:
            idx = random.choice(self.unused_indices)
            self.unused_indices.remove(idx)

        item = self.anno_data[idx]
        
        if '/' in item['image']:
            image_name = os.path.basename(item['image'])
        elif '.jpg' not in item['image']:
            image_name = item['image'] + '.jpg'
        else:
            image_name = item['image']
            
        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        anno = item['groups']
        return image, anno, image_name
    
##################################################################################
##################################################################################
##################################################################################
    
def process_one_sample(
    tokenizer,
    image_processor,
    anno,
    image,
    is_shuffle=True,
    args=None
):
    ###
    ### image processor
    image_width, image_height = image.size
    image_features = image_processor(image, return_tensors="pt")
    pixel_values = image_features['pixel_values'][0]

    ###
    ### tokenizer
    max_length = getattr(args, "token_padding_max_length", 1280)
    ##################################################################################
    words, bboxes, labels, polygons, ori_polygons = [], [], [], [], []
    for group in anno:
        for item in group:
            if item.get('label') is None: continue;
            poly = np.array(item['vertices']).reshape(-1, 2).astype(float)
            bbox = create_bounding_box(poly)
            if bbox[0] == bbox[2] or bbox[1] == bbox[3]: continue;
                
            bbox_scaled = scale_bounding_box(bbox, 1000/image_width, 1000/image_height)
            bboxes.append(bbox_scaled)
            
            ori_polygons.append(item['vertices'])
            padded_poly = compute_polygon(poly, image_width, image_height, sampling=False)
            padded_poly = pad_sequence(padded_poly, max_length=args.max_len, 
                                       pad_value=args.padding_token_id)
            
            if args.poly_only or len(item["text"]) == 0:
                text = 'NaN'
            else:
                text = item["text"].replace(' ', '')                

            if args.text_only:
                padded_poly = np.zeros_like(padded_poly)         

            poly_tensor = torch.tensor(padded_poly, dtype=torch.float)
            polygons.append(poly_tensor)
            words.append(text)
            labels.append(item['label'])
            
    if len(words) == 0 or len(polygons) == 0:
        return None, None
    ##################################################################################
    
    ###
    ### shuffle
    if is_shuffle:
        indices = [i for i in range(len(labels))]
        random.shuffle(indices)
    else:
        pts = [(poly[0], poly[1]) for poly in polygons]
        indices = sorted(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))

    words = [words[i] for i in indices]
    labels = [labels[i] for i in indices]
    polygons = [polygons[i] for i in indices]
    bboxes = [bboxes[i] for i in indices]
    ori_polygons = [ori_polygons[i] for i in indices]    
    
    ##################################################################################
    ori_data_dict = {
            "image": image,
            "words": words,
            "bboxes": bboxes,
            "polygons": ori_polygons,
            "labels": labels
    }    
    encoded_inputs = tokenizer(
        text=words,
        boxes=bboxes,
        word_labels=labels,
        return_special_tokens_mask=False,
        padding='max_length',
        max_length=max_length,
        truncation=True,
    )
    ##################################################################################
    # input_ids is the token ids starting with CLS token, typically longer than labels
    input_ids = torch.tensor(encoded_inputs['input_ids'], dtype=torch.long)
    ###################################################################################
    # bbox is the text bbox, duplicated for tokens belonging to the same word
    bbox = torch.tensor(encoded_inputs['bbox'], dtype=torch.long)
    bbox = torch.clip(bbox, 0, 1000)
    ###################################################################################
    # labels are at the first token location
    labels = torch.tensor(encoded_inputs['labels'], dtype=torch.long)
    attention_mask = torch.tensor(encoded_inputs['attention_mask'], dtype=torch.long)
    ###################################################################################
    # encoded_inputs.word_ids(): [None, 0, 0, 1, 1, 2, 3, 3, 3, ..., None, None, ...]
    # get first token locations; word_id maps the first token to word indices
    word_ids, first_token_indices = [], [] 
    new_polygons = torch.zeros((input_ids.shape[0], 32), dtype=torch.float)
    
    prev_word_id = -1
    for i, word_id in enumerate(encoded_inputs.word_ids()):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            word_ids.append(word_id)
            first_token_indices.append(i)
            new_polygons[i] = polygons[word_id]
            prev_word_id = word_id
        else:
            new_polygons[i] = polygons[prev_word_id]
        
    first_token_indices = torch.tensor(first_token_indices, dtype=torch.long)
    first_token_indices = F.pad(first_token_indices, (0, max_length-first_token_indices.shape[0]), value=-999)
    ori_data_dict['word_ids'] = word_ids
    output = {
        'input_ids': input_ids,
        'first_token_indices': first_token_indices,
        'bbox': bbox,
        'labels': labels,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'polygons': new_polygons
    }
    return output, ori_data_dict


class LinkingTrainDataset(Dataset):
    def __init__(self, tokenizer, image_processor, args):
        assert len(args.train_datasets) == len(args.train_data_probabilities)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args
        self.probabilities = args.train_data_probabilities
        self.shuffle = args.train_data_shuffle
        self.dataset_names = args.train_datasets

        self.datasets = {}
        for i, name in enumerate(self.dataset_names):
            print(f"Load training data: {name}")
            dataset = LinkingDataset(dataset_name=name,
                                     anno_path=DATASET_META[name]['anno_path'], 
                                     img_dir=DATASET_META[name]['img_dir'],
                                     return_labels=True)
            self.datasets[name] = dataset
    
    def __len__(self):
        if 'MapText' in self.dataset_names[0]:
            return 200
        else:
            return 500 

    def reset(self):
        for _, dataset in self.datasets.items():
            dataset.reset()

    def __getitem__(self, idx):
        dataset_name = random.choices(self.dataset_names, self.probabilities)[0]
        image, anno, image_name = self.datasets[dataset_name].get_item(idx, is_random=True)
        output, _ = process_one_sample(self.tokenizer, self.image_processor, 
                                       anno, image, 
                                       self.shuffle, self.args)
        return output

    
class LinkingTestDataset(Dataset):
    def __init__(self, tokenizer, image_processor, args, mode, return_ori=False):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args
        self.mode = mode
        if self.mode == 'val':
            self.dataset_name = args.val_dataset
            self.shuffle = args.val_data_shuffle
            self.return_labels = True
        else:
            self.dataset_name = args.test_dataset
            self.shuffle = args.test_data_shuffle
            self.return_labels = True

        if self.mode == 'val':
            self.dataset = LinkingDataset(dataset_name=self.dataset_name,
                                          anno_path=DATASET_META[self.dataset_name]['anno_path'], 
                                          img_dir=DATASET_META[self.dataset_name]['img_dir'],
                                          return_labels=self.return_labels)
        else:
            print(f"Load annotations from {args.anno_path}")
            self.dataset = LinkingDataset(dataset_name='test',
                                          anno_path=args.anno_path,
                                          img_dir=args.img_dir,
                                          return_labels=self.return_labels)
        self.return_ori = return_ori
    
    def __len__(self):
        return self.dataset.get_length()
        
    def __getitem__(self, idx):
        image, anno, image_name = self.dataset.get_item(idx)
        output, ori_data_dict = process_one_sample(self.tokenizer, self.image_processor, 
                                                   anno, image, 
                                                   self.shuffle, self.args)
        if output is None: return None;
        if self.return_ori:
            output['image_name'] = image_name
            output['ori_words'] = ori_data_dict["words"]
            output['ori_polygons'] = ori_data_dict["polygons"]
            output['ori_word_ids'] = ori_data_dict["word_ids"]
        return output






