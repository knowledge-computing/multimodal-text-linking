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
from multi_modal_tokenizers import DalleTokenizer

from dataset.data_utils import *
from dataset.buildin import DATASET_META


############################################################################################
############################################################################################
############################################################################################
def apply_mlm_mask(input_ids, tokenizer, mask_prob=0.15):
    """
    Applies MLM masking to text tokens.
    where 1 indicates "masked".
    """
    labels = input_ids.clone()  # Clone original tokens
    mask = torch.rand(input_ids.shape) < mask_prob  # Select 15% of tokens randomly

    # Don't mask special tokens ([CLS], [SEP], etc.)
    mask[input_ids == tokenizer.pad_token_id] = False
    mask[input_ids == tokenizer.cls_token_id] = False
    mask[input_ids == tokenizer.sep_token_id] = False

    # Replace with [MASK] token
    input_ids[mask] = tokenizer.mask_token_id # 50264
    return input_ids, mask, labels

class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return torch.tensor(mask.flatten()) == 1
        
def expand_mask(mask_14x14, patch_size=16):
    """Expand a 14x14 mask to match the 224x224 image resolution."""
    mask_224x224 = mask_14x14.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    return mask_224x224  # Shape: (224, 224)
        
############################################################################################
############################################################################################
############################################################################################

def process_one_sample(
    text_tokenizer,
    image_processor,
    image_mask_generator,
    anno,
    image,
    args=None
):
    ###
    ### image processor
    image_width, image_height = image.size
    image_features = image_processor(image, return_tensors="pt")
    pixel_values = image_features['pixel_values'][0] # 3x224x224

    mim_mask = image_mask_generator()
    mask_224x224 = expand_mask(mim_mask.reshape(14,14))    
    masked_pixel_values = pixel_values * (mask_224x224.unsqueeze(0) != True)  # masked image

    ###
    ### text tokenizer
    words, bboxes, image_id_labels, gt_masks, polygons = [], [], [], [], []
    for item in anno:
        words.append(item['text'])
        
        poly = np.array(item['pts']).reshape(-1, 2).astype(float)
        padded_poly = compute_polygon(poly, image_width, image_height, sampling=True)
        padded_poly = pad_sequence(padded_poly, args.max_len, 
                                   pad_value=args.padding_token_id)
        polygons.append(torch.tensor(padded_poly, dtype=torch.float))

        bbox = create_bounding_box(poly)
        bbox_scaled = scale_bounding_box(bbox, 1000/image_width, 1000/image_height)
        bboxes.append(bbox_scaled)

        center_x = (bbox[0] + bbox[2]) / 2. / image_width
        center_y = (bbox[1] + bbox[3]) / 2. / image_height
        image_id = int(center_x * 14) + int(center_y * 14) * 14
        image_id_labels.append(image_id)
            
    ##################################################################################
    indices = [i for i in range(len(words))]
    random.shuffle(indices)
    words = [words[i] for i in indices]
    bboxes = [bboxes[i] for i in indices]
    image_id_labels = [image_id_labels[i] for i in indices]
    polygons = [polygons[i] for i in indices]
    
    encoded_inputs = text_tokenizer(
        text=words,
        boxes=bboxes,
        word_labels=image_id_labels,
        return_special_tokens_mask=False,
        padding='max_length',
        max_length=getattr(args, "token_padding_max_length", 1280),
        truncation=True,
    )
    ##################################################################################
    # input_ids is the token ids starting with CLS token, typically longer than labels
    input_ids = torch.tensor(encoded_inputs['input_ids'], dtype=torch.long)
    input_ids, mlm_mask, mlm_labels = apply_mlm_mask(input_ids, text_tokenizer, mask_prob=0.3)
    bbox = torch.tensor(encoded_inputs['bbox'], dtype=torch.long)
    bbox = torch.clip(bbox, 0, 1000)
    attention_mask = torch.tensor(encoded_inputs['attention_mask'], dtype=torch.long)
    ##################################################################################
    # encoded_inputs.word_ids(): [None, 0, 0, 1, 1, 2, 3, 3, 3, ..., None, None, ...]
    labels = [l for l in encoded_inputs['labels'] if l != -100]
    if len(words) != len(labels): # hand craft
        return None

    image_id_labels = torch.full_like(input_ids, args.padding_token_id, dtype=torch.long)
    wpa_labels = torch.full_like(input_ids, args.padding_token_id, dtype=torch.float)
    new_ploygons = torch.full((input_ids.shape[0], 32), args.padding_token_id, dtype=torch.float)
    
    prev_word_id = -1
    for i, word_id in enumerate(encoded_inputs.word_ids()):
        if word_id is None: 
            continue
            
        if word_id != prev_word_id:
            image_index = labels[word_id]
            image_id_labels[i] = image_index
            new_ploygons[i] = polygons[word_id]
            prev_word_id = word_id
        else:  
            image_index = labels[prev_word_id]
            image_id_labels[i] = image_index
            new_ploygons[i] = polygons[prev_word_id]
            
        if mim_mask[image_index] == 0 and mlm_mask[i] == 0: 
            wpa_labels[i] = 1
        if mim_mask[image_index] == 1 and mlm_mask[i] == 0: 
            wpa_labels[i] = 0

    ###################################################################################
    output = {
        'input_ids': input_ids,
        'mlm_mask': mlm_mask,
        'mlm_labels': mlm_labels,
        'bbox': bbox,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'masked_pixel_values': masked_pixel_values,
        'mim_mask': mim_mask,
        'image_id_labels': image_id_labels,
        'wpa_labels': wpa_labels,
        'polygons': new_ploygons,
    }
    return output


class PretrainDataset(Dataset):
    def __init__(self, text_tokenizer, image_processor, args, mode='train'):
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.image_mask_generator = MaskGenerator(input_size=224, 
                                                  mask_patch_size=16, 
                                                  model_patch_size=16,
                                                  mask_ratio=0.4)
        self.args = args
        self.mode = mode
        
        self.samples = []
        # self.paths = {DATASET_META['Rumsey1_train']['anno_path']: DATASET_META['Rumsey1_train']['img_dir'],
        #               DATASET_META['Rumsey2_train']['anno_path']: DATASET_META['Rumsey2_train']['img_dir'],
        #               DATASET_META['MapText_json_train']['anno_path']: DATASET_META['MapText_json_train']['img_dir']}

        self.paths = {DATASET_META['hiertext_json_train']['anno_path']: DATASET_META['hiertext_json_train']['img_dir'],
                      DATASET_META['icdar15_json_train']['anno_path']: DATASET_META['icdar15_json_train']['img_dir'],
                      DATASET_META['icdar15_json_test']['anno_path']: DATASET_META['icdar15_json_test']['img_dir'],
                      DATASET_META['mlt_json_train']['anno_path']: DATASET_META['mlt_json_train']['img_dir'],
                      DATASET_META['textocr_json_train']['anno_path']: DATASET_META['textocr_json_train']['img_dir'],
                      DATASET_META['totaltext_json_train']['anno_path']: DATASET_META['totaltext_json_train']['img_dir'],
                      DATASET_META['hiertext_json_train']['anno_path']: DATASET_META['hiertext_json_train']['img_dir'],
                      DATASET_META['hiertext_json_val']['anno_path']: DATASET_META['hiertext_json_val']['img_dir'],                      
                      DATASET_META['hiertext_json_103624_test']['anno_path']: DATASET_META['hiertext_json_103624_test']['img_dir'],
                     }
                      # DATASET_META['MapText_json_train']['anno_path']: DATASET_META['MapText_json_train']['img_dir']
        
        self.samples = []
        for anno_path, _ in self.paths.items():
            print(anno_path, len(glob.glob(os.path.join(anno_path, '*.json'))))
            for p in glob.glob(os.path.join(anno_path, '*.json')):
                self.samples.append(p)
        self.samples = sorted(self.samples)
        random.seed(4321)
        random.shuffle(self.samples)

        if mode == 'train':
            self.samples = self.samples[:-200]
        else:
            self.samples = self.samples[-200:]
            
        print(f"Data ``{mode}'' contains {len(self.samples)} samples.")
        self.unused_indices = list(range(len(self.samples)))
        
    def reset(self):
        self.unused_indices = list(range(len(self.samples)))

    def __len__(self):
        return self.args.num_samples_per_epoch if self.mode == 'train' else len(self.samples)

    def get_data(self, idx, is_random=False):
        if is_random:
            random.seed()
            idx = random.choice(list(range(len(self.samples))))

        anno_file = self.samples[idx]
        anno_path = os.path.dirname(anno_file)
        image_path = self.paths[anno_path]
        name = os.path.basename(anno_file)
        if "maptext" in image_path:
            image_file = os.path.join(image_path, name.replace('.json', '.png'))
        else:
            image_file = os.path.join(image_path, name.replace('.json', '.jpg'))
        image = Image.open(image_file).convert("RGB")
        with open(anno_file, 'r') as f:
            anno = json.load(f)['annotations']
        return image, anno

    def __getitem__(self, idx):
        output = None
        while output is None:
            try:
                image, anno = self.get_data(idx, self.mode=='train')
                output =  process_one_sample(self.text_tokenizer, 
                                            self.image_processor, 
                                            self.image_mask_generator,
                                            anno, image, self.args)
            except Exception as e:
                print(e)
                output = None

            if output is None:
                # idx += 1
                idx = random.choice(list(range(len(self.samples))))

        return output











