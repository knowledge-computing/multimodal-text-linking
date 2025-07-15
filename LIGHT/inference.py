import os
import sys
import json
import argparse
import numpy as np
import yaml
from typing import Optional, Tuple, Union
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.dataset import LinkingTestDataset
from dataset.buildin import DATASET_META
from models.text_linking import LightTextLinking
from models.model_utils import get_processors


def group_successors(elements, successors):
    """
    Merge the elements that point to the same successor (ordered by indices)
    """
    ele2suc_mapping = {}
    suc2ele_mapping = {}
    for element, successor in zip(elements, successors):
        ele2suc_mapping[element] = successor
        if element != successor:
            suc2ele_mapping[successor] = element
    
    groups = []
    seen = set()
    def find_path(x):
        path = []
        local_seen = set()
        path = [x]
        while x not in local_seen and x not in seen:  # Keep following until self-loop
            if suc2ele_mapping.get(x) is not None:
                path.insert(0, suc2ele_mapping[x])
                local_seen.add(x)
                x = suc2ele_mapping[x]
            else:
                break
        return path

    # Iterate through all elements and assign them to groups
    for elem in elements:
        if elem not in seen:
            if ele2suc_mapping[elem] == elem:            
                path = find_path(elem)
                groups.append(path)
                for p in path:
                    seen.add(p)

    return groups


def group_successors_with_probability(words, probabilities, bi_probabilities):
    """
    Groups elements based on successor relationships based on probabilities.
    """
    word2succ = {}
    succ2word = {}
    
    iteration = 0
    while len(word2succ) < len(words):
        for i, word in enumerate(words):
            
            prob = probabilities[i]
            sorted_indices = np.argsort(prob)[::-1]
            j = sorted_indices[0]

            if succ2word.get(j) is None:
                word2succ[i] = j
                succ2word[j] = i
            else:
                old_i = succ2word[j]
                if old_i == i: continue;
                elif old_i == j:
                    word2succ[i] = j
                    succ2word[j] = i
                elif i == j:
                    word2succ[i] = j
                elif bi_probabilities[j][i] > bi_probabilities[j][old_i]:
                    word2succ[i] = j
                    succ2word[j] = i
                    word2succ.pop(old_i)
                    probabilities[old_i][j] = 0.
                    bi_probabilities[j][old_i] = 0.
                else:
                    probabilities[i][j] = 0.
                    bi_probabilities[j][i] = 0.
        
    new_successors = [word2succ[i] for i, word in enumerate(words)]
    return group_successors(words, new_successors)


def main():
    # python inference.py --test_dataset test --out_file lithium.json --model_dir _runs/best/ --anno_path /home/yaoyi/shared/critical-maas/12month-text-extraction/spot/lithium.json  --img_dir /home/yaoyi/shared/critical-maas/12month-text-extraction/img_crops/lithium
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--test_dataset', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--anno_path', type=str)
    parser.add_argument('--img_dir', type=str)
    args, remaining_args = parser.parse_known_args()
    
    with open(os.path.join(args.model_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    args = parser.parse_args()

    ### Load model ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(args.model_dir, 'best_model.pth')
    state_dict = torch.load(checkpoint_path)

    model = LightTextLinking(args)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model.to(device)
    
    ### Load data ###
    args.test_data_shuffle=False
    tokenizer, image_processor = get_processors(args.pretrained_model_name)
    test_dataset = LinkingTestDataset(tokenizer, image_processor, args, mode='test', return_ori=True)

    model.eval()
    with torch.no_grad():
        result_list = []
        for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
            sample_data = test_dataset[i]
            if sample_data is None:
                continue
                
            image_name = sample_data['image_name']
            input_data = sample_data.copy()
            input_data.pop('image_name')
            input_data.pop('ori_words')
            input_data.pop('ori_polygons')
            input_data.pop('ori_word_ids')
            input_data = {k: v.unsqueeze(0).to(device) for k, v in input_data.items()}
            all_logits, _ = model(input_data, return_loss=False)
            
            probabilities = torch.softmax(all_logits['logits'][0], dim=-1)
            bi_probabilities = torch.softmax(all_logits['bi_logits'][0], dim=-1)
            probabilities = probabilities.detach().cpu().numpy()
            bi_probabilities = bi_probabilities.detach().cpu().numpy()
            
            groups = group_successors_with_probability(sample_data['ori_word_ids'], 
                                                       probabilities, bi_probabilities)
                
            ori_words = sample_data['ori_words']
            ori_polygons = sample_data['ori_polygons']
            ori_word_ids = sample_data['ori_word_ids']

            if 'MapText' in args.test_dataset:
                result = {"image": "rumsey/test/" + sample_data['image_name'], "groups": []}
            # elif 'HierText' in args.test_dataset:
            #     result = {"image": sample_data['image_name'].split('.')[0], "groups": []}
            # elif 'IGN' in args.test_dataset:
            #     result = {"image": "ign/test/" + sample_data['image_name'], "groups": []}
            else:
                result = {"image": image_name, "groups": []}
                
            for group in groups:
                group_items = []
                for idx in group:
                    item_dict = {'text': ori_words[idx], 'vertices': ori_polygons[idx]}
                    group_items.append(item_dict)
                result['groups'].append(group_items)

            result_list.append(result)    

        with open(os.path.join(args.model_dir, args.out_file), 'w') as f:
            json.dump(result_list, f, indent=4)

    if 'MapText' in args.test_dataset:
        gt_path = DATASET_META['MapText_test']['anno_path']
    # if 'HierText' in args.test_dataset:
    #     gt_path = DATASET_META['HierText_test']['anno_path']
    # if 'IGN' in args.test_dataset:
    #     gt_path = DATASET_META['IGN_test']['anno_path']
    print(f"python evaluation/eval.py --gt {gt_path} --task detrecedges --pred {os.path.join(args.model_dir, args.out_file)}")
    os.system(f"python evaluation/eval.py --gt {gt_path} --task detrecedges --pred {os.path.join(args.model_dir, args.out_file)}")

if __name__ == '__main__':
    main()


