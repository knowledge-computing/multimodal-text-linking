import os
import json
import numpy as np
import random
import string
from collections import defaultdict
from fractions import Fraction
from PIL import Image
import cv2

import copy
import pycocotools.mask as mask_util
from typing import Any, Iterator, List, Union


def crop_image_into_patches(image, patch_size=500):
    img_w, img_h = image.size
    patches = []
    for y in range(0, img_h, patch_size):
        for x in range(0, img_w, patch_size):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
    return patches

####################################################################
####################################################################
####################################################################

def create_bounding_box(vertices):
    """
        bbox[x0, y0, x1, y1]
        (x0, y0) - upper left
        (x1, y1) - lower right
    """
    return [
        min([vert[0] for vert in vertices]),
        min([vert[1] for vert in vertices]),
        max([vert[0] for vert in vertices]),
        max([vert[1] for vert in vertices])]

def scale_bounding_box(bbox, width_scale, height_scale):
    return [
        int(bbox[0] * width_scale),
        int(bbox[1] * height_scale),
        int(bbox[2] * width_scale),
        int(bbox[3] * height_scale)]

####################################################################
####################################################################
####################################################################

def pt_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def get_char_height(vertices):
    return pt_distance(vertices[0], vertices[-1])
    
# def get_center(vertices):
#     cx = np.mean(np.array(vertices)[:, 0])
#     cy = np.mean(np.array(vertices)[:, 1])
#     return [cx, cy]

def get_center(vertices):
    polygon = np.array(vertices, dtype=np.float32)
    cx = (np.min(polygon[:, 0]) + np.max(polygon[:, 0])) / 2
    cy = (np.min(polygon[:, 1]) + np.max(polygon[:, 1])) / 2
    return [cx, cy]

# def get_angle(vertices):
#     vertices = np.array(vertices)
#     n = vertices.shape[0] // 2
#     v1 = vertices[n-1] - vertices[0]
#     u1 = v1 / np.linalg.norm(v1)
#     u2 = np.array([1, 0])
#     angle = np.arccos(np.clip(np.dot(u1, u2), -1, 1))
#     return angle / np.pi * 180

def get_angle(vertices):
    polygon = np.array(vertices, dtype=np.float32)
    rect = cv2.minAreaRect(polygon)  # (center, (width, height), angle)
    box = cv2.boxPoints(rect)  # Extract the 4 corners
    box = np.array(box, dtype=np.float32)

    # Get the angle from the rectangle
    angle = rect[2]  # Angle returned by OpenCV
    # Normalize angle to -90 to 90 degrees
    if rect[1][0] < rect[1][1]:  # Width < Height
        angle += 90  # Rotate if necessary
    return angle

####################################################################
####################################################################
####################################################################

def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.bool_)


def rasterize_polygons_within_box(
    polygons: List[np.ndarray], box: np.ndarray, mask_size: int
) -> np.ndarray:
    """
    Rasterize the polygons into a mask image and
    crop the mask content in the given box.
    The cropped mask is resized to (mask_size, mask_size).

    This function is used when generating training targets for mask head in Mask R-CNN.
    Given original ground-truth masks for an image, new ground-truth mask
    training targets in the size of `mask_size x mask_size`
    must be provided for each predicted box. This function will be called to
    produce such targets.

    Args:
        polygons (list[ndarray[float]]): a list of polygons, which represents an instance.
        box: 4-element numpy array
        mask_size (int):

    Returns:
        Tensor: BoolTensor of shape (mask_size, mask_size)
    """
    # 1. Shift the polygons w.r.t the boxes
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    # 2. Rescale the polygons to the new box size
    # max() to avoid division by small number
    ratio_h = mask_size / max(h, 0.1)
    ratio_w = mask_size / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    mask = polygons_to_bitmask(polygons, mask_size, mask_size)
    return mask
    
####################################################################
####################################################################
####################################################################
    
def generate_unique_strings(N, length=3):
    unique_strings = set()
    while len(unique_strings) < N:
        new_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        unique_strings.add(new_string)
    return list(unique_strings)
    
def generate_unique_group_id(N):
    group_id_list = [i for i in range(1, N) if '0' not in str(i)]
    return group_id_list
    
def is_number_extended(s):
    try:
        # Check if it's a float or integer
        s = s.replace('.', '')
        s = s.replace('/', '')
        float(s)
        return True
    except ValueError:
        try:
            # Check if it's a fraction
            Fraction(s)
            return True
        except ValueError:
            return False

def generate_labels(groups):
    WORD_IDs = generate_unique_strings(1000)
    GROUP_IDs = generate_unique_group_id(1000)

    out_groups = {}
    for i, group in enumerate(groups):
        group_id = GROUP_IDs[i]
        seq_id = 1
        for j, item in enumerate(group):
            out_item = item.copy()
            wid = WORD_IDs[0]
            del WORD_IDs[0]
            out_item['id'] = wid
            if item.get('illegible', False) or item.get('truncated', False) or '###' in item['text']:
                if j == 0 or j == len(group) - 1:
                    out_groups[wid] = out_item
                    continue

            out_item['label'] = int(f"{group_id}000{seq_id}")  
            seq_id += 1
            out_groups[wid] = out_item
    return out_groups

####################################################################
####################################################################
####################################################################

def pad_sequence(seq, max_length, pad_value=-999):
    """
    Pads a sequence to the desired max_length with pad_value.
    If the sequence is longer, it is truncated.
    """
    seq = np.array(seq).reshape(-1)
    padded_seq = np.full((max_length,), pad_value, dtype=seq.dtype)
    padded_seq[:min(len(seq), max_length)] = seq[:max_length]
    return padded_seq


def compute_polygon(polygon, image_width, image_height, sampling=False):
    
    poly = polygon.copy()
    if polygon.shape[0] == 16 and sampling:
        mask = np.random.rand(16)
        mask[[0, 7, 8, 15]] = 1.
        prob = random.random()
        poly = poly[mask > prob, :]
        
    poly[:, 0] /= image_width
    poly[:, 1] /= image_height
    return poly