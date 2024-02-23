import os, argparse, json, re
from collections import defaultdict
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

#KEYPOINT_CONSTANTS = [float(x) for x in '0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.072,0.072,0.062,0.062,0.107,0.107,0.087,0.087,0.089,0.089,0.026,0.026,0.107,0.025,0.025,0.025,0.025,0.025,0.025'.split(',')]
KEYPOINT_CONSTANTS = [float(x) for x in '0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.072,0.072,0.062,0.062,0.107,0.107,0.087,0.087,0.089,0.089,0.026,0.026,0.107,0.089,0.089,0.089,0.089,0.089,0.089'.split(',')]
KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel']

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_id_by_name(labels, name):
    return labels.index(name)


def writeJson(val,fname):
  with open(fname, 'w') as data_file:
    json.dump(val, data_file)

def get_bb_coco(points):
    x_values = [item["x"] for item in points]
    y_values = [item["y"] for item in points]

    # Get the minimum x and y values
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    box_width = max_x - min_x
    box_height = max_y - min_y

    return [min_x, min_y, box_width, box_height]

    
def get_bb(points):
    x_values = [item["x"] for item in points]
    y_values = [item["y"] for item in points]

    # Get the minimum x and y values
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    return [min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]

        
        
def compute_area(bounding_box):
    box_width = bounding_box[2][0] - bounding_box[0][0]
    box_height = bounding_box[2][1] - bounding_box[0][1]

    return box_width*box_height


def compute_area_coco(bounding_box):
    return bounding_box[2]*bounding_box[3]


def get_x_y_v_keypoints(kpts):
    kpts = np.array(kpts).reshape(-1,3)
    x = kpts[:,0]
    y = kpts[:,1]
    vi = kpts[:,2]
    return x, y, vi
    

def compute_area_keypoints(kpts):
    x_values, y_values, vi = get_x_y_v_keypoints(kpts)
    
    # Get the minimum x and y values
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    box_width = max_x - min_x
    box_height = max_y - min_y
    return box_width*box_height
    

def compute_area_keypoints(x_values, y_values):    
    # Get the minimum x and y values
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    box_width = max_x - min_x
    box_height = max_y - min_y
    return box_width*box_height


def load_images_dataframe():
    # Load images dataset
    running_annotations = '/mnt/d/Justine/uni/videos/lite_running_keypoints.json'
    with open(running_annotations, 'r') as f:
        annots = json.load(f)
        df_running_annotations = pd.DataFrame(annots)
    
    return df_running_annotations

    
def load_keypoints_dataframe():
    # Load images dataset
    running_annotations = '/mnt/d/Justine/uni/videos/person_keypoints_running.json'
    with open(running_annotations, 'r') as f:
        annots = json.load(f)
        df = pd.DataFrame(annots['annotations'])
    
    return df
    
def load_categories_dataframe():
    running_annotations = '/mnt/d/Justine/uni/videos/person_keypoints_running.json'
    with open(running_annotations, 'r') as f:
        annots = json.load(f)
        df = pd.DataFrame(annots['categories'])
    
    return df
    
def load_pe_dataframe(model):
    # Load images dataset
    running_annotations = '/mnt/d/Justine/uni/videos/results/' + model + '/person_keypoints_running.json'
    with open(running_annotations, 'r') as f:
        annots = json.load(f)
        df = pd.DataFrame(annots)
    
    return df

def get_head_size(x1,y1,x2,y2):
    headSize = 0.6*np.linalg.norm(np.subtract([x2,y2],[x1,y1]))
    return headSize


def compute_oks(joint, d, area, visibility):
    k = KEYPOINT_CONSTANTS[joint]
    
    # Compute the exponential part of the equation
    exp_vector = np.exp(-(d**2) / (2 * (area) * (k**2)))
    # The numerator expression
    numerator = np.dot(exp_vector, visibility)
    # The denominator expression
    denominator = np.sum(visibility)
    return numerator / denominator


def edit_keypoints(kpts):
    kpts = np.array(kpts).reshape(-1,3).astype(float)
    vi = kpts[:,2]
    kpts = kpts[:,0:2]
    return kpts, vi


def get_keypoint(kpts, id):
    index = KEYPOINTS.index(id)
    return [kpts[index*3],kpts[(index*3)+1]]

def OKS(kpts1, kpts2, area, start, end):

    kpts1, vi1 = edit_keypoints(kpts1[start*3:end*3])
    kpts2, vi2 = edit_keypoints(kpts2[start*3:end*3])

    if np.shape(kpts1) != np.shape(kpts2):
        print(kpts1, kpts2)
        print(np.shape(kpts1), np.shape(kpts2))
        raise ValueError("not same size")
    
    k = 2*KEYPOINT_CONSTANTS[start:end]
    s = area

    d = np.linalg.norm(kpts1 - kpts2, ord=2, axis=1)
    v = np.ones(len(d))

    for part in range(len(d)):
        if vi1[part] == 0 or vi2[part] == 0:
            d[part] = 0
            v[part] = 0
    
    if np.sum(v)!=0:
        OKS = (np.sum([(np.exp((-d[i]**2)/(2*s*(k[i]**2))))*v[i] for i in range(len(d))])/np.sum(v))
    else:
        OKS = 0
    
    OKS = float(Decimal(str(OKS)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))

    return OKS