import os
import sys
import time
import glob
import pickle
import shutil
import logging
import argparse
import traceback
import datetime
import numpy as np
import pandas as pd
import cv2
import dlib
import math
import random
import itertools
import statistics
import mediapipe as mp
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
from collections import OrderedDict
from pathlib import Path
from utils import *

## correspondence between 468 landmarks and key points for masculinity measurement
P1 = [33, 263]
P2 = [133, 362]
P3 = [234, 454]
P4 = [129, 358]
P5 = [61, 291]
P6 = [58, 288]
P7 = [10, 152]
pupillaries = [468, 473]
right_eyebrow = [46, 52, 55]
left_eyebrow = [276, 282, 285]
eyebrows = right_eyebrow + left_eyebrow
points = {'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'pupillaries': pupillaries, 'eyebrows': eyebrows}

## filter frames with neutral emotion and small eular angle
def filter_frames(frame_root, score_root, euler_thres=[10, 10, 10], nsample=10):
    
    '''
    Filter frames for measuring facial masculinity.
    
    :param score_root: directory saving euler angles and emotions
    :param euler_thres: thresholds of euler angles
    :param nsample: number of images to be used 
    :return all_frames: a dict saving frame files for all videos
    '''
    
    videos = os.listdir(score_root)
    # emotion and euler angle
    all_eulers = []
    for video in videos:
        # euler angle
        euler_file = os.path.join(score_root, video, 'target_frame_euler.csv')
        if os.path.exists(euler_file):
            df = pd.read_csv(euler_file, sep='\t')
            # emotion
            with open(os.path.join(score_root, video, 'scores_emonet.pkl'), 'rb') as f:
                emos = pickle.load(f)
            emo = emos['align_emos'] # a dataframe
            # neutral or not
            emo['neutral'] = np.where(emo.iloc[:, 1:].values.argmax(axis=1) == 0, 1, 0)
            # merge two dataframes
            df = df.merge(emo[['Frame_ID', 'neutral']], on='Frame_ID')
            df.insert(0, 'Video_ID', video)
            all_eulers.append(df)
    eulers = pd.concat(all_eulers)
    eulers[['alpha', 'beta', 'gamma']] = eulers[['alpha', 'beta', 'gamma']]
    
    g = eulers.loc[eulers.neutral == 1]
    g = g.groupby('Video_ID').apply(lambda x: x.sort_values(by=['alpha','beta','gamma'], key=abs)).reset_index(drop=True)
    tmp = g.loc[(abs(g.alpha) < euler_thres[0]) & (abs(g.beta) < euler_thres[1]) & (abs(g.gamma) < euler_thres[2])].copy()
    if len(tmp) > 0:
        g = tmp.groupby('Video_ID').head(nsample)
    else:
        g = g.groupby('Video_ID').head(nsample)
        
    all_frames = {k: [] for k in set(g.Video_ID)}
    all_imagesizes = {k: None for k in set(g.Video_ID)}
    for _, row in g.iterrows():
        all_frames[row.Video_ID].append(os.path.join(frame_root, row.Video_ID, f'{row.Frame_ID}.jpg'))
    for k in all_frames.keys():
        example_image = cv2.imread(all_frames[k][0])
        all_imagesizes[k] = example_image.shape[:2]
        # print(example_image.shape[:2])
        
    return all_frames, all_imagesizes

# convert landmarks object to np.ndarray (N, 3)
def landmarks_to_np_(landmarks, imagesize):
    arr = []
    for p in landmarks:
        arr.append([p.x, p.y, p.z])
    arr = np.vstack(arr)
    # retrieve absolute distances 
    arr[:, 0] = arr[:, 0] * imagesize[1]
    arr[:, 1] = arr[:, 1] * imagesize[0]
    arr[:, 2] = arr[:, 2] * imagesize[1]
    
    return arr

# function for detecting 478 facial landmarks
mp_face_mesh = mp.solutions.face_mesh
def detect_mesh_landmarks_(image_file):
    
    image = cv2.imread(image_file)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
        result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not result.multi_face_landmarks:
        return None
    
    return result.multi_face_landmarks[0].landmark

## detect face mesh landmarks: people to be studied
def detect_mesh_landmarks(all_frames, imagesize=(256, 256)):
    
    '''
    Detect face mesh keypoints (478 in total).
    
    :param all_frames: dict saving filtered frames for all videos
    :param all_imagesizes: dict saving image sizes for all videos
    :return all_landamrks: a dict saving face mesh landmarks for all videos
    '''
    
    all_landmarks = {k: [] for k in all_frames.keys()}
    for k, v in all_frames.items():
        landmarks = []           
        for frame in v:
            raw_landmark = detect_mesh_landmarks_(frame)
            if raw_landmark is not None:
                landmark = landmarks_to_np_(raw_landmark, imagesize)
                landmarks.append(landmark)

        all_landmarks[k] = landmarks
    
    return all_landmarks

# rotation angle
def rotation_angle_(landmarks):
    dX = landmarks[pupillaries[1], 0] - landmarks[pupillaries[0], 0]
    dY = landmarks[pupillaries[0], 1] - landmarks[pupillaries[1], 1]
    theta = np.arctan2(dY, dX)
    return theta

# function for rotating keypoints such that pupillaries connection line is horizontal
def rotate_mesh_(landmarks):
    
    theta = rotation_angle_(landmarks)
    c, s = np.cos(theta), np.sin(theta)
    # print(theta)
    # rotation matrix
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # rotation center
    center = (landmarks[pupillaries[0], :] + landmarks[pupillaries[1], :]) / 2
    # rotate keypoints
    new_landmarks = center + (landmarks - center) @ R.T
    
    return new_landmarks

def average_dists_(dists):
    
    # dists: a list of dist dictionary
    if len(dists) == 0:
        raise Exception('No distances to be averaged!')
    elif len(dists) == 1:
        return dists[0]
    else:
        dist = []
        for d in dists:
            df = pd.DataFrame(d, index=[0])
            dist.append(df)
        dist = pd.concat(dist).mean().to_dict()
    return dist

def get_distances_(landmarks):
    
    interpupillary = np.linalg.norm(landmarks[pupillaries[1]] - landmarks[pupillaries[0]])
    scale = 1 / interpupillary
    D0 = sum([abs(landmarks[P7[1],1] - landmarks[i,1]) for i in eyebrows]) / len(eyebrows) * scale
    D1 = np.linalg.norm(landmarks[P1[1]] - landmarks[P1[0]]) * scale
    D2 = np.linalg.norm(landmarks[P2[1]] - landmarks[P2[0]]) * scale
    D3 = np.linalg.norm(landmarks[P3[1]] - landmarks[P3[0]]) * scale
    D4 = np.linalg.norm(landmarks[P4[1]] - landmarks[P4[0]]) * scale
    D5 = np.linalg.norm(landmarks[P5[1]] - landmarks[P5[0]]) * scale
    D6 = np.linalg.norm(landmarks[P6[1]] - landmarks[P6[0]]) * scale
    D7 = (landmarks[P7[1],1] - landmarks[P7[0],1]) * scale
    D8 = (abs(landmarks[P7[1],1] - landmarks[P1[0],1]) + abs(landmarks[P7[1],1] - landmarks[P1[1],1])) / 2 * scale
    
    return {'D0': D0, 'D1': D1, 'D2': D2, 'D3': D3, 'D4': D4, 'D5': D5, 'D6': D6, 'D7': D7, 'D8': D8}

def get_distances(all_landmarks):
    
    '''
    Calculate distances for measuring facial masculinity and asymmetry.
    
    :param all_landmarks: dict saving face mesh landmarks for all videos
    :return all_distances, avg_distances: a dict saving distances for all videos, a dataframe saving average distances for all videos
    '''
    
    all_distances = {k: [] for k in all_landmarks.keys()}
    avg_distances = pd.DataFrame(columns=['Video_ID', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'])
    for i, (k, v) in enumerate(all_landmarks.items()):
        if len(v) == 0:
            continue
        distances = [get_distances_(landmark) for landmark in v]
        all_distances[k] = distances
        avg_distances.loc[i] = [k] + list(average_dists_(distances).values())
        
    return all_distances, avg_distances

def measure_facial_masculinity(dist):
        
    # facial masculinity for faces in study
    if 'gender' in dist.columns:
        mas = pd.DataFrame({'Video_ID': dist.iloc[:,0], 'gender': dist['gender']})
    else:
        mas = pd.DataFrame({'Video_ID': dist.iloc[:,0]})
    mas['eye_size'] = (dist['D1'] - dist['D2']) / 2
    mas['lower_height'] = dist['D8'] / dist['D7']
    mas['cheekbone_prominence'] = dist['D3'] / dist['D6']
    mas['width_lower'] = dist['D3'] / dist['D8']
    mas['eyebrow_height'] = dist['D0']
    
    # mas['facial_masculinity'] = mas['lower_height'] - mas['width_lower'] - mas['eye_size'] - mas['eyebrow_height']
    mas['facial_masculinity'] = mas['lower_height'] - mas['width_lower'] - mas['eye_size'] - mas['eyebrow_height'] - mas['cheekbone_prominence'] 
        
    return mas

def measure_facial_asymmetry_(landmarks):
    
    # scale factor
    interpupillary = np.linalg.norm(landmarks[pupillaries[1]] - landmarks[pupillaries[0]])
    scale = 1 / interpupillary
    
    mid_x, diffs_y = [], []
    for i in range(1, 7):
        idx = points[f'P{i}']
        mid_x.append((landmarks[idx[0],0] + landmarks[idx[1],0]) / 2)
        diffs_y.append(abs(landmarks[idx[0],1] - landmarks[idx[1],1]) * scale)
    vert_asym = sum(diffs_y)
    
    diffs_x = [abs(e[1] - e[0]) for e in itertools.permutations(mid_x, 2)]
    hori_asym = sum(diffs_x) / 2 * scale
    
    return hori_asym, vert_asym

def measure_facial_asymmetry(all_landmarks):
    
    '''
    Measure facial asymmetry.
    
    :param all_landmarks: dict saving landmarks for all videos
    :return mas: a dataframe saving facial asymmetry for all videos
    '''
    
    asym = pd.DataFrame(columns=['Video_ID', 'horizontal_asymmetry', 'vertical_asymmetry'])
    for i, (k, v) in enumerate(all_landmarks.items()):
        if len(v) == 0:
            asym.loc[i] = [k, None, None]
            continue
        hori_asyms, vert_asyms = [], []
        for landmark in v:
            hori_asym, vert_asym = measure_facial_asymmetry_(landmark)
            hori_asyms.append(hori_asym)
            vert_asyms.append(vert_asym)
        asym.loc[i] = [k] + [statistics.mean(hori_asyms), statistics.mean(vert_asyms)]
    
    return asym


if __name__ == '__main__':

    ## parse arguments
    parser = argparse.ArgumentParser()
    # directories/files
    parser.add_argument('--frame-root', type=str, help='Specify the directory containing video frames.')
    parser.add_argument('--score-root', type=str, help='Specify the directory containing euler angles and emotions.')
    parser.add_argument('--align-root', type=str, help='Specify the directory for saving aligned faces.')
    parser.add_argument('--output-root', type=str, help='Specify the directory for saving result.')
    parser.add_argument('--gender-est', type=str, help='Specify the file containing gender estimates.')
    parser.add_argument('--all-landmarks', type=str, help='Specify the file for saving face mesh landmarks for all videos.')
    
    args = parser.parse_args()
    
    ## 1. Filter frames with neutral emotion and small eular angle
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 1: filter frames for measuring facial musculinity **********")
    all_frames, all_imagesizes = filter_frames(args.frame_root, args.score_root)
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 1: filter frames for measuring facial musculinity **********")

    ## 2. Retrieve 68 landmarks and align faces 
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 2: align faces in filtered frames **********")
    all_2d_landmarks = dict.fromkeys(all_frames.keys())
    for k, v in all_frames.items():
        landmarks = {}
        for frame in v:
            framename = frame.split('/')[-1].split('.')[0]
            filename = f'{args.score_root}/{k}/frame_2d_68pts_encodings.pkl'
            with open(filename, 'rb') as f:
                tmp = pickle.load(f)
            landmarks[framename] = tmp[framename]['2d_landmarks'][0]
        all_2d_landmarks[k] = landmarks
        
    for vid_id, d in all_2d_landmarks.items():
        Path(f'{args.align_root}/{vid_id}').mkdir(parents=True, exist_ok=True)
        for frame, landmarks in d.items():
            image = cv2.imread(f'{args.frame_root}/{vid_id}/{frame}.jpg')
            align = face_align(image, landmarks, desiredLeftEye=(0.4,0.4), desiredFaceWidth=256)
            cv2.imwrite(f'{args.align_root}/{vid_id}/{frame}.jpg', align)
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 2: align faces in filtered frames **********")

    ## 3. Detect face meshes on aligned faces
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 3: detect face meshes on aligned faces **********")
    all_aligned_images = {k: [f'{args.align_root}/{k}/{x}.jpg' for x in list(v.keys())] for k, v in all_2d_landmarks.items()}
    all_landmarks = detect_mesh_landmarks(all_aligned_images, imagesize=(256, 256))
    with open(args.all_landmarks, 'wb') as f:
        pickle.dump(all_landmarks, f)
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 3: detect face meshes on aligned faces **********")

    ## 4. Calculate distances
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 4: calculate distances between facial keypoints **********")
    for k, v in all_landmarks.items():
        new_landmark = []
        for landmark in v:
            new = rotate_mesh_(landmark)
            new_landmark.append(new)
        all_landmarks[k] = new_landmark
    all_distances, avg_distances = get_distances(all_landmarks)
    if args.gender_est is not None:
        gender_est = pd.read_csv(args.gender_est, sep='\t')
        gender_est['Video_ID'] = gender_est['Video_ID'].apply(str)
        avg_distances['Video_ID'] = avg_distances['Video_ID'].apply(str)
        avg_distances = avg_distances.merge(gender_est, on='Video_ID')
        male_distances = avg_distances.loc[avg_distances.gender=='Male']
        female_distances = avg_distances.loc[avg_distances.gender=='Female']      
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 4: calculate distances between facial keypoints **********")

    ## 5. Measure facial masculinity
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 5: measure facial masculinity **********")
    if args.gender_est is not None:
        male_masculinity = measure_facial_masculinity(male_distances).sort_values(by='Video_ID', key=lambda x: x.astype(int)).reset_index(drop=True)
        female_masculinity = measure_facial_masculinity(female_distances).sort_values(by='Video_ID', key=lambda x: x.astype(int)).reset_index(drop=True)
        masculinity = pd.concat([male_masculinity, female_masculinity]).reset_index(drop=True)
    else:
        masculinity = measure_facial_masculinity(avg_distances).sort_values(by='Video_ID', key=lambda x: x.astype(int)).reset_index(drop=True)
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 5: measure facial masculinity **********")

    ## 6. Measure facial asymmetry
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 6: measure facial asymmetry **********")
    asymmetry = measure_facial_asymmetry(all_landmarks)
    asymmetry = asymmetry.dropna().sort_values(by='Video_ID', key=lambda x: x.astype(int)).reset_index(drop=True)
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 6: measure facial asymmetry **********")
    
    ## 7. Save result
    face_result = pd.merge(masculinity, asymmetry, on='Video_ID')
    face_result.to_csv(os.path.join(args.output_root, 'facial_masculinity_asymmetry.csv'), sep='\t', index=False)
