import os
threads = '20'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
import re
import sys
import time
import pickle
import shutil
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
import random
import cv2
import scipy.io as sio
from skimage import io

# from utils import *
from multiprocessing import Pool
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

import DFA.mobilenet_v1 as mobilenet_v1
from DFA.utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool, reconstruct_vertex
from DFA.utils.inference import parse_roi_box_from_landmark, crop_img, predict_68pts
from DFA.utils.estimate_pose import parse_pose

std_size_dfa = 120

class DFADataset(Dataset):
    def __init__(self, frame_dir, frame_ids, frame_dict, transform=None):
        self.frame_dir = frame_dir
        self.frame_ids = frame_ids
        self.frame_dict = frame_dict
        self.transform = transform
        
    def __len__(self):
        return len(self.frame_ids)
    
    def __getitem__(self, index):
        frame_id = self.frame_ids[index]
        image = cv2.imread(os.path.join(self.frame_dir, f'{frame_id}.jpg'))
        pts = self.frame_dict[frame_id]['2d_landmarks'][self.frame_dict[frame_id]['target_idx']].T # (2, 68)
        roi_box = parse_roi_box_from_landmark(pts)  # list: [left, top, right, bottom]
        img = crop_img(image, roi_box)
        img = cv2.resize(img, dsize=(std_size_dfa, std_size_dfa), interpolation=cv2.INTER_LINEAR)
        img = self.transform(img) # (3, 120, 120)
        return dict(id=frame_id, image=img)
    
## 3d landmark detection -> obtain euler angle
def detect_3d_landmarks(frame_dir, score_dir):
    '''
    Estimate 3d landmarks and Euler angles using 3DDFA, and save euler angles, 3DDFA parameters
    to score_dir if the video has target.

    :param frame_dir: directory containing frames
    :param score_dir: directory for saving euler angles and 3DDFA parameters
    :return: no return
    '''

    frame_info = pd.read_csv(os.path.join(score_dir, 'frame_info.csv'), sep='\t')
    with open(os.path.join(score_dir, 'frame_2d_68pts_encodings.pkl'), 'rb') as f:
        frame_dict = pickle.load(f)

    # all frames
    frame_files = os.listdir(frame_dir)
    frame_files = [x for x in frame_files if '.jpg' in x]
    frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[0].replace('frame', '')))
    # filter frames
    frame_ids = frame_info.loc[frame_info['has_target'] == 1, 'frame_id'].tolist()
    if len(frame_ids) == 0:
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] No target, will not estimate euler angles for video {frame_dir.split('/')[-1]}")
        return

    ## load pre-tained model: 3DDFA
    checkpoint_fp = './DFA/models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    dfa_model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
    dfa_model_dict = dfa_model.state_dict()
    for k in checkpoint.keys():
        dfa_model_dict[k.replace('module.', '')] = checkpoint[k]
    dfa_model.load_state_dict(dfa_model_dict)
    if device != 'cpu':
        cudnn.benchmark = True
        dfa_model = dfa_model.to(device)
    dfa_model.eval()

    ## dataset
    batch_size = 128
    n_workers = 8
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    dfa_dataset = DFADataset(frame_dir, frame_ids, frame_dict, transform)
    dfa_dataloader = DataLoader(dfa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    # record euler angles
    eulers, params = [], []
    for _, data in enumerate(dfa_dataloader):
        with torch.no_grad():
            input_dfa = data['image'].to(device)
            param_dfa = dfa_model(input_dfa)
            param_dfa = param_dfa.cpu().numpy()  # (bs, 62)
        param_df = pd.DataFrame(param_dfa)
        param_df.insert(0, 'Frame_ID', data['id']) # (bs, 63)
        params.append(param_df)
        euler_batch = []
        for j in range(len(param_dfa)):
            _, euler_dfa = parse_pose(param_dfa[j, :])
            euler_batch.append(np.array(euler_dfa)/np.pi*180)
        euler_df = pd.DataFrame(np.vstack(euler_batch), columns=['alpha', 'beta', 'gamma'])
        euler_df.insert(0, 'Frame_ID', data['id']) # (bs, 4)
        eulers.append(euler_df) 
    if device != 'cpu':
        torch.cuda.empty_cache()
    eulers_df = pd.concat(eulers)
    params_df = pd.concat(params)
    # sort dataframe
    sort_index = eulers_df['Frame_ID'].apply(lambda x: x.split('_')[0].replace('frame','')).astype(int).argsort()
    eulers_df = eulers_df.iloc[sort_index].reset_index(drop=True)
    sort_index = params_df['Frame_ID'].apply(lambda x: x.split('_')[0].replace('frame','')).astype(int).argsort()
    params_df = params_df.iloc[sort_index].reset_index(drop=True)
    # convert params to 3d landmarks
    landmarks = dict.fromkeys(params_df['Frame_ID'].to_numpy().tolist())
    for k in landmarks.keys():
        pts = frame_dict[k]['2d_landmarks'][frame_dict[k]['target_idx']].T # (2, 68)
        roi_box = parse_roi_box_from_landmark(pts)
        param = params_df.loc[params_df['Frame_ID']==k].iloc[0, 1:].to_numpy()
        landmarks[k] = predict_68pts(param, roi_box) # (3, 68)

    ## save
    eulers_df.to_csv(os.path.join(score_dir, 'target_frame_euler.csv'), sep='\t', index=False)
    params_df.to_csv(os.path.join(score_dir, 'target_frame_dfa_params.csv'), sep='\t', index=False)
    with open(os.path.join(score_dir, 'target_frame_3d_landmarks.pkl'), 'wb') as f:
        pickle.dump(landmarks, f)
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Euler angles estimated video {frame_dir.split('/')[-1]}")
    logging.info(f"Euler angles estimated video {frame_dir.split('/')[-1]} (PID: {os.getpid()})")   
    
if __name__ == '__main__':

    ## parse arguments
    parser = argparse.ArgumentParser()
    # directories
    parser.add_argument('--video-root', type=str, help='Specify the directory containing the videos.')
    parser.add_argument('--frame-root', type=str, help='Specify the directory for saving extracted frames.')
    parser.add_argument('--score-root', type=str, help='Specify the directory for saving information and scores.')
    # global setting
    parser.add_argument('--device', type=str, default='cpu', help='Specify the device for 3DDFA. Default is cpu.')

    args = parser.parse_args()

    device = args.device
    assert device == 'cpu' or bool(re.match(r'cuda:[0-9]*', device)), "Device name should be 'cpu' or 'cuda:<0-9>'"
    device_name = device
    if device != 'cpu':
        device_name = 'gpu'
    video_root = args.video_root
    frame_root = args.frame_root
    score_root = args.score_root
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file = f"logs/{frame_root.split('/')[1].split('_')[0]}_3d_landmarks_{device_name}.log"
    open(log_file, 'w').close()
    logging.basicConfig(filename=log_file,
                        level=logging.INFO, 
                        format='[%(asctime)s] [%(levelname)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a+')

    video_ids = os.listdir(video_root)
    video_ids = sorted([x.split('.')[0] for x in video_ids])

    ## 3. 3d landmarks and euler angle
    for idx, vid_id in enumerate(video_ids):
        frame_dir = os.path.join(frame_root, vid_id)
        score_dir = os.path.join(score_root, vid_id)
        try:
            detect_3d_landmarks(frame_dir, score_dir)
        except Exception as e:
            print(e)
            logging.error(e)
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 3: estimate 3d landmarks and euler angles **********")
    logging.info('********** Finish step 3: estimate 3d landmarks and euler angles **********')

            