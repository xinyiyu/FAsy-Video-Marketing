import os
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
from multiprocessing import Pool, Process
from multiprocessing_logging import install_mp_handler
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from emonet.emonet.models import EmoNet

## facial expression
class EmonetDataset(Dataset):
    def __init__(self, root_path, transform_image=None, suffix=None):
        self.root_path = root_path
        self.transform_image = transform_image
        self.suffix = suffix

    def __len__(self):
        files = os.listdir(self.root_path)
        length = sum(['.jpg' in x for x in files])
        return length if self.suffix is None else int(length / 2)

    def __getitem__(self, index):
        image_files = os.listdir(self.root_path)
        image_files = [x for x in image_files if x.split('.')[1] == 'jpg']
        if self.suffix is not None:
            image_files = list(filter(lambda x: self.suffix in x, image_files))
        image_files = sorted(image_files, key=lambda x: int(x.split('_')[0].replace('frame', '')))
        image = io.imread(os.path.join(self.root_path, image_files[index]))
        if self.transform_image is not None:
            image = self.transform_image(image)
        if self.suffix is None:
            return dict(id=image_files[index].split('.')[0], image=image)
        else:
            return dict(id=image_files[index].split('.')[0].replace(f'_{self.suffix}', ''), image=image)

def facial_expression_analysis(align_dir, compo_dir, score_dir, fer_model=['ferplus', 'emonet'], dev='cpu'):
    '''
    :param align_dir: directory containing aligned face images.
    :param compo_dir: directory containing composite face images.
    :param fer_model: facial expression analysis model, 'ferplus' only classifies emotions, 'emonet' classifies emotions and scores valence-arousal.
    :return: no return
    '''
    ## aligned face images
    align_files = os.listdir(align_dir)
    align_files = [x for x in align_files if '.jpg' in x]
    align_files = sorted(align_files, key=lambda x: int(x.split('_')[0].replace('frame', '')))
    align_ids = [x.split('.')[0] for x in align_files]
    if len(align_ids) == 0:
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Video {align_dir.split('/')[-1]} has no target faces, will not analyze emotions")
        logging.info(f"Video {align_dir.split('/')[-1]} has no target faces, will not analyze emotions")
        return
        
    # 'ferplus': cpu, multiprocess
    if fer_model == 'ferplus':
        
        # save as a dict with 4 keys
        scores = dict.fromkeys(['align_emos', 'left_emos', 'right_emos', 'FAsy_emos'])

        emotions = ['Neutral', 'Happy', 'Surprise', 'Sad',
                    'Anger', 'Disgust', 'Fear', 'Contempt']
        # dnn emotion model
        ferplus_model = './emotion_ferplus/model.onnx'
        ferplus_net = cv2.dnn.readNetFromONNX(ferplus_model)
        
        for idx, dir_suf in enumerate(zip([align_dir, compo_dir, compo_dir], ['', '_left', '_right'])):
            image_files = [os.path.join(dir_suf[0], f'{k}{dir_suf[1]}.jpg') for k in align_ids]
            # print(image_files[:3])
            processed_faces = []
            for _, image_file in enumerate(image_files):
                image = cv2.imread(image_file)
                # color to gray, resize & reshape
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray, (64, 64))
                processed_faces.append(resized_face[np.newaxis,np.newaxis,:,:])
            processed_faces = np.concatenate(processed_faces, axis=0) # np.ndarray (N, 1, 64, 64)
            # score emotions
            ferplus_net.setInput(processed_faces) 
            Output = ferplus_net.forward() # (N, 8)
            # softmax
            exp = np.exp(Output)
            prob = exp / np.sum(exp, axis=1)[:, None] # (N, 8)
            score_df = pd.DataFrame(prob, columns=emotions) # (N, 8)
            score_df.insert(0, 'Frame_ID', align_ids) # (N, 9)
            # sort dataframes
            sort_index = score_df['Frame_ID'].apply(lambda x: x.split('_')[0].replace('frame','')).astype(int).argsort()
            score_df = score_df.iloc[sort_index].reset_index(drop=True)
            # record
            if idx == 0:
                scores['align_emos'] = score_df
            elif idx == 1:
                scores['left_emos'] = score_df
            else:
                scores['right_emos'] = score_df
        score_df = scores['left_emos'].copy()
        score_df.iloc[:, 1:] = scores['left_emos'].iloc[:, 1:] - scores['right_emos'].iloc[:, 1:]
        scores['FAsy_emos'] = score_df
        
    # 'emonet': gpu, serialization, batch
    elif fer_model == 'emonet':

        # save as a dict with 8 keys
        scores = dict.fromkeys(['align_emos', 'left_emos', 'right_emos', 'FAsy_emos',
                               'align_vai', 'left_vai', 'right_vai', 'FAsy_vai'])

        emotions_emonet = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
                           4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'}
        n_expression = 8
        batch_size = 64 if dev != 'cpu' else len(align_ids)
        n_workers = min(8, os.cpu_count()//len(device)+1) if dev != 'cpu' else 0 # num_workers set to 4 or 8 fastest
        if dev != 'cpu':
            cudnn.benchmark = True

        # load and transform images
        trans_emonet = transforms.ToTensor()
        
        # load emonet
        emonet_state_path = './emonet/pretrained/emonet_8.pth'
        emonet_state_dict = torch.load(str(emonet_state_path), map_location='cpu')
        emonet_state_dict = {k.replace('module.', ''): v for k, v in emonet_state_dict.items()}
        emonet = EmoNet(n_expression=n_expression).to(dev)
        emonet.load_state_dict(emonet_state_dict, strict=False)
        emonet.eval()

        
        for idx, dir_suf in enumerate(zip([align_dir, compo_dir, compo_dir], [None, '_left', '_right'])):
            
            emonet_dataset = EmonetDataset(root_path=dir_suf[0], transform_image=trans_emonet, suffix=dir_suf[1])
            emonet_dataloader = DataLoader(emonet_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
            # forward emonet
            emotions, vais = [], []
            for _, data in enumerate(emonet_dataloader):

                images = data['image'].to(dev)
                with torch.no_grad():
                    out = emonet(images)
                emos = torch.softmax(out['expression'], axis=1).cpu().detach().numpy()  # (bs, 8)
                emos_df = pd.DataFrame(emos, columns=list(emotions_emonet.keys()))
                emos_df.insert(0, 'Frame_ID', data['id'])
                emotions.append(emos_df)
                vals = out['valence'].cpu().detach().numpy()  # (bs,)
                arous = out['arousal'].cpu().detach().numpy()  # (bs,)
                intens = np.sqrt(vals ** 2 + arous ** 2)
                vai_df = pd.DataFrame({'Frame_ID': data['id'], 'Valence': vals, 'Arousal': arous, 'Intensity': intens})
                vais.append(vai_df)
            if dev != 'cpu':
                torch.cuda.empty_cache()
            emotions_df = pd.concat(emotions)
            vais_df = pd.concat(vais)
            # sort dataframes
            sort_index = emotions_df['Frame_ID'].apply(lambda x: x.split('_')[0].replace('frame','')).astype(int).argsort()
            emotions_df = emotions_df.iloc[sort_index].reset_index(drop=True)
            sort_index = vais_df['Frame_ID'].apply(lambda x: x.split('_')[0].replace('frame','')).astype(int).argsort()
            vai_df = vais_df.iloc[sort_index].reset_index(drop=True)
            # record
            if idx == 0:
                scores['align_emos'] = emotions_df
                scores['align_vai'] = vais_df
            elif idx == 1:
                scores['left_emos'] = emotions_df
                scores['left_vai'] = vais_df
            else:
                scores['right_emos'] = emotions_df
                scores['right_vai'] = vais_df
        score_df = scores['left_emos'].copy()
        score_df.iloc[:, 1:] = scores['left_emos'].iloc[:, 1:] - scores['right_emos'].iloc[:, 1:]
        scores['FAsy_emos'] = score_df
        score_df = scores['left_vai'].copy()
        score_df.iloc[:, 1:] = scores['left_vai'].iloc[:, 1:] - scores['right_vai'].iloc[:, 1:]
        scores['FAsy_vai'] = score_df
        
    # save scores
    with open(os.path.join(score_dir, f'scores_{fer_model}.pkl'), 'wb') as f:
        pickle.dump(scores, f)
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Facial expression scored video {align_dir.split('/')[-1]}")
    logging.info(f"Facial expression scored video {align_dir.split('/')[-1]} (PID: {os.getpid()})")
    
def multi_facial_expression_analysis(video_ids_chunk, dev):
    for idx, vid_id in enumerate(video_ids_chunk):
        align_dir = os.path.join(align_root, vid_id)
        compo_dir = os.path.join(compo_root, vid_id)
        score_dir = os.path.join(score_root, vid_id)
        try:
            facial_expression_analysis(align_dir, compo_dir, score_dir, fer_model, dev)
        except Exception as e:
            logging.error(e)
        

if __name__ == '__main__':

    ## parse arguments
    parser = argparse.ArgumentParser()
    # directories
    parser.add_argument('--video-root', type=str, help='Specify the directory containing the videos.')
    parser.add_argument('--align-root', type=str, help='Specify the directory containing the aligned face images.')
    parser.add_argument('--compo-root', type=str, help='Specify the directory containing the composite face images.')
    parser.add_argument('--score-root', type=str, help='Specify the directory for saving information and scores.')
    # global setting
    parser.add_argument('--gpus', type=str, help='Specify the gpu indices for EmoNet.')
    # facial expression analysis
    parser.add_argument('--fer-model', type=str, choices=['ferplus', 'emonet'], help="Specify the model for emotion analysis. Available model: 'ferplus' and 'emonet'.")

    args = parser.parse_args()

    if args.gpus is not None:
        device = args.gpus.split(',') # a list of gpus
        device_name = f"gpu{args.gpus.replace(',','')}"
    else:
        device = 'cpu'
        device_name = 'cpu'
    video_root = args.video_root
    align_root = args.align_root
    compo_root = args.compo_root
    score_root = args.score_root
    fer_model = args.fer_model
    if fer_model == 'ferplus':
        cv2.setNumThreads(1)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file = f"logs/{align_root.split('/')[1].split('_')[0]}_emotion_{fer_model}_{device_name}.log"
    open(log_file, 'w').close()
    logging.basicConfig(filename=log_file,
                        level=logging.INFO, 
                        format='[%(asctime)s] [%(levelname)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a+')
    install_mp_handler()
    logging.info(f'Devices: {device}, Emotion model: {fer_model}')
    print(f'Devices: {device}, Emotion model: {fer_model}')
    
    # check device and model
    if fer_model == 'ferplus':
        assert device == 'cpu', "Ferplus should run with cpu!"

    video_ids = os.listdir(video_root)
    video_ids = sorted([x.split('.')[0] for x in video_ids])

    ## 4. facial expression
    fer_model = args.fer_model
    if device == 'cpu':
        inp_args = []
        for idx, vid_id in enumerate(video_ids):
            align_dir = os.path.join(align_root, vid_id)
            compo_dir = os.path.join(compo_root, vid_id)
            score_dir = os.path.join(score_root, vid_id)
            # if os.path.exists(os.path.join(score_dir, f'scores_{fer_model}.pkl')):
            #     continue
            inp_args.append((align_dir, compo_dir, score_dir, fer_model, device))
        if len(inp_args) > 0:
            try:
                with Pool(os.cpu_count()) as pool:
                    pool.starmap(facial_expression_analysis, inp_args)
                    pool.close()
                    pool.join()
            except Exception as e:
                print(traceback.format_exc())
                logging.error(e)
    else:
        if len(device) == 1:
            list_video_ids = [video_ids]
        else:
            chunk_size = len(video_ids) // len(device) + 1
            list_video_ids = [video_ids[i:i+chunk_size] for i in range(0, len(video_ids), chunk_size)]
        procs = []
        for idx, video_ids_chunk in enumerate(list_video_ids):
            dev = f'cuda:{device[idx]}'
            print(f'{dev} will process {len(video_ids_chunk)} videos.')
            p = Process(target=multi_facial_expression_analysis, args=(video_ids_chunk, dev))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()
                                
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 4: analyze facial expression **********")
    logging.info('********** Finish step 4: analyze facial expression **********')
