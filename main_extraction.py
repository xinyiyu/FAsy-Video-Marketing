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
import cv2
from multiprocessing import Pool, Process
from datetime import datetime

## extract all frames: save all frames
def extract_all_frames(video_file, img_dir=None, freq=1):
    '''
    :param video_file: the path of video
    :param img_dir: directory for saving the extracted frames
    :param freq: number of frames per second, default is 1. If freq=-1, extract all frames.
    :return: no return
    '''
    if img_dir and not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = -1
    success = True
    while success:
        success, image = cap.read()
        # milisecond at current time point
        if success:
            count += 1
            if freq > 0 and count % int(fps/freq) != 0:
                continue
            msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if img_dir:
                cv2.imwrite(os.path.join(img_dir, f'frame{count}_msec{round(count*1000/fps)}.jpg'), image)
    cap.release()
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Extracted video {img_dir.split('/')[-1]}")
    logging.info(f"Extracted video {img_dir.split('/')[-1]}")
                

if __name__ == '__main__':

    ## parse arguments
    parser = argparse.ArgumentParser()
    # directories
    parser.add_argument('--video-root', type=str, help='Specify the directory containing the videos.')
    parser.add_argument('--frame-root', type=str, help='Specify the directory for saving extracted frames.')
    # frame extraction
    parser.add_argument('--extract-freq', type=int, help="Specify the frame extraction frequency. Should be a positive integer or -1 for extracting all frames.")

    args = parser.parse_args()

    freq = args.extract_freq
    video_root = args.video_root
    frame_root = args.frame_root
    
    if not os.path.exists(frame_root):
        os.makedirs(frame_root)
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    log_file = f"logs/{frame_root.split('/')[1].split('_')[0]}_frame_extraction.log"
    open(log_file, 'w').close()
    logging.basicConfig(filename=log_file,
                        level=logging.INFO, 
                        format='[%(asctime)s] [%(levelname)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a+')

    video_ids = os.listdir(video_root)
    video_ids = sorted([x.split('.')[0] for x in video_ids])

    ## 1. extract frames
    try:
        for idx, vid_id in enumerate(video_ids):
            vid_dir = os.path.join(video_root, vid_id)
            img_dir = os.path.join(frame_root, vid_id)
            if os.path.exists(img_dir) and len(os.listdir(img_dir)) != 0:
                continue
            extract_all_frames(f'{vid_dir}.mp4', img_dir, freq)
    except Exception as e:
        print(e)
        logging.error(e)
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 1: extract frames **********")
    logging.info('********** Finish step 1: extract frames **********')
    
