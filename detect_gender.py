# credit: https://github.com/smahesh29/Gender-and-Age-Detection
import os
import sys
import time
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import random
import cv2
from pathlib import Path

genderProto="gender_age_models/gender_deploy.prototxt"
genderModel="gender_age_models/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList=['Male','Female']

genderNet=cv2.dnn.readNet(genderModel,genderProto)

def detect_gender(frame):
    
    # use aligned faces
    face=frame
    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    
    return gender, genderPreds[0][0]


if __name__ == '__main__':
    
    ## parse arguments
    parser = argparse.ArgumentParser()
    # directories/files
    parser.add_argument('--video-root', type=str, help="Specify the directory containing candidate videos required to detect speaker's gender.")
    parser.add_argument('--output-root', type=str, help='Specify the directory for saving detect genders.')
    parser.add_argument('--nsamples', type=int, default=100, help='Number of images for detecting gender.')
    
    args = parser.parse_args()
    
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    
    ## detect gender
    print(f'***** Begin detect gender (using {args.nsamples} samples for each video) *****')
    videos = os.listdir(args.video_root)
    all_genders = dict.fromkeys(sorted(videos, key=lambda x: int(x)))
    for vid in videos:
        fimgs = glob.glob(os.path.join(args.video_root, f'{vid}/*.jpg'))
        if len(fimgs) == 0:
            continue
        genders = []
        random.shuffle(fimgs)
        sample_fimgs = fimgs[:args.nsamples]
        for fimg in sample_fimgs:
            frame_id = fimg.split('/')[-1].split('.')[0]
            img = cv2.imread(fimg)
            gender, male_prob = detect_gender(img)
            genders.append([frame_id, gender, male_prob])
        genders = np.array(genders)
        all_genders[vid] = pd.DataFrame(genders, columns=['Frame_ID', 'gender', 'male_prob'])
        print(f'Finish detecting gender: video {vid}')

    # save
    with open(os.path.join(args.output_root, 'all_genders.pkl'), 'wb') as f:
        pickle.dump(all_genders, f)

    ## gender estimation
    gender_est = {k: [] for k in ['Video_ID', 'gender']}
    for vid, df in all_genders.items():
        gender_est['Video_ID'].append(vid)
        if df is None:
            gender_est['gender'].append(None)
            continue
        if sum(df.gender == 'Male') / len(df) >= 0.5:
            gender = 'Male'
        else:
            gender = 'Female'
        gender_est['gender'].append(gender)
    gender_est = pd.DataFrame.from_dict(gender_est)
    # save
    gender_est.to_csv(os.path.join(args.output_root, 'gender_est.csv'), sep='\t', index=False)
    print('***** Saved estimated genders for all videos *****')