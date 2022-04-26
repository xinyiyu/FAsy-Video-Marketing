import os
threads = '1'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

import sys
import time
import pickle
import shutil
import argparse
import traceback
import numpy as np
import pandas as pd
import random
import cv2
import dlib
import scipy.io as sio
from skimage import io
from datetime import datetime

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from utils import *
from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

## detect -> label -> recognize, align & compose: save aligned, composed faces and frame information
def recognize_align_compose(frame_dir, align_dir, compo_dir, score_dir, filt_info_file,
                            cluster_n=2, thres_dist=0.6, min_faces=10,
                            min_targets=10, min_variation=0.2, min_dist=0.6, verbose=True):
    '''
    (1) Detect all face locations, 68 2d landmarks in all frames and encode faces.
        Record information of each frame:
        {'num_faces': 0/1/2, 'has_target': 0/1, 'target_idx': None/0/1/2, 'locations': [(top, right, bottom, left)], '2d_landmarks': [ np.ndarray(68, 2)], 'encodings': [ np.ndarray(128,)]}
    (2) Label images as known target reference images.
    (3) Preprocess: record videos should be excluded.
    (4) Recognize target faces. Along the way, align and compose faces then save images.
    (5) Save frame information: {'frame001': {...}, 'frame002': {...}, ...},
        save table: ['frame_id', 'num_faces', 'has_target'],
        save possibly excluded video ids.
    :param frame_dir: directory containing frames
    :param align_dir: directory for saving aligned face images
    :param compo_dir: directory for saving composite face images
    :param filt_info_file: file for saving non-candidate video information
    :param cluster_n: number of known images chosen in each cluster, default is 2.
    :param thres_dist: threshold distance for recogizing face as target, default is 0.6.
    :param min_faces: minimum number of faces detected from all frames, default is 10. Will no recognize target if very few faces.
    :param min_targets: miminum number of target faces, default is 10.
    :param min_variation: minimum variation of all target faces, smaller videos will be discarded, default is 0.2.
    :param mim_dist: minimum distance between two clusters (only target faces), larger videos will be discarded, default is 0.6.
    :param verbose: verbose the process or not
    :return: no return
    '''

    ## extract video id
    vid_id = frame_dir.split('/')[-1]

    ## save aligned and composite faces to
    if align_dir and not os.path.exists(align_dir):
        os.makedirs(align_dir)
    if compo_dir and not os.path.exists(compo_dir):
        os.makedirs(compo_dir)
    if score_dir and not os.path.exists(score_dir):
        os.makedirs(score_dir)

    ## frames
    frame_files = os.listdir(frame_dir)
    frame_files = [x for x in frame_files if '.jpg' in x]
    frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[0].replace('frame', '')))
    frame_ids = [x.split('.')[0] for x in frame_files]

    ## record
    frame_dict = dict.fromkeys(frame_ids)
    info_df = pd.DataFrame(columns=['frame_id', 'num_faces', 'has_target'])
    comment = ''

    # print(f'Begin detection {vid_id}')
    ## detect and encode
    for i, image_file in enumerate(frame_files):
        frame_info = dict.fromkeys(['num_faces', 'has_target', 'target_idx', 'locations', '2d_landmarks', 'encodings'])
        frame_id = image_file.split('.')[0]
        # original frame
        image = cv2.imread(os.path.join(frame_dir, image_file))
        # face located in the rectangle (0.2-0.3s): list of dlib rectangles
        face_locations = face_detector(image)
        # coordinates of 68 landmarks (0.004s): list of dlib full_object_detection objects
        landmarks = [pose_predictor_68_point(image, face_location) for face_location in face_locations]
        # face encodings (0.6-0.8s): list of (128,) arrays
        encodings = [np.array(face_encoder.compute_face_descriptor(image, landmark)) for landmark in landmarks]
        # record
        save_locations = [rect_to_css(x) for x in face_locations] # convert rectangle to (top, right, bottom, left)
        save_2d_landmarks = [landmarks_to_np(x) for x in landmarks] # convert dlib full_object_detection object to (68, 2) np.ndarray
        frame_info['num_faces'], frame_info['locations'], frame_info['2d_landmarks'], frame_info['encodings'] = \
            len(face_locations), save_locations, save_2d_landmarks, encodings
        if verbose and i % 100 == 0:
            print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Encoded {i} / {len(frame_files)}")
        frame_dict[frame_id] = frame_info
    if verbose:
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Finish face detection.")

    ## label images: closest to mean + k closest to mean of each cluster
    tot_num_faces = sum([v['num_faces'] for _, v in frame_dict.items()])
    if tot_num_faces >= min_faces:
        known_encodings, all_encodings = [], []
        for k, v in frame_dict.items():
            all_encodings.extend(v['encodings'])
        X = np.array(all_encodings)
        kmeans = KMeans(n_clusters=2).fit(X)
        y = kmeans.labels_
        large_id = 1 if sum(y) > len(y) / 2 else 0
        small_id = 1 - large_id
        mean_encoding = np.mean(X, axis=0)
        mean_encoding_l = np.mean(X[np.where(y == large_id)[0]], axis=0)
        mean_encoding_s = np.mean(X[np.where(y == small_id)[0]], axis=0)
        known_encodings.extend(X[np.argsort(np.linalg.norm(X[np.where(y == large_id)[0]] - mean_encoding_s, axis=1))[:cluster_n].tolist()])
        
        X_l = X[np.where(y == large_id)[0], :]
        X_s = X[np.where(y == small_id)[0], :]
        X_cluster = np.vstack([X_l, X_s])
        dist_cluster = euclidean_distances(X_cluster, X_cluster)
        min_between_dist_cluster = np.min(dist_cluster[X_l.shape[0]:, :X_l.shape[0]]) if X_l.shape[0] < X.shape[0] else 0
        # only use larger cluster if minimum distance between two clusters is greater than min_dist
        if min_between_dist_cluster <= min_dist:
            known_encodings.append(X[np.argmin(np.linalg.norm(X - mean_encoding, axis=1))])
            known_encodings.extend(X[np.argsort(np.linalg.norm(X[np.where(y == small_id)[0]] - mean_encoding_l, axis=1))[:cluster_n].tolist()])
        known_encodings = np.array(known_encodings)

        ## recognize
        encoding_list = [] 
        n = 0
        for i, (k, v) in enumerate(frame_dict.items()):
            # no faces detected
            if v['num_faces'] == 0:
                v['has_target'] = 0
            # have faces detected
            else:
                closest_dists, closest_dists_l, closest_dists_s = [], [], []
                for j, encoding in enumerate(v['encodings']):
                    if y[n+j] == large_id:
                        closest_dists_l.append(np.min(np.linalg.norm(known_encodings - encoding, axis=1)))
                    else:
                        closest_dists_s.append(np.min(np.linalg.norm(known_encodings - encoding, axis=1)))
                    closest_dists.append(np.min(np.linalg.norm(known_encodings - encoding, axis=1)))
                n += j + 1
                if len(closest_dists_l) > 0:
                    v['has_target'] = 1
                    v['target_idx'] = closest_dists.index(min(closest_dists_l))
                    encoding_list.append(np.insert(encoding, 0, 1))
                elif len(closest_dists_l) == 0 and len(closest_dists_s) > 0:
                    v['has_target'] = 1
                    v['target_idx'] = closest_dists.index(min(closest_dists_s))
                    encoding_list.append(np.insert(encoding, 0, 1))
                else:
                    v['has_target'] = 0
                    v['target_idx'] = None
                    encoding_list.append(np.insert(encoding, 0, 0))
                if v['target_idx'] is not None:
                    # align
                    image = cv2.imread(os.path.join(frame_dir, f'{k}.jpg'))
                    landmarks = v['2d_landmarks'][v['target_idx']]  # (68, 2)
                    aligned_face = face_align(image, landmarks)
                    # save aligned face
                    if align_dir:
                        cv2.imwrite(os.path.join(align_dir, f"{k}.jpg"), aligned_face)
                    # compose
                    left2, right2 = face_composite(aligned_face)
                    # save left2 and right2
                    if compo_dir:
                        cv2.imwrite(os.path.join(compo_dir, f"{k}_left.jpg"), left2)
                        cv2.imwrite(os.path.join(compo_dir, f"{k}_right.jpg"), right2)
            # record
            info_df.loc[i] = [k, v['num_faces'], v['has_target']]
        # non-candidate video info
        if sum(info_df['has_target']) < min_targets:
            comment = f"Less than {min_faces} target faces in all extracted frames."

        X_target = np.array(encoding_list)
        y_target = kmeans.labels_[np.where(X_target[:, 0] == 1)]
        X_target = X_target[np.where(X_target[:, 0] == 1)[0], 1:]
        X0_target = X_target[np.where(y_target == 0)[0], :]
        X1_target = X_target[np.where(y_target == 1)[0], :]
        X_cluster = np.vstack([X0_target, X1_target])
        dist_cluster = euclidean_distances(X_cluster, X_cluster)
        min_between_dist_cluster = np.min(dist_cluster[X0_target.shape[0]:, :X0_target.shape[0]]) if X0_target.shape[0] < X_target.shape[0] else 0
        if min_between_dist_cluster > min_dist:
            comment = f"Minimum distance between two target encodings is larger than {min_dist}"
        if np.max(dist_cluster) < min_variation:
            comment = f"Maximum distance between target encodings is smaller than {min_variation}"

    else:
        # some videos may not have faces detected
        for i, (k, v) in enumerate(frame_dict.items()):
            info_df.loc[i] = [k, v['num_faces'], None]
        # non-candidate video info
        comment = f"Less than {min_faces} faces in all extracted frames."

    # save frame info
    info_df.to_csv(os.path.join(score_dir, 'frame_info.csv'), sep='\t', index=False)
    # save all info
    with open(os.path.join(score_dir, 'frame_2d_68pts_encodings.pkl'), 'wb') as f:
        pickle.dump(frame_dict, f)
    # save non-candidate video info
    if len(comment) > 0:
        filt_info = f'{vid_id}: {comment}\n'
        with open(filt_info_file, 'a+') as f:
            f.write(filt_info)
            
    print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Recognized and aligned video {frame_dir.split('/')[-1]}")
    logging.info(f"Recognized and aligned video {frame_dir.split('/')[-1]} (PID: {os.getpid()})")

if __name__ == '__main__':

    ## parse arguments
    parser = argparse.ArgumentParser()
    # directories
    parser.add_argument('--video-root', type=str, help='Specify the directory containing the videos.')
    parser.add_argument('--frame-root', type=str, help='Specify the directory for saving extracted frames.')
    parser.add_argument('--align-root', type=str, help='Specify the directory for saving the aligned face images.')
    parser.add_argument('--compo-root', type=str, help='Specify the directory for saving the composite face images.')
    parser.add_argument('--score-root', type=str, help='Specify the directory for saving information and scores.')
    parser.add_argument('--pre-filter-file', type=str, help='Specify the file for saving the pre-filtered video information.')
    # global setting
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose inside each step.')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet inside each step.')

    args = parser.parse_args()

    video_root = args.video_root
    frame_root = args.frame_root
    align_root = args.align_root
    compo_root = args.compo_root
    score_root = args.score_root
    pre_filter_file = args.pre_filter_file
    open(pre_filter_file, 'w').close()
    if args.verbose:
        verbose = True
    elif args.quiet:
        verbose = False
    
    if not os.path.exists(align_root):
        os.makedirs(align_root)
    if not os.path.exists(compo_root):
        os.makedirs(compo_root)
    if not os.path.exists(score_root):
        os.makedirs(score_root)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file = f"logs/{frame_root.split('/')[1].split('_')[0]}_recognition_align.log"
    open(log_file, 'w').close()
    logging.basicConfig(filename=log_file,
                        level=logging.INFO, 
                        format='[%(asctime)s] [%(levelname)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a+')
    install_mp_handler()

    video_ids = os.listdir(video_root)
    video_ids = sorted([x.split('.')[0] for x in video_ids])

    ## 2. recognize, align and compose
    # if os.path.exists(pre_filter_file):
    #     os.remove(pre_filter_file)
    inp_args = []
    for idx, vid_id in enumerate(video_ids):
        frame_dir = os.path.join(frame_root, vid_id)
        align_dir = os.path.join(align_root, vid_id)
        compo_dir = os.path.join(compo_root, vid_id)
        score_dir = os.path.join(score_root, vid_id)
        # if os.path.exists(os.path.join(score_dir, 'frame_2d_68pts_encodings.pkl')):
        #     continue
        inp_args.append((frame_dir, align_dir, compo_dir, score_dir, pre_filter_file,
                         2, 0.6, 10, 10, 0.2, 0.6, verbose))
    if len(inp_args) > 0:
        try:
            with Pool(os.cpu_count()) as pool:
                pool.starmap(recognize_align_compose, inp_args)
                pool.close()
                pool.join()
        except Exception as e:
            print(e)
            logging.error(e)
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 2: recognize, align and compose **********")
        logging.info('********** Finish step 2: recognize, align and compose **********')
        
