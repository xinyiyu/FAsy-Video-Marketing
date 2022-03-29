# Facial asymmetry measurement
This repository implements facial asymmetry measurement part of "Asymmetry of Facial Expressions and Emotional Labor in Video Marketing".

## Installation
1. Git clone the repository and create an environment:

    ```
    git clone https://github.com/xinyiyu/FAsy-Video-Marketing.git
    cd FAsy-Video-Marketing
    conda create --name FAVM python
    conda activate FAVM
    pip install -r requirements.txt
    ```

2. Check the installation status:
    
    ```
    python main_extraction.py -h
    ```
    
3. Move the video folder (e.g `x_xideo/`) into this directory.

## Usage
Take processing videos in `x_video/` as an example. All the processed images as well as scores will be saved in `results/`. `test_script.py` shows example commands. 

1. Step 1: Extract frames from each video. Sample videos are in `./x_video`. Specify the extraction frequency and directory for saving frames and run `main_extraction.py`. For example, extract one frame per second:

    ```
    python main_extraction.py \
        --video-root x_video/ \
        --frame-root results/freq1_frames/ \
        --extract-freq 1
    ```
    
    A directory named `freq1_frames/` will be created which contains subdirectories for all videos.
    
    In this step, videos are processed in a loop. For processing 33 sample videos, it takes ~3 minutes for extracting 1 frame per second and ~40 minutes for extracting all frames.
    
2. Step 2: Recognize, align and compose faces. Specify the directory of extracted frames in step 1 and directories for saving aligned and composite faces. Specify the directory for saving 2D 68 facial landmarks and 128D face encodings. Also specify a file for saving comments on videos potentially to be filtered. Run `main_extraction.py`.
    
    ```
    python main_recognition.py \
        --video-root x_video/ \
        --frame-root results/freq1_frames/ \
        --align-root results/freq1_aligned/ \
        --compo-root results/freq1_composite/ \
        --score-root results/freq1_scores/ \
        --pre-filter-file results/freq1_pre_filter.txt \
        --verbose
    ```
    
    Directories `freq1_aligned/`, `freq1_composite/`, `freq1_scores/` which contain subdirectories for all videos and a file `freq_pre_filter.txt` will be created.
    
    For each frame, face locations are detected by [dlib.get_frontal_face_detector()](http://dlib.net/python/index.html#dlib.get_frontal_face_detector) which is based on histogram of oriented gradients (HOG). Additionally, the 2D 68 facial landmarks will be detected by [dlib.shape_predictor](http://dlib.net/python/index.html#dlib.shape_predictor). Then the original image together with the 68 facial landmarks will be passed to [dlib.face_recognition_model_v1.compute_face_descriptor()](http://dlib.net/python/index.html?highlight=compute_face_descriptor#dlib.face_recognition_model_v1.compute_face_descriptor) which will output a 128D encoding for the face. 
    
    After all faces are detected and encoded, we need to select some faces as benchmarks of the target in each video. There may be some faces of other non-target persons or static images of the target and there is no label for any face, but it is reasonable to assume that most of the faces detected in one video are target's, whose encodings are closer to each other than to other people's faces. Instead of only choosing the faces closest to the global mean of all faces as the labeled face (this strategy may fail when there are a few non-target faces whose encodings are far from target faces), we can perform k-means clustering (2 clusters). If the minimum distance between 2 clusters are no greater than a threshold (e.g 0.6), choose two faces closest to each cluster as well as the face closest to the global mean. This strategy can also be benificial when only one person appears in a video because in this case we allow some variation of target's face by including faces a bit far from the global mean. Otherwise, choose two faces closest to the larger cluster as benchmarks (since when the two clusters are largely seperated, it is likely that the smaller cluster is composed of other people's faces).
    
    Having selected benchmark target faces, we next perform face recognition, alignment and composition: for any face, if the closest distance between its encoding and the labeled encodings is smaller than 0.6 (0.6 is suggested by [face_recognition](https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py)), then it will be recognized as target face, and the face will be aligned and composed along the way. The face is aligned by rotating and translating the original frame to make the line connecting eyes horizontal and its center in the middle of x axis. The size of aligned and composite face image is (256, 256). The aligned and composite faces are saved under `freq1_aligned/`, `freq1_composite/`, respectively.
    
    The information of each frame will be saved in `frame_2d_68pts_encodings.pkl`, a dictionary where each key is frame ID and value is a dictionary saving face information of each frame: 
    
    ```
    {'frame0_msec0': { 
        'num_faces': number of faces detected in the frame (0, 1, 2, ...).
        'has_target': whether this frame contains target (1) or not (0).
        'target_idx': index of the target face among all the detected faces in the frame (0, 1, 2, ...). If no face detected, this value will be `None`.
        'locations': a list of face locations in the frame. Each location is a tuple `(top, right, bottom, left)`.
        '2d_landmarks': a list of 2D 68 landmarks in the frame. Each 2D landmark set is a `(68, 2) numpy.ndarray`.
        'encodings': a list of encodings in the frame. Each encoding is a `(128,) numpy.ndarray`.}, 
    'frame30_msec1000': {...}, 
    'frame60_msec2000': {...}, 
    ...}
    ```
    
    And `frame_info.csv` recording 'num_faces' and 'has_target' will be generate for the convenience of previewing. In addition, `freq1_pre_filter.txt` will record some information of candidate videos potentially to be excluded in the modeling procedure. Specifically, these conditions will be checked: (a) whether a video contains too few faces (e.g 10); (b) whether a video contains too few target faces (e.g 10); (c) whether the minimum distance between recognized target faces is too large (e.g 0.6); (d) whether the maximum distance between target encodings is too small (e.g 0.2).
    
    In this step, videos will be processed in parallel while frames of each video will be processed in a loop. For the 33 sample videos, it takes ~30 minutes for face recognition.
    

3. Step 3: Estimate Euler angles using `3DDFA` model. Specify the directory of extracted frames in step 1 and the score directory created in step 2 for saving Euler angles, 3DDFA parameters and 3D landmarks. The device should be 'cpu' or 'cuda:<0-9>'. Run `main_3d_landmarks.py`. 

    ```
    python main_3d_landmarks.py \
        --video-root x_video/ \
        --frame-root results/freq1_frames/ \
        --score-root results/freq1_scores/ \
        --device cuda:0 
    ```
    
    One set of Euler angles consist of 3 angles: `alpha` -- angle of clockwise rotation around x axis; `beta` -- angle of anti-clockwise rotation around y axis; `gamma` -- angle of anti-clockwise rotation around z axis. The Euler angles can be used in the post processing. For example, discard faces with large `alpha` (profile face). `target_frame_euler.csv` records Euler angles of all frames, `target_frame_dfa_params.csv` records 3DDFA parameters, `target_frame_3d_landmarks.pkl` records 3D landmarks and they will be saved under `freq1_scores/`.
    
    In this step, videos are processed in a loop. For processing 33 sample videos, it takes ~8 minutes for estimating Euler angles of all frames (the speeds of CPU and GPU are similar). Batch size probably need to be adjusted according to memory (batch size = 128 works for the server I used). 
    
4. Step 4: Classify facial expression and measure facial asymmetry. Specify the directory of aligned, composite faces and the score directory created in step 2 for saving emotion scores. Specify the model for analyze facial expression: 'ferplus' or 'emonet'. For 'ferplus', use CPU; for 'emonet', specify GPU device IDs otherwise use CPU. Run `main_emotion.py`.

    ```
    python main_emotion.py \
        --video-root x_video/ \
        --align-root results/freq1_aligned/ \
        --compo-root results/freq1_composite/ \
        --score-root results/freq1_scores/ \
        --gpus 2,3 \
        --fer-model emonet \
        --verbose
    ```
    
    The network ('ferplus') trained on [FER+ dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) will output the probabilities of 8 emotions saved in `scores_ferplus.pkl`, which compresses a dictionary of 4 tables: 'align_emos', 'left_emos', 'right_emos', 'FAsy_emos'. The network ('emonet') trained on [AffectNet dataset](http://mohammadmahoor.com/affectnet/) will output the probabilities of 8 emotions as well as the valence-arousal score saved in `scores_emonet.pkl`, which compresses a dictionary of 8 tables, each row of one table record the scores of one face (8 emotion probabilities or valence-arousal-intensity).
    
    ```
    'align_emos': 8 emotions probabilities of aligned face.
    'left_emos': 8 emotions probabilities of left-left composite face.
    'right_emos': 8 emotions probabilities of right-right composite face.
    'FAsy_emos': difference between 8 emotions probabilities of left-left composite face and right-right composite face.
    'align_vai': (valence, arousal, intensity) of aligned face. Both valence and arousal scores are in range [0,1] and intensity^2=valence^2+arousal^2.
    'left_vai': (valence, arousal, intensity) of left-left composite face.
    'right_vai': (valence, arousal, intensity) of right-right composite face.
    'FAsy_vai': difference between valence, arousal, intensity of left-left composite face and right-right composite face.
    ```
    
    In this step, for 'ferplus', I use multiprocessing with CPU (~15 minutes for all frames); for 'emonet', both CPU and GPU (single or multiple) are supported, and videos will be process in a loop (GPU is recommended and it takes ~5 minutes for all frames with 4 GPUs and batch size 64). 
    
    Finally, the structure of `results/` folder will be:

    ```
    Structure:
    results/
    |-- freq1_pre_filter.txt
    |-- freq1_frames/
    |   |-- <video_1>/
    |   |   |-- frame0_msec0.jpg
    |   |   |-- frame1_msec1000.jpg
    |   |   |-- ...
    |   |-- ...
    |-- freq1_aligned/
    |   |-- <video_1>/
    |   |   |-- frame0_msec0.jpg
    |   |   |-- frame1_msec1000.jpg
    |   |   |-- ...
    |   |-- ...
    |-- freq1_composite/
    |   |-- <video_1>/
    |   |   |-- frame0_msec0_left.jpg
    |   |   |-- frame0_msec0_right.jpg
    |   |   |-- frame1_msec1000_left.jpg
    |   |   |-- frame1_msec1000_right.jpg
    |   |   |-- ...
    |   |-- ...
    |-- freq1_scores/
    |   |-- <video_1>/
    |   |   |-- frame_2d_68pts_encodings.pkl
    |   |   |-- frame_info.csv
    |   |   |-- target_frame_euler.csv
    |   |   |-- target_frame_dfa_params.csv
    |   |   |-- target_frame_3d_landmarks.pkl
    |   |   |-- scores_emonet.pkl
    |   |   |-- scores_ferplus.pkl
    |   |-- ...
    ```

## Result
Some result analysis can be found in `result_analysis.ipynb`.

## References
[1] face_recognition: https://github.com/ageitgey/face_recognition \
[2] 3DDFA: https://github.com/cleardusk/3DDFA \
[3] Emotion FERPlus: https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus \
[4] EmoNet: https://github.com/face-analysis/emonet
