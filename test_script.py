import os

root = 'test' # directory root for saving results
prefix = 'freq1' # prefix indicating extraction frequency
extract_freq = 1 # extracting frequency, '-1' means extracting all frames
fer_model = 'emonet' # facial expression analysis model

# extract frames
step1 = f"python main_extraction.py \
    --video-root x_video/ \
    --frame-root {root}/{prefix}_frames/ \
    --extract-freq {extract_freq}"

# detect, recognize and align faces
step2 = f"python main_recognition.py \
    --video-root x_video/ \
    --frame-root {root}/{prefix}_frames/ \
    --align-root {root}/{prefix}_aligned/ \
    --compo-root {root}/{prefix}_composite/ \
    --score-root {root}/{prefix}_scores/ \
    --pre-filter-file {root}/{prefix}_pre_filter.txt \
    --quiet"

# detect 3d landmarks and estimate euler angles
step3 = f"python main_3d_landmarks.py \
    --video-root x_video/ \
    --frame-root {root}/{prefix}_frames/ \
    --score-root {root}/{prefix}_scores/ \
    --device cpu"
    
# analyze emotions    
step4 = f"python main_emotion.py \
    --video-root x_video/ \
    --align-root {root}/{prefix}_aligned/ \
    --compo-root {root}/{prefix}_composite/ \
    --score-root {root}/{prefix}_scores/ \
    --gpus 2,3,5,6 \
    --fer-model {fer_model}"

# os.system(step1)
# os.system(step2)
# os.system(step3)
# os.system(step4)
