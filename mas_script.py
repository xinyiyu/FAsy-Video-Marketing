import os

## detect gender (using aligned images)
gender_detect = 'python detect_gender.py \
    --video-root test/all_aligned \
    --output-root gender\
    --nsamples 100'

## voice gender
# measure voice masculinity
extract_audio = 'python voice_masculinity.py \
    --video-root x_video \
    --audio-root gender/audio \
    --subclip 10,20'

# convert stereo sound to mono sound
stereo_to_mono = 'python voice_masculinity.py \
    --audio-root gender/audio \
    --mono-root gender/mono'

# separate human voice
separate_voice = 'python voice_masculinity.py \
    --mono-root gender/mono \
    --voice-root gender/voice \
    --plot-root gender/plot'

# measure pitch and other acoustic values
measure_pitch = 'python voice_masculinity.py \
    --voice-root gender/voice \
    --output-root gender \
    --pitch-root gender/pitch \
    --plot-root gender/plot \
    --pitch-package parselmouth'

# scale voice by gender
scale_voice = 'python voice_masculinity.py \
    --output-root gender \
    --gender-est gender/gender_est.csv \
    --pitch-package parselmouth'

## face gender
face_gender = 'python face_masculinity.py \
    --frame-root test/all_frames \
    --score-root test/all_scores \
    --align-root gender/align \
    --output-root gender \
    --gender-est gender/gender_est.csv \
    --all-landmarks gender/all_mesh_landmarks.pkl'

os.system(scale_voice)