import os
import sys
import time
import glob
import pickle
import argparse
import traceback
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm

import librosa
import librosa.display
import soundfile as sf
import statsmodels.api as sm
import crepe
import parselmouth
import statistics

from pathlib import Path
from datetime import datetime
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.stats import norm
from parselmouth.praat import call

## global constants
male_mean, male_std = 111.74, 15.59
female_mean, female_std = 200.53, 16.86

## 1. Extract audio track from video
def extract_audio(video_root, audio_root, start=10, end=20):
    
    '''
    Extract audio segments from videos and save as .wav files.
    
    :param video_root: directory saving videos
    :param audio_root: directory for saving audios
    :param start: start time of subclip
    :param end: end time of subclip
    '''
    
    video_files = glob.glob(os.path.join(video_root, '*.mp4'))
    Path(audio_root).mkdir(parents=True, exist_ok=True)
    
    for v in video_files:
        clip = AudioFileClip(v).subclip(start, end)
        clip.write_audiofile(os.path.join(audio_root, f"{os.path.basename(v).split('.')[0]}.wav"))
    
## 2. Convert stereo sound to mono sound
def stereo_to_mono(audio_root, mono_root):
    
    '''
    Convert stereo audios to mono audios and save mono audios as .wav file.
    
    :param audio_root: directory saving audios
    :param mono_root: directory for saving mono audios
    '''
    
    audio_files = glob.glob(os.path.join(audio_root, '*.wav'))
    Path(mono_root).mkdir(parents=True, exist_ok=True)
    
    for a in audio_files:
        sound = AudioSegment.from_wav(a)
        monosound = sound.set_channels(1)
        monosound.export(os.path.join(mono_root, f"{os.path.basename(a).split('.')[0]}.wav"), format='wav')

## 3. Separate human voice by REPET-SIM
# credit: https://librosa.org/librosa_gallery/auto_examples/plot_vocal_separation.html
def separate_voice_(audio_file, voice_file, plot_file=None):
    
    y, sr = librosa.load(audio_file, sr=44100) # audio signal
    S_full, phase = librosa.magphase(librosa.stft(y)) # spectrogram

    # median filter on similar time-frequency bins
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr))) 
    S_filter = np.minimum(S_full, S_filter)

    # soft-masks
    margin_i, margin_v = 2, 10
    power = 2
    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)
    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # foreground (voice) and background (music) spectrograms
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    # istft
    fore_signal = librosa.istft(S_foreground * phase)
    back_signal = librosa.istft(S_background * phase)

    # write separated audio signals
    sf.write(voice_file, fore_signal, sr)
    
    if plot_file is not None:
        display_separation_(plot_file, S_full, S_background, S_foreground, sr)
                         
def display_separation_(plot_file, S_full, S_background, S_foreground, sr=44100):
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Full spectrum')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_background, ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Background')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(S_foreground, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.title('Foreground')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(plot_file)
    
def separate_voice(audio_root, voice_root, plot_root=None):
    
    '''
    Separate human voice from mixture audio and save voice audio as a .wav file.
    
    :param audio_root: directory saving mono audios
    :param voice_root: directory for saving voice audios
    :param plot_root: if specified, save spectrograms
    '''
    
    if voice_root:
        Path(voice_root).mkdir(parents=True, exist_ok=True)
    if plot_root:
        Path(plot_root).mkdir(parents=True, exist_ok=True)
        
    audio_files = glob.glob(os.path.join(audio_root, '*.wav'))
    for audio_file in audio_files:
        video_id = os.path.basename(audio_file).split('.')[0]
        voice_file = os.path.join(voice_root, f'{video_id}.wav')
        if plot_root:
            plot_file = os.path.join(plot_root, f'{video_id}_separation.png')
        separate_voice_(audio_file, voice_file, plot_file)
        print(f'Video {video_id}')

## 4. Measure pitch and other acoustic values using autocorrelation (parselmouth) or neural network (crepe)
# credit: https://parselmouth.readthedocs.io/en/stable/examples/plotting.html
def draw_spectrogram_(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_pitch_(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

def measure_pitch_parselmouth(voice_root, score_root, pitch_root=None, plot_root=None):
    
    '''
    Measure pitch using 'parselmouth' package and save estimated pitch of each voice audio in a dataframe.
    
    :param voice_root: directory root saving voice audios
    :param score_root: directory root for saving pitch estimates
    :param pitch_root: if specified, save pitch track for each audio 
    :param plot_root: if specified, plot pitch track for each audio
    '''
    
    voice_files = glob.glob(os.path.join(voice_root, '*.wav'))
    voice_files = sorted(voice_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    pitch_est = {k: [] for k in ['Video_ID', 'pitch']}
    if score_root:
        Path(score_root).mkdir(parents=True, exist_ok=True)
    if pitch_root:
        Path(pitch_root).mkdir(parents=True, exist_ok=True)
    if plot_root:
        Path(plot_root).mkdir(parents=True, exist_ok=True)
    
    for v in voice_files:
        video_id = v.split('/')[-1].split('.')[0]
        snd = parselmouth.Sound(v)
        pitch_track = snd.to_pitch()
        freq = pitch_track.selected_array['frequency']
        est_pitch = np.median(freq[(freq < 300) & (freq > 60)])
        pitch_est['Video_ID'].append(video_id)
        pitch_est['pitch'].append(est_pitch)
        if pitch_root is not None:
            freq_df = pd.DataFrame(np.vstack([np.arange(len(freq)) / 100, freq]).T, columns=['time', 'freq'])
            freq_df.to_csv(os.path.join(pitch_root, f'{video_id}_parselmouth.csv'), sep='\t', index=False)
        if plot_root is not None:
            pitch = snd.to_pitch()
            # If desired, pre-emphasize the sound fragment before calculating the spectrogram
            pre_emphasized_snd = snd.copy()
            pre_emphasized_snd.pre_emphasize()
            spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
            plt.figure()
            draw_spectrogram_(spectrogram)
            plt.twinx()
            draw_pitch_(pitch)
            plt.xlim([snd.xmin, snd.xmax])
            plt.savefig(os.path.join(plot_root, f'{video_id}_parselmouth.png'))
        print(f'Video {video_id}')

    pitch_est = pd.DataFrame.from_dict(pitch_est)
    pitch_est.to_csv(os.path.join(score_root, f'pitch_est_parselmouth.csv'), sep='\t', index=False)
    
# credit: https://github.com/marl/crepe
def measure_pitch_crepe(voice_root, score_root, pitch_root=None, plot_root=None, conf_thres=0.5):
    
    '''
    Measure pitch using 'crepe' package and save estimated pitch of each voice audio in a dataframe.
    
    :param voice_root: directory saving voice audios
    :param score_root: directory for saving pitch estimates
    :param pitch_root: if specified, save pitch track for each audio 
    :param plot_root: if specified, plot pitch track for each audio
    :param conf_thres: confidence threshold for estimating pitch
    '''

    voice_files = glob.glob(os.path.join(voice_root, '*.wav'))
    voice_files = sorted(voice_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    pitch_est = {k: [] for k in ['Video_ID', 'pitch']}
    if score_root:
        Path(score_root).mkdir(parents=True, exist_ok=True)
    if pitch_root:
        Path(pitch_root).mkdir(parents=True, exist_ok=True)
    if plot_root:
        Path(plot_root).mkdir(parents=True, exist_ok=True)
        
    for v in voice_files:
        video_id = v.split('/')[-1].split('.')[0]
        sr, audio = wavfile.read(v)
        time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size=10)
        est_pitch = np.median(pitch_df.loc[(pitch_df.freq > 60) & (pitch_df.freq < 300) & (pitch_df.conf > conf_thres), 'freq'].values)
        pitch_est['Video_ID'].append(video_id)
        pitch_est['pitch'].append(est_pitch)
        if pitch_root is not None:
            freq_df = pd.DataFrame(np.vstack([time, frequency, confidence]).T, columns=['time', 'freq', 'conf'])
            freq_df.to_csv(os.path.join(pitch_root, f'{video_id}_crepe.csv'), sep='\t', index=False)
        if plot_root is not None:
            # draw the low pitches in the bottom
            salience = np.flip(activation, axis=1)
            inferno = matplotlib.cm.get_cmap('inferno')
            image = inferno(salience.transpose())
            # attach a soft and hard voicing detection result under the salience plot
            image = np.pad(image, [(0, 20), (0, 0), (0, 0)], mode='constant')
            image[-20:-10, :, :] = inferno(confidence)[np.newaxis, :, :]
            image[-10:, :, :] = (inferno((confidence > 0.5).astype(np.float))[np.newaxis, :, :])
            plt.rcParams['axes.grid'] = False
            fig, ax = plt.subplots(figsize = (8, 3), dpi=150)
            ax.imshow(image)
            plt.savefig(os.path.join(plot_root, f'{video_id}_crepe.png'))
        print(f'Video {video_id}')
            
    pitch_est = pd.DataFrame.from_dict(pitch_est)
    pitch_est.to_csv(os.path.join(score_root, f'pitch_est_crepe.csv'), sep='\t', index=False)
  
# credit: https://github.com/drfeinberg/PraatScripts
def measure_speechrate_(filename, return_type=['speech_rate', 'dict']):
    silencedb = -25
    mindip = 2
    minpause = 0.3
    sound = parselmouth.Sound(filename)
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses = call(silencetable, "Get number of rows")
    speakingtot = 0
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

    intensity_matrix = call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime) 
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object
    timecorrection = originaldur / intensity_duration

    # Insert voiced peaks in TextGrid
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        call(textgrid, "Insert point", 1, position, "")

    # return results
    speakingrate = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot
    npause = npauses - 1
    asd = speakingtot / voicedcount
    
    if return_type == 'dict':
        speechrate_dictionary = {'soundname':filename,
                                 'nsyll':voicedcount,
                                 'npause': npause,
                                 'dur(s)':originaldur,
                                 'phonationtime(s)':intensity_duration,
                                 'speechrate(nsyll / dur)': speakingrate,
                                 "articulation rate(nsyll / phonationtime)":articulationrate,
                                 "speakingtime(s)": speakingtot,
                                 "ASD(speakingtime / nsyll)":asd}

        return speechrate_dictionary
    else:
        return speakingrate

# other acoustic measures
def measure_others_(sound_file, f0min=60, f0max=300, unit='Hertz'):
    
    # speech rate
    speechRate = measure_speechrate_(sound_file, 'speechRate')
    
    # duration, f0, hnr, jitter, shimmer
    sound = parselmouth.Sound(sound_file) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    intensity = sound.to_intensity(50)
    meanIntensity = call(intensity, "Get mean", 0, 0)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # formant dispersion
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
        
    # calculate median formants across pulses
    f1_median = statistics.median(f1_list)
    f4_median = statistics.median(f4_list)
    
    formantDispersion = (f4_median - f1_median) / 3
        
    return {'duration': duration,
            'meanF0': meanF0,
            'stdevF0': stdevF0,
            'speechRate': speechRate,
            'meanIntensity': meanIntensity,
            'hnr': hnr,
            'localJitter': localJitter,
            'localShimmer': localShimmer,
            'formantDispersion': formantDispersion}

def measure_others(voice_root, score_root):
    
    '''
    Measure other acoustice measures: speech rate, mean intensity, formant dispersion, vocal jitter, harmonic-to-noise ratio (hnr).
    
    :param voice_root: directory saving voice audios
    :param score_root: directory for saving other acoustice measures
    '''
    
    if score_root:
        Path(score_root).mkdir(parents=True, exist_ok=True)
    
    voice_files = glob.glob(os.path.join(voice_root, '*.wav'))
    voice_files = sorted(voice_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    all_others = pd.DataFrame(columns=['Video_ID', 'speech_rate', 'mean_intensity', 'formant_dispersion', 'jitter', 'hnr'])
    for i, v in enumerate(voice_files):
        vid = v.split('/')[-1].split('.')[0]
        others = measure_others_(v)
        all_others.loc[i] = [vid, others['speechRate'], others['meanIntensity'], others['formantDispersion'], others['localJitter'], others['hnr']]
        print(f'Video {vid}')

    # save
    all_others.to_csv(os.path.join(score_root, 'other_acoustic_measures.csv'), sep='\t', index=False)

## 5. Scale voice by gender
def voice_map_(score):
    cuts = norm.ppf(np.arange(0, 1, 0.2))
    for i in range(4):
        if score > cuts[i] and score <= cuts[i+1]:
            degree = i + 1
    if score > cuts[4]:
        degree = 5
    return degree

def scale_voice_by_gender(gender_est_file, pitch_est_file):
    
    '''
    Scale the voice gender from 1 (very masculine) to 5 (very feminine).
    
    :param gender_est_file: gender estimates file
    :param pitch_est_file: pitch estimates file
    :return pitch_gender: dataframe saving raw pitch estimates and scaled voice gender
    '''
    
    gender_est = pd.read_csv(gender_est_file, sep='\t')
    pitch_est = pd.read_csv(pitch_est_file, sep='\t')
    pitch_gender = pd.merge(gender_est, pitch_est, on='Video_ID').dropna()

    pitch_gender['pitch_std'] = 0
    pitch_gender.loc[pitch_gender.gender == 'Male', 'pitch_std'] = (pitch_gender.loc[pitch_gender.gender == 'Male'].pitch - male_mean) / male_std
    pitch_gender.loc[pitch_gender.gender == 'Female', 'pitch_std'] = (pitch_gender.loc[pitch_gender.gender == 'Female'].pitch - female_mean) / female_std
    pitch_gender['voice_gender'] = pitch_gender.pitch_std.map(voice_map_)
    
    return pitch_gender
    


if __name__ == '__main__':

    ## parse arguments
    parser = argparse.ArgumentParser()
    # directories/files
    parser.add_argument('--video-root', type=str, help='Specify the directory containing the videos.')
    parser.add_argument('--audio-root', type=str, help='Specify the directory for saving audio segments.')
    parser.add_argument('--mono-root', type=str, help='Specify the directory for saving mono audios.')
    parser.add_argument('--voice-root', type=str, help='Specify the directory for saving separeted voice audios.')
    parser.add_argument('--output-root', type=str, help='Specify the directory for saving result (pitch estimation).')
    parser.add_argument('--pitch-root', type=str, help='Specify the directory for saving estimated pitch track for each video.')
    parser.add_argument('--plot-root', type=str, help='Specify the directory for saving spectrograms and pitch tracks.')
    parser.add_argument('--gender-est', type=str, help='Specify the file containing gender estimates.')
    # parameters
    parser.add_argument('--subclip', type=str, default='10,20', help="Specify the start and end time of subclip to extract audio tracks, separete start and end by ','.")
    parser.add_argument('--pitch-package', type=str, choices=['parselmouth', 'crepe'], help="Package for measuring pitch, 'parselmouth' (recommended) or 'crepe'.")

    args = parser.parse_args()
    
    ## 1. Extract audio track from video
    if args.video_root is not None:
        
        assert args.audio_root is not None, 'Please specify the directory for saving audio segments!'
        
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 1: extract audio **********")
        subclip = args.subclip.split(',')
        extract_audio(args.video_root, args.audio_root, subclip[0], subclip[1])
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 1: extract audio **********")
  
    ## 2. Convert stereo sound to mono sound
    if args.audio_root is not None and args.mono_root is not None:
        
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 2: convert stereo sound to mono sound **********")
        stereo_to_mono(args.audio_root, args.mono_root)
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 2: convert stereo sound to mono sound **********")
    
    ## 3. Separate human voice
    if args.mono_root is not None and args.voice_root is not None:
        
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 3: separate human voice **********")
        separate_voice(args.mono_root, args.voice_root, args.plot_root)
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 3: separate human voice **********")
    
    ## 4. Measure pitch and other acoustic values
    if args.voice_root is not None and args.output_root is not None:
        
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 4.1: measure pitch (by {args.pitch_package}) **********")
        if args.pitch_package == 'parselmouth':
            measure_pitch_parselmouth(args.voice_root, args.output_root, args.pitch_root, args.plot_root)
        elif args.pitch_package == 'crepe':
            measure_pitch_crepe(args.voice_root, args.output_root, args.pitch_root, args.plot_root)
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 4.1: measure pitch (by {args.pitch_package}) **********")
        
        # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 4.2: measure other acoustic values **********")
        # measure_others(args.voice_root, args.output_root)
        # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 4.2: measure other acoustic values **********")

    ## 5. Scale voice by gender
    if args.gender_est is not None and args.output_root is not None:
        
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin step 5: scale voice by gender **********")
        pitch_est_file = os.path.join(args.output_root, f'pitch_est_{args.pitch_package}.csv')
        voice_gender = scale_voice_by_gender(args.gender_est, pitch_est_file)
        voice_gender.to_csv(os.path.join(args.output_root, f'voice_gender_{args.pitch_package}.csv'), sep='\t')
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Finish step 5: scale voice by gender **********")
