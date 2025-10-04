#!/usr/bin/env python3
import ast
import json
import os
import ollama
import whisper
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch
import numpy as np
import shutil
import soundfile as sf
from scipy.io import wavfile
from moviepy.editor import VideoFileClip
import random



def trim_audio(audio_array: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    # Convert seconds to sample indices
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    
    # Ensure indices are within bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_array), end_sample)
    
    # Slice the audio
    trimmed_audio = audio_array[start_sample:end_sample]
    
    return trimmed_audio

def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_audio_segment(segment_array, sr, path):
    wavfile.write(path, sr, segment_array)


def save_video_segment(video_path, start_sec, end_sec, out_path):
    with VideoFileClip(video_path) as clip:
        subclip = clip.subclip(start_sec, end_sec)
        subclip.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)


def sample_frames(video_path, count):
    """Randomly sample `count` frames from a video as numpy arrays (H,W,C)."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    # Pick random timestamps within the clip duration
    times = sorted([random.uniform(0, duration) for _ in range(count)])
    
    frames = [clip.get_frame(t) for t in times]  # returns np.array(H, W, C)
    clip.close()
    return frames

def video_to_audio_array(video_path):
    """Extract full audio from video segment as a mono numpy array and sample rate."""
    clip = VideoFileClip(video_path)
    audio = clip.audio
    if audio is None:
        clip.close()
        return None, None

    audio_path = "tmp/_temp_audio.wav"
    audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)

    sr, audio_array = wavfile.read(audio_path)
    clip.close()
    os.remove(audio_path)

    # Convert to mono if stereo/multi-channel
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1).astype(np.int16)

    return sr, audio_array


def merge_emotions_with_transcriptions(segment_file, transcription_file):
    # Load JSON files
    with open(segment_file, 'r') as f:
        segments = json.load(f)
        
    with open(transcription_file, 'r') as f:
        transcriptions = json.load(f)
    
    merged_text = []
    
    for segment, transcription in zip(segments, transcriptions):
        text = transcription.get("text", "")
        
        facial_emotion = segment.get("facial")
        audio_emotion = segment.get("audio")
        
        # Build the bracket content
        if facial_emotion is None and audio_emotion is None:
            emotion_str = "[]"
        else:
            parts = []
            if facial_emotion is not None:
                parts.append(f"facial: {facial_emotion}")
            if audio_emotion is not None:
                parts.append(f"audio: {audio_emotion}")
            emotion_str = f"[{', '.join(parts)}]"
        
        merged_text.append(f"{text}{emotion_str}")
    
    return "".join(merged_text)



