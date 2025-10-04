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
import soundfile as sf
from utils import *
from transformers import pipeline


# Load model and processor
audio_classification_pipe = pipeline("audio-classification", 
                                     model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

facial_model_name = "prithivMLmods/Facial-Emotion-Detection-SigLIP2"
facial_model = SiglipForImageClassification.from_pretrained(facial_model_name)
facial_processor = AutoImageProcessor.from_pretrained(facial_model_name)
facial_labels = {
                 "0": "Ahegao", "1": "Angry", "2": "Happy", "3": "Neutral",
                 "4": "Sad", "5": "Surprise"
}

whisper_model = whisper.load_model("small")


# --------- Utility Functions ---------

def update_memory(messages, resp):
    """Updates messages with model response from Ollama"""
    last_output = dict(resp['message'])['content']
    messages.append({"role": "assistant", "content": last_output})
    return messages

def facial_emotion_classification_batch(images):   # shape (B, H, W, C)
    pil_images = [Image.fromarray(img).convert("RGB") for img in images]
    inputs = facial_processor(images=pil_images, return_tensors="pt")

    # Model inference
    with torch.no_grad():
        outputs = facial_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)  # shape: [batch_size, num_classes]

    # Convert to list of dicts
    batch_predictions = []
    for prob in probs:
        batch_predictions.append({facial_labels[str(i)]: round(prob[i].item(), 3) for i in range(len(facial_labels))})
    
    return batch_predictions

def audio_emotion_classification(audios):
    results = audio_classification_pipe(audios)
    predicted_labels = []
    for audio_result in results:
        top_prediction = max(audio_result, key=lambda x: x['score'])
        predicted_labels.append(top_prediction['label'])
    return predicted_labels

def audio_transcribe(filepath):
    result = whisper_model.transcribe(filepath, verbose=True)
    return [{"text": segment['text'], "interval": (segment['start'], segment['end'])} for segment in result["segments"]]
    

# Global Varibles
CURRENT_TRANSCRIPTION = False


# --------- Core Module Functions ---------

MODEL_NAME = "gemma3:4b"  # replace with your model

def read_annotated_message(messages, annotated_message):
    messages.append({"role": "user", "content": f"Below I will provide a piece of text message annotated with emotions enclosed in \
    brackets []. For example, 'Hi, welcome! [Happy]'. Please read the annotated text first, then be ready for some modifications. In \
    the next message that I sent you, the message will be provided"})
    
    resp = ollama.chat(model=MODEL_NAME, messages=messages)
    update_memory(messages, resp)

    messages.append({"role": "user", "content": annotated_message})

    resp = ollama.chat(model=MODEL_NAME, messages=messages)
    update_memory(messages, resp)
    return messages



def prompt_input(messages, prompt):
    """Generate any inference from any prompt"""
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(model=MODEL_NAME, messages=messages)
    update_memory(messages, resp)
    return messages
