from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from models import *
import requests
import shutil
import ollama

app = Flask(__name__)
CORS(app)  # allow all origins for frontend JS

# Global state
server_status = {"busy": False}

# Define commands and corresponding functions
command_list = ["upload_recording", "transcribe", "emotion_from_video", "emotion_from_audio", "merge_emotions_with_LLM"]

url = "http://localhost:11434/api/chat"
model = "gemma3:4b" 
FACIAL_SAMPLES_COUNT = 3  

# Global Varibles
CURRENT_RECORDING_FORMAT = False



def upload_recording(request):
    global CURRENT_RECORDING_FORMAT

    if 'file' not in request.files:
        print("No Recording Included!")
        return {"error": "No Recording Uploaded!"}, 400

    file = request.files['file']
    if file.content_type == "audio/wav":
        CURRENT_RECORDING_FORMAT = 'audio'
        filename = "recording.wav"
    elif file.content_type == "video/mp4":
        CURRENT_RECORDING_FORMAT = 'video'
        filename = "recording.mp4"
    else:
        CURRENT_RECORDING_FORMAT = False
        print(f"Error: Unsupported file format: {file.content_type}")
        return {"error": f"Unsupported file format: {file.content_type}"}, 400

    file_path = os.path.join("tmp", filename)

    # Clear tmp directory before saving new file
    tmp_dir = "tmp"
    if os.path.exists(tmp_dir):
        for f in os.listdir(tmp_dir):
            f_path = os.path.join(tmp_dir, f)
            try:
                if os.path.isfile(f_path) or os.path.islink(f_path):
                    os.unlink(f_path)
                elif os.path.isdir(f_path):
                    shutil.rmtree(f_path)
            except Exception as e:
                print(f"Failed to delete {f_path}. Reason: {e}")

    # Ensure tmp directory exists
    os.makedirs(tmp_dir, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(file.read())

    print(f"Recording saved at: {file_path}")
    return {"message": f"Recording saved at: {file_path}: Success"}, 200

def transcribe(request):
    if not CURRENT_RECORDING_FORMAT:
        print(f"Error: No recordings avaliable")
        return {"error": "Error: No recordings avaliable"}, 400

    # Determine file type
    if CURRENT_RECORDING_FORMAT == "audio":
        segmentations = audio_transcribe("tmp/recording.wav")

        with open("tmp/transcription.json", "w") as f:
            json.dump(segmentations, f, indent=2)

        # Clear the folder first
        clear_dir("tmp/audio_segments")
        sr, audio_array = wavfile.read("tmp/recording.wav")
        for i, seg in enumerate(segmentations):
            start, end = seg["interval"]
            trimmed_audio = trim_audio(audio_array, sr, start, end)
            save_audio_segment(trimmed_audio, sr, f"tmp/audio_segments/{i}.wav")

        return {"segments": segmentations}, 200

    elif CURRENT_RECORDING_FORMAT == "video":
        segmentations = audio_transcribe("tmp/recording.mp4")

        with open("tmp/transcription.json", "w") as f:
            json.dump(segmentations, f, indent=2)

        # Clear the folder first
        clear_dir("tmp/video_segments")
        for i, seg in enumerate(segmentations):
            start, end = seg["interval"]
            save_video_segment("tmp/recording.mp4", start, end, f"tmp/video_segments/{i}.mp4")

        return {"segments": segmentations}, 200

def emotion_from_video(request):
    if CURRENT_RECORDING_FORMAT != 'video':
        print("Emotion from video not supported!")
        return {"error": "Error: Emotion from video not supported!"}, 400

    transcription_json_path = "tmp/transcription.json"
    with open(transcription_json_path, "r") as f:
        segmentations = json.load(f)

    all_frames = []  # will store all sampled frames
    frames_per_segment = []  # track how many frames belong to each segment

    # Step 1: sample frames from all segments
    for i, seg in enumerate(segmentations):
        video_path = os.path.join("tmp/video_segments", f"{i}.mp4")
        frames = sample_frames(video_path, FACIAL_SAMPLES_COUNT)
        all_frames.extend(frames)
        frames_per_segment.append(len(frames))

    # Step 2: run batch facial emotion classification for all frames at once
    batch_predictions = facial_emotion_classification_batch(all_frames)

    # Step 3: distribute predictions back to each segment and vote
    segment_emotions = []
    idx = 0  # pointer in batch_predictions
    for frame_count in frames_per_segment:
        # aggregate scores for this segment
        agg_scores = {}
        for _ in range(frame_count):
            pred = batch_predictions[idx]
            for label, score in pred.items():
                agg_scores[label] = agg_scores.get(label, 0) + score
            idx += 1

        # normalize by number of frames
        for label in agg_scores:
            agg_scores[label] /= frame_count

        voted_emotion = max(agg_scores, key=agg_scores.get)
        segment_emotions.append({"facial": voted_emotion})

    # Step 4: save results
    with open("tmp/segment_emotions.json", "w") as f:
        json.dump(segment_emotions, f, indent=2)

    return {"segment_emotions": segment_emotions}, 200


def emotion_from_audio(request):
    if CURRENT_RECORDING_FORMAT not in ("audio", "video"):
        print("Emotion extraction not supported!")
        return {"error": "Error: Emotion extraction not supported!"}, 400

    transcription_json_path = "tmp/transcription.json"
    with open(transcription_json_path, "r") as f:
        segmentations = json.load(f)

    all_audio_segments = []  

    for i, seg in enumerate(segmentations):
        # Load segment audio
        if CURRENT_RECORDING_FORMAT == "audio":
            sr, audio_array = wavfile.read(f"tmp/audio_segments/{i}.wav")
        else:  # video: convert video segment to audio array
            video_path = os.path.join("tmp/video_segments", f"{i}.mp4")
            sr, audio_array = video_to_audio_array(video_path)

        if audio_array is None:
            # fallback empty array if extraction fails
            print("Cannot obtain audio segments!")
            audio_array = np.zeros((1,), dtype=np.int16)

        all_audio_segments.append(audio_array)

    # Run batch audio emotion classification
    predicted_labels = audio_emotion_classification(all_audio_segments)

    # Assign predictions per segment
    # Load existing segment_emotions.json if it exists
    segment_emotions_path = "tmp/segment_emotions.json"
    if os.path.exists(segment_emotions_path):
        with open(segment_emotions_path, "r") as f:
            segment_emotions = json.load(f)
    else:
        # If the file doesn't exist, initialize with empty dicts
        segment_emotions = [{} for _ in predicted_labels]
    
    # Merge audio predictions into each segment's dict
    for i, label in enumerate(predicted_labels):
        if i >= len(segment_emotions):
            segment_emotions.append({"audio": label})
        else:
            segment_emotions[i]["audio"] = label
    
    # Save the updated segment_emotions.json
    with open(segment_emotions_path, "w") as f:
        json.dump(segment_emotions, f, indent=2)

    return {"segment_emotions": segment_emotions}, 200


def merge_emotions_with_LLM(request):
    if CURRENT_RECORDING_FORMAT not in ("audio", "video"):
        print("No recordings avaliable.")
        return {"error": "Error: No recordings avaliable."}, 400

    annotated_transcript = merge_emotions_with_transcriptions("tmp/segment_emotions.json", "tmp/transcription.json")

    messages = [
                {
                    "role": "system",
                    "content": "You will be provided with a annotated transcript from a speech or monologue. The annotations will be emotions \
                    from the corresponding part of the transcript, quoted in pairs of []. You will see things like \
                    [facial: <a facial emotio>, audio: <an audio emotion>] - the adjectives before the column represent from which \
                    source the emotions were obtained. Wav2vec and image classification models are employed to detect the emotions from \
                    the speaker as he/she speaks. One important thing to ponder: since both modalities capture only a part of the \
                    emotional information from the full speech, they are not necessaryly accurate (and for most cases not). YOU would have to \
                    analyse these evaluations and ALONG WITH THE TEXT, make your own evaluation on the emotions. You have to refill the \
                    brackets [] with the your evaluations. You may choose from \
                    ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']. Also, don't have any indicators like \
                    facial or audio when you return your responce (Just one plain [<emotion>] in the middle of both brackets) and remember to keep the original transcript - YOU ARE ONLY EDITING THE BRACKETS. ONE EMOTION PER BRACKET. Sometimes, you \
                    might encounter empty brackets. These are cases where emotional detections are not avaliable. You have to evaluate the \
                    emotion solely upon the text. The annotated transcript will be provided."
                },
                {
                    "role": "user",
                    "content": f"Here is an annotated transcript '{annotated_transcript}'. In your response, please ONLY provide your version \
                    of the annotation along with the transcript, NO ADDTIONAL CONTENTS ARE ALLOWED!!!"
                },
    ]

    response = ollama.chat(
        model=model,
        messages=messages
    )
    assistant_content = dict(response['message'])['content']
    
    return {"Anotated Transcript": assistant_content}, 200




def inference_from_prompt(request):
    global CURRENT_SESSION_ID
    if not CURRENT_SESSION_ID:
        print("Session not initialized.")
        return {"error": "Session not initialized."}, 400

    prompt = request.form.get("prompt")
    if not prompt:
        print("No prompt input.")
        return {"error": "No prompt input."}, 400

    with open(f"{CURRENT_SESSION_ID}/messages.json", "r") as f:
        messages = json.load(f)
        
    messages = prompt_input(messages, prompt)
    
    with open(f"{CURRENT_SESSION_ID}/messages.json", "w") as f:
        json.dump(messages, f, indent=2)
        
    last_content = messages[-1].get("content", "")
    print("Last message:", messages[-1])
    print("Prompting Successful")
    return {"last_response": last_content}, 200
    


function_list = [upload_recording, transcribe, emotion_from_video, emotion_from_audio, merge_emotions_with_LLM]


@app.route("/upload", methods=["POST"])
def upload_content():
    global server_status

    if server_status["busy"]:
        return jsonify({"error": "Server is busy"}), 503

    # Extract command from request
    command = request.form.get("command")
    if not command:
        return jsonify({"error": "No command provided"}), 400

    # Look up command index
    try:
        cmd_index = command_list.index(command)
    except ValueError:
        return jsonify({"error": f"Unknown command '{command}'"}), 400

    # Set busy, run function synchronously, then set idle
    server_status["busy"] = True
    try:
        # Capture the function's return value
        func_response = function_list[cmd_index](request)  # always (dict, status)
        if isinstance(func_response, tuple) and len(func_response) == 2:
            data, status_code = func_response
            return jsonify(data), status_code
        else:
            return jsonify(func_response), 200
    except Exception as e:
        server_status["busy"] = False
        return jsonify({"error": f"Function execution failed: {e}"}), 500
    finally:
        server_status["busy"] = False

     
    # Capture the function's return value
    # server_status["busy"] = False
    # func_response = function_list[cmd_index](request)  # always (dict, status)
    # if isinstance(func_response, tuple) and len(func_response) == 2:
    #     data, status_code = func_response
    #     return jsonify(data), status_code
    # else:
    #     return jsonify(func_response), 200

    # server_status["busy"] = False


    
@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Returns HTTP 200 with a JSON indicating the server is alive.
    """
    return jsonify({"status": "healthy", "server_status": "busy" if server_status["busy"] else "idle"}), 200  


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "busy" if server_status["busy"] else "idle"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=61016)
