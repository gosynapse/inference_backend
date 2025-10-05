from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_file
from flask import url_for
import os
import json
import time
from models import *
import requests
import shutil
import ollama
from moviepy.editor import AudioFileClip
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import subprocess
from eleven_labs import *
from dotenv import load_dotenv
from supabase import create_client



app = Flask(__name__)
CORS(app)  # allow all origins for frontend JS

# Global state
server_status = {"busy": False}

# Define commands and corresponding functions
command_list = ["upload_recording", "upload_keyboard_video", "transcribe", "transcribe_keyboard_video_input", "emotion_from_video", "emotion_from_audio", "merge_emotions_with_LLM", "annotate_pure_text", "generate_11labs_voice", "generate_11labs_text_to_speech"]

model = "gemma3:4b-it-q8_0" 
FACIAL_SAMPLES_COUNT = 3  

# Global Varibles
CURRENT_RECORDING_FORMAT = False

load_dotenv()

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)



def upload_recording(request):
    global CURRENT_RECORDING_FORMAT

    if 'file' not in request.files:
        print("No Recording Included!")
        return {"error": "No Recordings Uploaded!"}, 400

    file = request.files['file']

    tmp_dir = "tmp"
    # Clear tmp directory before saving new file
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
    os.makedirs(tmp_dir, exist_ok=True)

    # Handle audio/webm
    if file.content_type == "audio/webm":
        CURRENT_RECORDING_FORMAT = 'audio'
        # Save the uploaded webm temporarily
        tmp_webm_path = os.path.join(tmp_dir, "recording.webm")
        with open(tmp_webm_path, "wb") as f:
            f.write(file.read())

        # Convert to WAV 16kHz
        wav_path = os.path.join(tmp_dir, "recording.wav")
        try:
            audio = AudioSegment.from_file(tmp_webm_path, format="webm")
            audio = audio.set_frame_rate(16000).set_channels(1)  # mono 16kHz
            audio.export(wav_path, format="wav")
        except Exception as e:
            print(f"Failed to convert webm to wav: {e}")
            return {"error": f"Failed to convert audio: {e}"}, 500

        print(f"Audio saved at: {wav_path}")
        return {"message": f"Recording saved at: {wav_path}: Success"}, 200

    # Handle video/webm
    elif file.content_type == "video/webm":
        CURRENT_RECORDING_FORMAT = 'video'
        file_path = os.path.join(tmp_dir, "recording.webm")
        with open(file_path, "wb") as f:
            f.write(file.read())
        print(f"Video saved at: {file_path}")
        return {"message": f"Recording saved at: {file_path}: Success"}, 200

    else:
        CURRENT_RECORDING_FORMAT = False
        print(f"Error: Unsupported file format: {file.content_type}")
        return {"error": f"Unsupported file format: {file.content_type}"}, 400


    

def upload_keyboard_video(request):
    """
    Upload a video in WEBM format along with keyboard-video timestamp correspondence.
    Converts the video to MP4 using ffmpeg and saves both the video and correspondence JSON.
    Expects:
      - 'file': the video (video/webm)
      - 'keyboard_video': JSON string with {"input_text": "...", "video_timestamps": [...]}
    Saves:
      - Video: tmp/recording.mp4
      - Correspondence JSON: tmp/keyboard_video.json
    """
    if 'file' not in request.files or 'keyboard_video' not in request.form:
        print("Error: Missing video file or keyboard-video data")
        return {"error": "Missing video file or keyboard-video data"}, 400

    file = request.files['file']
    keyboard_json_str = request.form['keyboard_video']

    if file.content_type != "video/webm":
        print(f"Error: Unsupported file format: {file.content_type}")
        return {"error": f"Unsupported file format: {file.content_type}"}, 400

    tmp_dir = "tmp"

    # Clear tmp directory before saving new file
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
    os.makedirs(tmp_dir, exist_ok=True)

    # Save temporary WEBM video
    webm_path = os.path.join(tmp_dir, "recording.webm")
    with open(webm_path, "wb") as f:
        f.write(file.read())
    print(f"WEBM video saved at: {webm_path}")

    # Convert WEBM video to MP4 using ffmpeg subprocess
    mp4_path = os.path.join(tmp_dir, "recording.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", webm_path,
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
            mp4_path
        ], check=True)
        print(f"Converted video saved at: {mp4_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert WebM to MP4: {e}")
        return {"error": f"Video conversion failed: {e}"}, 500
    finally:
        # Optional: remove the temporary WEBM file
        if os.path.exists(webm_path):
            os.remove(webm_path)

    # Save keyboard-video correspondence
    try:
        keyboard_data = json.loads(keyboard_json_str)
        if "input_text" not in keyboard_data or "video_timestamps" not in keyboard_data:
            raise ValueError("keyboard_video JSON must contain 'input_text' and 'video_timestamps'")
        json_path = os.path.join(tmp_dir, "keyboard_video.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(keyboard_data, f, indent=2)
        print(f"Keyboard-video correspondence saved at: {json_path}")
    except Exception as e:
        print(f"Failed to save keyboard-video JSON: {e}")
        return {"error": f"Invalid keyboard-video JSON: {e}"}, 400

    global CURRENT_RECORDING_FORMAT
    CURRENT_RECORDING_FORMAT = 'video'
    return {"message": "Video converted to MP4 and keyboard-video correspondence saved successfully"}, 200

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
        segmentations = audio_transcribe("tmp/recording.webm")

        with open("tmp/transcription.json", "w") as f:
            json.dump(segmentations, f, indent=2)

        # Clear the folder first
        clear_dir("tmp/video_segments")
        for i, seg in enumerate(segmentations):
            start, end = seg["interval"]
            save_video_segment("tmp/recording.webm", start, end, f"tmp/video_segments/{i}.mp4")

        return {"segments": segmentations}, 200

# def transcribe_keyboard_video_input(request):
#     keyboard_file = "tmp/keyboard_video.json"
#     video_file = "tmp/recording.mp4"
#     output_json = "tmp/transcription.json"
#     segment_folder = "tmp/video_segments"

#     # Ensure segment folder exists and is empty
#     if os.path.exists(segment_folder):
#         for f in os.listdir(segment_folder):
#             os.remove(os.path.join(segment_folder, f))
#     else:
#         os.makedirs(segment_folder)

#     # Load keyboard-video JSON
#     try:
#         with open(keyboard_file, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except Exception as e:
#         return {"error": f"Failed to read {keyboard_file}: {e}"}, 400

#     input_text = data.get("input_text", "")
#     video_timestamps = data.get("video_timestamps", [])

#     # Validate lengths
#     if len(input_text) != len(video_timestamps):
#         return {"error": "Length of video_timestamps does not match input_text"}, 400

#     # Generate sentence intervals
#     try:
#         intervals = tokenized_intervals(input_text, video_timestamps)
#     except Exception as e:
#         return {"error": f"Failed during tokenization: {e}"}, 500

#     # Build segment JSON
#     sentences = sent_tokenize(input_text)
#     segmentations = []
#     for idx, (sentence, (start_ts, end_ts)) in enumerate(zip(sentences, intervals)):
#         segmentations.append({
#             "text": sentence,
#             "interval": [start_ts, end_ts]
#         })
#         # Trim video segment
#         save_video_segment(video_file, start_ts, end_ts, os.path.join(segment_folder, f"{idx}.mp4"))

#     # Save transcription JSON
#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(segmentations, f, indent=2)

#     return {"segments": segmentations}, 200

def transcribe_keyboard_video_input(request):
    start_total = time.time()

    keyboard_file = "tmp/keyboard_video.json"
    video_file = "tmp/recording.mp4"
    output_json = "tmp/transcription.json"
    segment_folder = "tmp/video_segments"

    # Ensure segment folder exists and is empty
    start = time.time()
    if os.path.exists(segment_folder):
        for f in os.listdir(segment_folder):
            os.remove(os.path.join(segment_folder, f))
    else:
        os.makedirs(segment_folder)
    print(f"Segment folder setup time: {time.time() - start:.3f}s")

    # Load keyboard-video JSON
    start = time.time()
    try:
        with open(keyboard_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"error": f"Failed to read {keyboard_file}: {e}"}, 400
    print(f"Loading JSON time: {time.time() - start:.3f}s")

    input_text = data.get("input_text", "")
    video_timestamps = data.get("video_timestamps", [])

    # Validate lengths
    start = time.time()
    if len(input_text) != len(video_timestamps):
        return {"error": "Length of video_timestamps does not match input_text"}, 400
    print(f"Validation time: {time.time() - start:.3f}s")

    # Generate sentence intervals
    start = time.time()
    try:
        intervals = tokenized_intervals(input_text, video_timestamps)
    except Exception as e:
        return {"error": f"Failed during tokenization: {e}"}, 500
    print(f"Tokenized intervals time: {time.time() - start:.3f}s")

    # Build segment JSON
    start = time.time()
    sentences = sent_tokenize(input_text)
    segmentations = []
    for idx, (sentence, (start_ts, end_ts)) in enumerate(zip(sentences, intervals)):
        segmentations.append({
            "text": sentence,
            "interval": [start_ts, end_ts]
        })
        # Trim video segment
        save_video_segment(video_file, start_ts, end_ts, os.path.join(segment_folder, f"{idx}.mp4"))
    print(f"Segment processing time: {time.time() - start:.3f}s")

    # Save transcription JSON
    start = time.time()
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(segmentations, f, indent=2)
    print(f"Saving JSON time: {time.time() - start:.3f}s")

    print(f"Total elapsed time: {time.time() - start_total:.3f}s")
    return {"segments": segmentations}, 200

def emotion_from_video(request):
    start_total = time.time()

    if CURRENT_RECORDING_FORMAT != 'video':
        print("Emotion from video not supported!")
        return {"error": "Error: Emotion from video not supported!"}, 400

    # Load transcription JSON
    start = time.time()
    transcription_json_path = "tmp/transcription.json"
    with open(transcription_json_path, "r") as f:
        segmentations = json.load(f)
    print(f"Loading transcription JSON time: {time.time() - start:.3f}s")

    all_frames = []  # will store all sampled frames
    frames_per_segment = []  # track how many frames belong to each segment

    # Step 1: sample frames from all segments
    start = time.time()
    for i, seg in enumerate(segmentations):
        video_path = os.path.join("tmp/video_segments", f"{i}.mp4")
        seg_start = time.time()
        frames = sample_frames(video_path, FACIAL_SAMPLES_COUNT)
        print(f"Segment {i} frame sampling time: {time.time() - seg_start:.3f}s")
        all_frames.extend(frames)
        frames_per_segment.append(len(frames))
    print(f"Total frame sampling time: {time.time() - start:.3f}s")

    # Step 2: run batch facial emotion classification for all frames at once
    start = time.time()
    batch_predictions = facial_emotion_classification_batch(all_frames)
    print(f"Facial emotion batch classification time: {time.time() - start:.3f}s")

    # Step 3: distribute predictions back to each segment and vote
    start = time.time()
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
    print(f"Aggregating predictions and voting time: {time.time() - start:.3f}s")

    # Step 4: save results
    start = time.time()
    with open("tmp/segment_emotions.json", "w") as f:
        json.dump(segment_emotions, f, indent=2)
    print(f"Saving results JSON time: {time.time() - start:.3f}s")

    print(f"Total elapsed time: {time.time() - start_total:.3f}s")
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
                    ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']. Also, do NOT have any indicators like \
                    facial or audio when you return your responce (Just one plain [<emotion>] in the middle of both brackets, stuffs like [facial: happy] is NOT ALLOWED) and remember to keep the original transcript - YOU ARE ONLY EDITING THE BRACKETS. ONE EMOTION PER BRACKET. LEAVE THE TRANSCRIPT VISIBLE. Sometimes, you \
                    might encounter empty brackets. These are cases where emotional detections are not avaliable. You have to evaluate the \
                    emotion solely upon the text. The annotated transcript will be provided."
                },
                {
                    "role": "user",
                    "content": f"Here is an annotated transcript '{annotated_transcript}'. In your response, please ONLY provide your version \
                    of the annotation along with the transcript, NO ADDTIONAL CONTENTS ARE ALLOWED!!! ALSO, always put the annotation at the \
                    end of the segments"
                },
    ]

    response = ollama.chat(
        model=model,
        messages=messages
    )
    
#     assistant_content = dict(response['message'])['content']
#     messages.append({"role": "assistant", "content": assistant_content})
#     messages.append({"role": "user", "content": """TASK: Re-format your previous response so that 100% of the transcript is wrapped in HTML-style emotion tags.

# ZERO-TOLERANCE FORMAT REQUIREMENTS:
# 1. All text must be enclosed in <emotion> ... </emotion> tags. No text may remain outside of tags.
# 2. Wrap continuous segments of text that share the same emotion. Do NOT wrap each word individually.
#    - Correct: <happy>Text segment 1</happy><sad>Text segment 2</sad>
#    - Incorrect: <happy>Text</happy> <happy>segment</happy> <happy>1</happy>
# 3. Do NOT create new segment boundaries. Preserve the segmentation and order from the last annotated response exactly.
# 4. Never produce empty tags. Every <emotion> ... </emotion> must contain at least one character.
# 5. If the emotion is uncertain for a segment, use <neutral>.
# 6. The output must contain ONLY the fully wrapped transcript. Do NOT add explanations, comments, leading/trailing whitespace, or any content outside the tags.
# 7. DO NOT include code fences or headers (e.g., ```html, ```text, etc.).
# 8. Start immediately with the first tag; no extra whitespace or newlines at the beginning or end.
# 9. The format MUST be strictly followed. Any deviation is unacceptable."""})

#     response = ollama.chat(
#         model=model,
#         messages=messages
#     )

    assistant_content = dict(response['message'])['content']
    return {"Anotated Transcript": assistant_content}, 200


def annotate_pure_text(request):
    prompt = request.form.get("text_input")
    if not prompt:
        print("No text input.")
        return {"error": "No text input."}, 400
    
    messages = [{"role": "system", "content": "You will be provided with a transcript from a speech or monologue. Your task is to first segment the script into several segments and identify where the emotional tone of the speaker changes. At those locations, you would have to annotate your infered emotion in square brackets, like ...<here is some text segments> [<an emotion>]... You may choose from ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']. Remember to keep the original transcript - YOU ARE ONLY ADDING THE BRACKETS. ONE EMOTION PER BRACKET. The annotated transcript will be provided."}, {"role": "user", "content": f"Here is a transcript '{prompt}'. In your response, please ONLY provide your version of the annotation along with the transcript, NO ADDTIONAL CONTENTS ARE ALLOWED!!!"}]

    response = ollama.chat(
        model=model,
        messages=messages
    )
    
    assistant_content = dict(response['message'])['content']
    return {"Anotated Transcript": assistant_content}, 200



def generate_11labs_voice(request):
    if 'file' not in request.files:
        print("No voice sample included!")
        return {"error": "No voice sample included!"}, 400

    file = request.files['file']

    tmp_dir = "tmp_11labs"
    # Clear tmp directory before saving new file
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
    os.makedirs(tmp_dir, exist_ok=True)

    # Handle audio/webm
    if file.content_type == "audio/mpeg":
        tmp_webm_path = os.path.join(tmp_dir, "sample.webm")
        with open(tmp_webm_path, "wb") as f:
            f.write(file.read())
            
        voice_id = create_voice("tmp_11labs/sample.webm")

        return {"voice_id": voice_id}, 200

    else:
        CURRENT_RECORDING_FORMAT = False
        print(f"Error: Unsupported file format: {file.content_type}")
        return {"error": f"Unsupported file format: {file.content_type}"}, 400

def generate_11labs_text_to_speech(request):
    
    text = request.form.get('text')
    voice_id = request.form.get('voice_id')
    audio_id = request.form.get('audio_id')
    user_id = request.form.get('user_id')

    messages = [{"role": "user", "content": "You will be given a annotated transcript. Please extract the texts from it and apply any of the following tags at anywhere appropriate: [excited], [nervous], [frustrated], [sorrowful], [calm], [sigh], [laughs], [gulps], [gasps], [whispers], [pauses], [hesitates], [stammers], [resigned tone], [cheerfully], [flatly], [deadpan], [playfully]. Note, some of the emotions does not exist. Use only tags that are provided" + f"Here is an annotated transcript '{text}'. In your response, please ONLY provide your version of the tagged speech, NO ADDTIONAL CONTENTS ARE ALLOWED!!!"}]

    response = ollama.chat(
        model=model,
        messages=messages
    )

    print(f"generated_audio/{user_id}/{audio_id}")


    assistant_content = dict(response['message'])['content']
    print(assistant_content)
    text_to_speech(voice_id, assistant_content, "tmp_11labs/output.mp3")

    # Upload to Supabase
    with open("tmp_11labs/output.mp3", "rb") as f:
        print(f"generated_audio/{user_id}/{audio_id}")
        supabase.storage.from_("media").upload(
            f"generated_audio/{user_id}/{audio_id}",
            f,
            file_options={"content-type": "audio/mpeg"}
        )

    
    return {"voice_id": voice_id}, 200



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
    


function_list = [upload_recording, upload_keyboard_video, transcribe, transcribe_keyboard_video_input, emotion_from_video, emotion_from_audio, merge_emotions_with_LLM, annotate_pure_text, generate_11labs_voice, generate_11labs_text_to_speech]


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
    # server_status["busy"] = True
    # try:
    #     # Capture the function's return value
    #     func_response = function_list[cmd_index](request)  # always (dict, status)
    #     if isinstance(func_response, tuple) and len(func_response) == 2:
    #         data, status_code = func_response
    #         return jsonify(data), status_code
    #     else:
    #         return jsonify(func_response), 200
    # except Exception as e:
    #     server_status["busy"] = False
    #     return jsonify({"error": f"Function execution failed: {e}"}), 500
    # finally:
    #     server_status["busy"] = False

     
    # Capture the function's return value
    server_status["busy"] = False
    func_response = function_list[cmd_index](request)  # always (dict, status)
    if isinstance(func_response, tuple) and len(func_response) == 2:
        data, status_code = func_response
        return jsonify(data), status_code
    else:
        return jsonify(func_response), 200

    server_status["busy"] = False


    
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
