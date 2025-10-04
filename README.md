# Sendable Requests to Server

This server allows you to upload audio (`.wav`) or video (`.mp4`) recordings, transcribe them, extract emotions from audio and video, and merge these emotions with a large language model (LLM) to generate an annotated transcript.

All communication is done via `POST` requests to the server endpoint:

```
https://q0ki6holrvby0r-61016.proxy.runpod.net/upload
```

Each request uses `FormData` with a `command` key and optional additional fields.

> **Note:** This server does **not** rely on session management. To start any workflow, first upload a `.wav` or `.mp4` file. Re-uploading will overwrite existing recordings.

---

## 1. Upload Recording

**Command:** `upload_recording`  
**FormData:**  
```js
formData.append("command", "upload_recording");
formData.append("file", <File Object>);
```

**Status:**  
- Success:  
```json
{
  "message": "Recording saved at: tmp/recording.wav: Success"
}
```
or
```json
{
  "message": "Recording saved at: tmp/recording.mp4: Success"
}
```

- Failure:  
```json
{"error": "No Recording Uploaded!"}
```
```json
{"error": "Unsupported file format: <file_content_type>"}
```

---

## 1a. Upload Keyboard Video

**Command:** `upload_keyboard_video`  
**FormData:**  
```js
formData.append("command", "upload_keyboard_video");
formData.append("file", <video File Object>);
formData.append("keyboard_video", JSON.stringify({
    input_text: "...",
    video_timestamps: [...]
}));
```

**Status:**  
- Success:  
```json
{"message": "Video and keyboard-video correspondence saved successfully"}
```

- Failure:  
```json
{"error": "Missing video file or keyboard-video data"}
```
```json
{"error": "Unsupported file format: <file_content_type>"}
```
```json
{"error": "Invalid keyboard-video JSON: <error>"}
```

**Notes:**  
- Saves video to `tmp/recording.mp4`  
- Saves keyboard-video correspondence to `tmp/keyboard_video.json`  
- Clears previous tmp files before saving  

---

## 2. Transcribe Recording

**Command:** `transcribe`  
**FormData:**  
```js
formData.append("command", "transcribe");
```

**Status:**  
- Success:  
```json
{
  "segments": [
    {"interval": [0.0, 2.5], "text": "Hello everyone."},
    {"interval": [2.5, 5.0], "text": "Today we will discuss emotions."}
  ]
}
```

- Failure:  
```json
{"error": "Error: No recordings avaliable"}
```

**Notes:**  
- For audio, segments are saved in `tmp/audio_segments/`  
- For video, segments are saved in `tmp/video_segments/`  

---

## 2a. Transcribe Keyboard Video Input

**Command:** `transcribe_keyboard_video_input`  
**FormData:**  
```js
formData.append("command", "transcribe_keyboard_video_input");
```

**Status:**  
- Success:  
```json
{
  "segments": [
    {"text": "Hello everyone.", "interval": [0.0, 2.5]},
    {"text": "Today we will discuss emotions.", "interval": [2.5, 5.0]}
  ]
}
```

- Failure:  
```json
{"error": "Failed to read tmp/keyboard_video.json"}
```
```json
{"error": "Length of video_timestamps does not match input_text"}
```
```json
{"error": "Failed during tokenization: <error>"}
```

**Notes:**  
- Uses the `keyboard_video.json` timestamps to segment the video  
- Saves each video segment in `tmp/video_segments/`  
- Saves segment transcription to `tmp/transcription.json`  

---

## 3. Emotion from Video

**Command:** `emotion_from_video`  
**FormData:**  
```js
formData.append("command", "emotion_from_video");
```

**Status:**  
- Success:  
```json
{
  "segment_emotions": [
    {"facial": "happy"},
    {"facial": "neutral"}
  ]
}
```

- Failure:  
```json
{"error": "Error: Emotion from video not supported!"}
```

**Notes:**  
- Only works if the uploaded recording is a `.mp4` video.  
- Uses sampled frames from each video segment to classify facial emotions.  

---

## 4. Emotion from Audio

**Command:** `emotion_from_audio`  
**FormData:**  
```js
formData.append("command", "emotion_from_audio");
```

**Status:**  
- Success:  
```json
{
  "segment_emotions": [
    {"audio": "happy"},
    {"audio": "sad"}
  ]
}
```

- Failure:  
```json
{"error": "Error: Emotion extraction not supported!"}
```

**Notes:**  
- Works for both audio and video recordings.  
- Uses audio segments from the transcription to classify emotions.

---

## 5. Merge Emotions with LLM

**Command:** `merge_emotions_with_LLM`  
**FormData:**  
```js
formData.append("command", "merge_emotions_with_LLM");
```

**Status:**  
- Success:  
```json
{
  "Anotated Transcript": "Hello everyone [happy]. Today we will discuss emotions [neutral]."
}
```

- Failure:  
```json
{"error": "Error: No recordings avaliable."}
```

## 6. Annotate Pure Text

This command provides an **annotated transcript directly from a plain text input**, without needing any uploaded audio or video or prior emotion extraction. It works similarly to `merge_emotions_with_LLM` but infers emotions purely from the text.

**Command:** `annotate_pure_text`  
**FormData:**  
```js
formData.append("command", "annotate_pure_text");
formData.append("text_input", "Your transcript text here");
```

**Status:**  
- Success:  
```json
{
  "Annotated Transcript": "Hello everyone [happy]. Today we will discuss emotions [neutral]."
}
```

- Failure:  
```json
{"error": "No text input."}
```

**Notes:**  
- Segments the text into multiple portions where emotional tone changes.  
- Annotates each segment with an inferred emotion in square brackets.  
- Only one emotion per bracket, chosen from: `['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']`.  
- Returns a **plain string** of the annotated transcript; original text is preserved.  
- No audio or video files are required.


**Notes:**  
- Combines segment-level facial and audio emotions and produces an annotated transcript using the LLM.  
- Returns a **plain string** for the annotated transcript; brackets are filled with your evaluations of emotion.  

---

## 7. Health Check

**Endpoint:** `/health`  
**Method:** `GET`  

**Response:**  
```json
{
  "status": "healthy",
  "server_status": "idle"  // or "busy"
}
```

**Notes:**  
- Indicates whether the server is running and whether it is currently processing a request.

---

## 8. Server Status

**Endpoint:** `/status`  
**Method:** `GET`  

**Response:**  
```json
{
  "status": "idle" // or "busy"
}
```

**Notes:**  
- Shows if the server is currently busy processing a command.  

---

### Error Handling

- All command errors return JSON in the format:
```json
{"error": "<error message>"}
```
- Common error messages:
  - `No Recording Uploaded!`
  - `Unsupported file format: <type>`
  - `Error: No recordings avaliable`
  - `Error: Emotion from video not supported!`
  - `Error: Emotion extraction not supported!`
  - `Missing video file or keyboard-video data`
  - `Invalid keyboard-video JSON: <error>`
  - `Length of video_timestamps does not match input_text`
  - `Failed during tokenization: <error>`

---

### Notes on Usage

- Start your workflow by **uploading a `.wav` or `.mp4` file**, or **upload a keyboard-video** file if using that workflow.  
- Re-uploading a file will overwrite previous recordings.  
- Transcription must be performed before emotion extraction.  
- Emotions from video or audio can be merged into a final annotated transcript via the LLM.  
- JSON outputs show segment-level results for transcription and emotion classification.  
- `merge_emotions_with_LLM` outputs a **plain annotated transcript string**.  

---

### Recommended Workflow

```text
                 +-----------------------+
                 | Upload Recording      |
                 | (.wav or .mp4)        |
                 +-----------------------+
                          |
                          v
                 +-----------------------+
                 | Transcribe Recording  |
                 | (command: transcribe)|
                 +-----------------------+
                          |
                          v
                 +-----------------------+
                 | Extract Emotions      |
                 | (Audio / Video)       |
                          |
                          v
                 +-----------------------+
                 | Merge Emotions with   |
                 | LLM                   |
                          |
                          v
                 +-----------------------+
                 | Annotated Transcript  |
                 +-----------------------+

                 OR

                 +-----------------------+
                 | Upload Keyboard Video |
                 | (video + timestamps)  |
                 +-----------------------+
                          |
                          v
                 +-------------------------------+
                 | Transcribe Keyboard Video     |
                 | (command: transcribe_keyboard_|
                 | video_input)                  |
                 +-------------------------------+
                          |
                          v
                 +-----------------------+
                 | Extract Emotions      |
                 | (Audio / Video)       |
                          |
                          v
                 +-----------------------+
                 | Merge Emotions with   |
                 | LLM                   |
                          |
                          v
                 +-----------------------+
                 | Annotated Transcript  |
                 +-----------------------+
```


