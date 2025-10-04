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

**Notes:**  
- Combines segment-level facial and audio emotions and produces an annotated transcript using the LLM.  
- Returns a **plain string** for the annotated transcript; brackets are filled with your evaluations of emotion.  

---

## 6. Health Check

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

## 7. Server Status

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

---

### Notes on Usage

- Start your workflow by **uploading a `.wav` or `.mp4` file**.  
- Re-uploading a file will overwrite previous recordings.  
- Transcription must be performed before emotion extraction.  
- Emotions from video or audio can be merged into a final annotated transcript via the LLM.  
- JSON outputs show segment-level results for transcription and emotion classification.  
- `merge_emotions_with_LLM` outputs a **plain annotated transcript string**.   

I can also create a **diagram showing the recommended workflow** (Upload → Transcribe → Emotion → Merge LLM) if you want it to make the guide even clearer.  

Do you want me to create that diagram?
