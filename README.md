# Sendable Requests to Server (Audio & Video Analysis)

This server allows you to **upload and analyse audio and video files**. All communication is done via `POST` requests to the server endpoint:

```
https://your-server-endpoint.com/upload
```

Each request uses `FormData` with a `command` key and additional fields depending on the command.

---

## 1. Analyse Audio

**Command:** `analyse_audio`  
**FormData:**  
```js
formData.append("command", "analyse_audio");
formData.append("file", <File Object>); // 16kHz .wav file
```

**Status:**  
- Success: *(Not implemented yet – placeholder for server response)*  
- Failure: *(Not implemented yet – placeholder for server response)*  

**Notes:**  
- The uploaded audio must be in **16kHz WAV format**.  

---

## 2. Analyse Video

**Command:** `analyse_video`  
**FormData:**  
```js
formData.append("command", "analyse_video");
formData.append("file", <File Object>); // .mp4 file
```

**Status:**  
- Success: *(Not implemented yet – placeholder for server response)*  
- Failure: *(Not implemented yet – placeholder for server response)*  

**Notes:**  
- The uploaded video must be in **MP4 format**.  

---

### General Notes

- All requests **require valid File objects**.  
- Always check the server response for errors.  
- Errors will be returned as JSON:  
```json
{"error": "...message..."}
```

---

### Example Workflow

1. Prepare a `FormData` object.  
2. Append the `command` key with either `"analyse_audio"` or `"analyse_video"`.  
3. Append the file object.  
4. Send a `POST` request to the server endpoint.  
5. Parse JSON response for status or results.  
