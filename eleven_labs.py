import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from io import BytesIO


load_dotenv()
elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)


def create_voice(voice_sample_path):
    voice = elevenlabs.voices.ivc.create(
        name="My Voice Clone",
        # Replace with the paths to your audio files.
        # The more files you add, the better the clone will be.
        files=[BytesIO(open(voice_sample_path, "rb").read())]
    )
    return voice.voice_id

def text_to_speech(voice_id, annotated_text, save_path):
    audio = elevenlabs.text_to_speech.convert(
    text=annotated_text,
    voice_id=voice_id,
    model_id="eleven_v3",
    output_format="mp3_44100_128",
    )
    
    with open(save_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
