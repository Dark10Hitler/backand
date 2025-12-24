# services.py — FINAL

import os
import gc
import requests
from moviepy import VideoFileClip, AudioFileClip
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")
ELEVEN_KEY = os.getenv("VITE_ELEVENLABS_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

UPLOAD_DIR = "/tmp/uploads"
FINAL_DIR = "/tmp/final_videos"
AUDIO_DIR = "/tmp/audio_files"

os.makedirs(AUDIO_DIR, exist_ok=True)

eleven = ElevenLabs(api_key=ELEVEN_KEY)

# ================= AUDIO =================
def extract_audio(video_path: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = f"{AUDIO_DIR}/{base}.mp3"

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(
        audio_path,
        codec="mp3",
        bitrate="96k",
        logger=None
    )
    video.close()
    gc.collect()

    return audio_path

# ================= WHISPER =================
def transcribe_audio(audio_path: str):
    with open(audio_path, "rb") as f:
        r = requests.post(
            "https://openrouter.ai/api/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            files={"file": f},
            data={"model": "whisper-1"},
            timeout=120
        )
    r.raise_for_status()
    data = r.json()
    return data.get("text", ""), data.get("language", "en")

# ================= TRANSLATE =================
def translate_text(text, src, target, client):
    res = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Translate text accurately. Return only translation."},
            {"role": "user", "content": f"Translate from {src} to {target}: {text}"}
        ]
    )
    return res.choices[0].message.content.strip()

# ================= TTS =================
def generate_cloned_audio(text, source_audio):
    base = os.path.splitext(os.path.basename(source_audio))[0]
    out = f"{AUDIO_DIR}/dubbed_{base}.mp3"

    stream = eleven.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_96"
    )

    with open(out, "wb") as f:
        if isinstance(stream, bytes):
            f.write(stream)
        else:
            for chunk in stream:
                f.write(chunk)

    return out

# ================= VIDEO =================
def assemble_video(video_path, audio_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    final = f"{FINAL_DIR}/dubbed_{base}.mp4"

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path).set_duration(video.duration)

    final_clip = video.set_audio(audio)
    final_clip.write_videofile(
        final,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=1,
        logger=None
    )

    video.close()
    audio.close()
    final_clip.close()
    gc.collect()

    return final
