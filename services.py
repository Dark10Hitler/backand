import os
import gc
import uuid
from moviepy.editor import VideoFileClip, AudioFileClip
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Ключи
OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_KEY = os.getenv("VITE_ELEVENLABS_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

# Папки
UPLOAD_DIR = "/tmp/uploads"
FINAL_DIR = "/tmp/final_videos"
AUDIO_DIR = "/tmp/audio_files"
for d in [UPLOAD_DIR, FINAL_DIR, AUDIO_DIR]: os.makedirs(d, exist_ok=True)

# КЛИЕНТЫ
# Текст через OpenRouter
text_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    default_headers={"HTTP-Referer": "https://smartdub.ai", "X-Title": "SmartDub"}
)

# Звук через Groq (чтобы не было ошибки 405)
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

eleven = ElevenLabs(api_key=ELEVEN_KEY)

def extract_audio(video_path: str) -> str:
    base = os.path.basename(video_path).split('.')[0]
    audio_path = f"{AUDIO_DIR}/{base}_{uuid.uuid4().hex[:4]}.mp3"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="mp3", bitrate="64k", logger=None)
    video.close()
    gc.collect()
    return audio_path

def transcribe_audio(audio_path: str):
    with open(audio_path, "rb") as f:
        res = groq_client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3"
        )
    return res.text, "auto"

def translate_text(text, src, target):
    prompt = f"Act as a professional ads translator. Translate this text into local slang for {target}. Return ONLY translation: {text}"
    res = text_client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()

def generate_cloned_audio(text, source_audio):
    out = f"{AUDIO_DIR}/dubbed_{uuid.uuid4().hex[:8]}.mp3"
    audio_stream = eleven.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_96"
    )
    with open(out, "wb") as f:
        for chunk in audio_stream: f.write(chunk)
    return out

def assemble_video(video_path, audio_path):
    final = f"{FINAL_DIR}/final_{uuid.uuid4().hex[:8]}.mp4"
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
