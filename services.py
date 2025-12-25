import os
import gc
import uuid
from moviepy.editor import VideoFileClip, AudioFileClip
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Настройки из .env
OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")
ELEVEN_KEY = os.getenv("VITE_ELEVENLABS_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

# Папки для работы на Render
UPLOAD_DIR = "/tmp/uploads"
FINAL_DIR = "/tmp/final_videos"
AUDIO_DIR = "/tmp/audio_files"

for d in [UPLOAD_DIR, FINAL_DIR, AUDIO_DIR]:
    os.makedirs(d, exist_ok=True)

# Единый клиент OpenRouter для текста и Whisper
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    default_headers={
        "HTTP-Referer": "https://smartdub.ai",
        "X-Title": "SmartDub AI"
    }
)

eleven = ElevenLabs(api_key=ELEVEN_KEY)

def extract_audio(video_path: str) -> str:
    """Извлекает аудио из видео"""
    base = os.path.basename(video_path).split('.')[0]
    audio_path = f"{AUDIO_DIR}/{base}_{uuid.uuid4().hex[:4]}.mp3"
    
    video = VideoFileClip(video_path)
    # Сжимаем битрейт для экономии памяти и лимитов API
    video.audio.write_audiofile(audio_path, codec="mp3", bitrate="64k", logger=None)
    video.close()
    gc.collect()
    return audio_path

def transcribe_audio(audio_path: str):
    """Транскрибация через OpenRouter (модель openai/whisper-1)"""
    with open(audio_path, "rb") as f:
        res = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-1"
        )
    return res.text, "auto"

def translate_text(text: str, target_country: str):
    """Перевод через OpenRouter (Llama 3.1)"""
    prompt = (
        f"Translate this advertising text into the local street slang of {target_country}. "
        f"Make it aggressive and engaging. Return ONLY the translation: {text}"
    )
    
    res = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return res.choices[0].message.content.strip()

def generate_cloned_audio(text: str):
    """Генерация озвучки через ElevenLabs"""
    out = f"{AUDIO_DIR}/dubbed_{uuid.uuid4().hex[:8]}.mp3"
    
    audio_gen = eleven.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_96"
    )
    
    with open(out, "wb") as f:
        for chunk in audio_gen:
            if chunk:
                f.write(chunk)
    return out

def assemble_video(video_path: str, audio_path: str):
    """Сборка финального видео"""
    final_output = f"{FINAL_DIR}/final_{uuid.uuid4().hex[:8]}.mp4"
    
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path).set_duration(video.duration)
    
    final_video = video.set_audio(audio)
    
    # threads=1 критически важно для Render Free, чтобы не было OOM Error
    final_video.write_videofile(
        final_output,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=1,
        logger=None
    )
    
    video.close()
    audio.close()
    final_video.close()
    gc.collect()
    return final_output

def cleanup_files(*files):
    """Удаление временных файлов"""
    for file in files:
        try:
            if file and os.path.exists(file):
                os.remove(file)
        except:
            pass
