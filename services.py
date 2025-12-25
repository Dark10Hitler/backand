import os
import gc
import uuid
from moviepy.editor import VideoFileClip, AudioFileClip
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

# Ключи
OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")
ELEVEN_KEY = os.getenv("VITE_ELEVENLABS_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

# Папки
UPLOAD_DIR = "/tmp/uploads"
FINAL_DIR = "/tmp/final_videos"
AUDIO_DIR = "/tmp/audio_files"
for d in [UPLOAD_DIR, FINAL_DIR, AUDIO_DIR]:
    os.makedirs(d, exist_ok=True)

# Инициализация ЛОКАЛЬНОГО Whisper (не требует ключей и интернета)
# Модель "tiny" — самая легкая для Render Free
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Клиент ТОЛЬКО для перевода текста
text_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    default_headers={
        "HTTP-Referer": "https://smartdub.ai",
        "X-Title": "SmartDub AI"
    }
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
    """Локальное распознавание через faster-whisper"""
    segments, info = whisper_model.transcribe(audio_path, beam_size=1)
    full_text = " ".join([segment.text for segment in segments])
    return full_text.strip(), info.language

def translate_text(text: str, target_country: str):
    """Перевод через OpenRouter (это работает стабильно)"""
    prompt = (
        f"Translate this text to local street slang of {target_country}. "
        f"Return ONLY translation: {text}"
    )
    res = text_client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return res.choices[0].message.content.strip()

def generate_cloned_audio(text: str):
    out = f"{AUDIO_DIR}/dubbed_{uuid.uuid4().hex[:8]}.mp3"
    audio_gen = eleven.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_96"
    )
    with open(out, "wb") as f:
        for chunk in audio_gen:
            if chunk: f.write(chunk)
    return out

def assemble_video(video_path: str, audio_path: str):
    final_output = f"{FINAL_DIR}/final_{uuid.uuid4().hex[:8]}.mp4"
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path).set_duration(video.duration)
    final_video = video.set_audio(audio)
    
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
    for f in files:
        try:
            if f and os.path.exists(f): os.remove(f)
        except: pass
