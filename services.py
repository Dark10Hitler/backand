import os
import gc
import uuid
from moviepy.editor import VideoFileClip, AudioFileClip
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Ключи и настройки
OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_KEY = os.getenv("VITE_ELEVENLABS_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

# Директории (Render использует /tmp для записи)
UPLOAD_DIR = "/tmp/uploads"
FINAL_DIR = "/tmp/final_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# Инициализация клиентов
# 1. OpenRouter для текста
text_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    default_headers={
        "HTTP-Referer": "https://global-voice-ads.com", # Обязательно для OpenRouter
        "X-Title": "Global Voice Ads",
    }
)

# 2. Groq для транскрибации (Whisper)
audio_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# 3. ElevenLabs
eleven = ElevenLabs(api_key=ELEVEN_KEY)

def extract_audio(video_path: str) -> str:
    """Извлекает аудио из видео во временный файл mp3"""
    base = os.path.basename(video_path).split('.')[0]
    audio_path = f"/tmp/{base}_{uuid.uuid4().hex[:4]}.mp3"
    
    video = VideoFileClip(video_path)
    # Оптимизация для Render: низкий битрейт для экономии RAM
    video.audio.write_audiofile(audio_path, codec="mp3", bitrate="64k", logger=None)
    video.close()
    gc.collect() # Очистка памяти
    return audio_path

def transcribe_audio(audio_path: str):
    """Транскрибация через Groq (Whisper-3)"""
    with open(audio_path, "rb") as f:
        response = audio_client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3",
            response_format="json"
        )
    return response.text, "auto"

def translate_text(text: str, target_country: str):
    """Перевод и адаптация под сленг через OpenRouter"""
    prompt = f"""
    Act as a professional traffic arbitrage expert and native speaker. 
    Translate the following advertising text into the local street slang of {target_country}. 
    Make it punchy, aggressive, and highly engaging for a local audience. 
    Maintain the emotional hook. 
    Text: {text}
    Return ONLY the translated text without comments.
    """
    
    response = text_client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def generate_cloned_audio(text: str):
    """Генерация озвучки через ElevenLabs"""
    output_path = f"/tmp/dubbed_{uuid.uuid4().hex[:6]}.mp3"
    
    audio_gen = eleven.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_96"
    )
    
    with open(output_path, "wb") as f:
        for chunk in audio_gen:
            if chunk:
                f.write(chunk)
                
    return output_path

def assemble_video(video_path: str, audio_path: str):
    """Сборка финального видео"""
    base = os.path.basename(video_path).split('.')[0]
    final_output = f"{FINAL_DIR}/final_{base}_{uuid.uuid4().hex[:4]}.mp4"
    
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    
    # Синхронизируем длительность аудио с видео
    final_audio = audio.set_duration(video.duration)
    final_video = video.set_audio(final_audio)
    
    # Оптимизация для Render Free (один поток, ультрабыстрый пресет)
    final_video.write_videofile(
        final_output,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="/tmp/temp-audio.m4a",
        remove_temp=True,
        preset="ultrafast",
        threads=1,
        logger=None
    )
    
    # Закрываем все дескрипторы для очистки памяти
    video.close()
    audio.close()
    final_video.close()
    gc.collect()
    
    return final_output

def cleanup_files(*files):
    """Удаление временных файлов после обработки"""
    for file in files:
        try:
            if file and os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Cleanup error: {e}")
