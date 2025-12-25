import os
import gc
import uuid
from moviepy.editor import VideoFileClip, AudioFileClip
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")
ELEVEN_KEY = os.getenv("VITE_ELEVENLABS_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

UPLOAD_DIR = "/tmp/uploads"
FINAL_DIR = "/tmp/final_videos"
AUDIO_DIR = "/tmp/audio_files"
for d in [UPLOAD_DIR, FINAL_DIR, AUDIO_DIR]:
    os.makedirs(d, exist_ok=True)

# Глобальную переменную модели убираем, будем грузить в функции
def transcribe_audio(audio_path: str):
    """Локальное распознавание с экономией RAM"""
    from faster_whisper import WhisperModel
    
    # Загружаем модель ТОЛЬКО в момент вызова
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=1)
    full_text = " ".join([segment.text for segment in segments])
    
    # Сразу удаляем модель из памяти
    del model
    gc.collect()
    
    return full_text.strip(), info.language

def translate_text(text: str, target_country: str):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
        default_headers={"HTTP-Referer": "https://smartdub.ai", "X-Title": "SmartDub AI"}
    )
    prompt = f"Translate to {target_country} street slang. Return ONLY translation: {text}"
    res = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return res.choices[0].message.content.strip()

def generate_cloned_audio(text: str):
    eleven = ElevenLabs(api_key=ELEVEN_KEY)
    out = f"{AUDIO_DIR}/dubbed_{uuid.uuid4().hex[:8]}.mp3"
    audio_gen = eleven.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_64" # Еще ниже качество для экономии
    )
    with open(out, "wb") as f:
        for chunk in audio_gen:
            if chunk: f.write(chunk)
    return out

def assemble_video(video_path: str, audio_path: str):
    final_output = f"{FINAL_DIR}/final_{uuid.uuid4().hex[:8]}.mp4"
    video = VideoFileClip(video_path)
    
    # Если видео слишком длинное, обрежем для теста (опционально)
    if video.duration > 60:
        video = video.subclip(0, 60)

    audio = AudioFileClip(audio_path).set_duration(video.duration)
    final_video = video.set_audio(audio)
    
    final_video.write_videofile(
        final_output,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        fps=24, # Снижаем FPS для скорости
        threads=1,
        logger=None
    )
    
    video.close()
    audio.close()
    final_video.close()
    
    del video, audio, final_video
    gc.collect()
    return final_output

def extract_audio(video_path: str) -> str:
    base = os.path.basename(video_path).split('.')[0]
    audio_path = f"{AUDIO_DIR}/{base}_{uuid.uuid4().hex[:4]}.mp3"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="mp3", bitrate="64k", logger=None)
    video.close()
    gc.collect()
    return audio_path

def cleanup_files(*files):
    for f in files:
        try:
            if f and os.path.exists(f): os.remove(f)
        except: pass
    gc.collect()
