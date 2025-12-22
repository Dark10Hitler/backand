import os
from moviepy import VideoFileClip, AudioFileClip
import whisper
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

VITE_ELEVENLABS_KEY = os.getenv("VITE_ELEVENLABS_KEY")
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

if not VITE_ELEVENLABS_KEY:
    raise RuntimeError("VITE_ELEVENLABS_KEY is not set")

# -------------------------------
# DIRECTORIES
# -------------------------------
AUDIO_DIR = "audio_files"
FINAL_DIR = "final_videos"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# -------------------------------
# CLIENTS (инициализируются 1 раз)
# -------------------------------
eleven_client = ElevenLabs(api_key=VITE_ELEVENLABS_KEY)

# Whisper загружаем один раз
print("🔊 Loading Whisper model (tiny)...")
WHISPER_MODEL = whisper.load_model("tiny")

# -------------------------------
# FUNCTIONS
# -------------------------------

def extract_audio(video_path: str) -> str:
    """Извлекаем аудио из видео"""
    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(AUDIO_DIR, f"{base}.mp3")

    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            audio_path,
            codec="mp3",
            bitrate="128k",
            logger=None
        )
        video.close()
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


def transcribe_audio(audio_path: str) -> tuple[str, str]:
    """Транскрибация аудио с помощью Whisper"""
    try:
        result = WHISPER_MODEL.transcribe(audio_path)
        return result["text"], result["language"]
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")


def translate_text(
    text: str,
    source_language_code: str,
    target_language: str,
    client_router
) -> str:
    """Перевод текста через OpenRouter / LLM"""
    try:
        completion = client_router.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. "
                        "Translate the text accurately and naturally. "
                        "Return ONLY the translated text."
                    )
                },
                {
                    "role": "user",
                    "content": f"Translate from {source_language_code} to {target_language}: {text}"
                }
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Translation failed: {e}")


def generate_cloned_audio(
    translated_text: str,
    source_audio_path: str,
    voice_id: str | None = None
) -> str:
    """Генерация озвучки через ElevenLabs"""
    base = os.path.splitext(os.path.basename(source_audio_path))[0]
    output_audio = os.path.join(AUDIO_DIR, f"dubbed_{base}.mp3")

    try:
        stream = eleven_client.text_to_speech.convert(
            text=translated_text,
            voice_id=voice_id or DEFAULT_VOICE_ID,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )

        with open(output_audio, "wb") as f:
            if isinstance(stream, bytes):
                f.write(stream)
            else:
                for chunk in stream:
                    f.write(chunk)

        return output_audio
    except Exception as e:
        raise RuntimeError(f"ElevenLabs TTS failed: {e}")


def assemble_video(video_path: str, dubbed_audio_path: str) -> str:
    """Объединяем оригинальное видео с озвучкой"""
    base = os.path.splitext(os.path.basename(video_path))[0]
    final_path = os.path.join(FINAL_DIR, f"dubbed_{base}.mp4")

    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(dubbed_audio_path)

        # Совпадение длительности
        audio = audio.set_duration(video.duration)

        final = video.set_audio(audio)
        final.write_videofile(
            final_path,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            ffmpeg_params=["-movflags", "faststart"],
            temp_audiofile="temp-audio.m4a",
            logger=None
        )

        return final_path

    except Exception as e:
        raise RuntimeError(f"Video assembly failed: {e}")

    finally:
        for obj in ("video", "audio", "final"):
            if obj in locals():
                locals()[obj].close()


