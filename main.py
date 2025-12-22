import os
import uuid
import asyncio
import httpx
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from db import (
    create_user,
    get_user_by_code,
    bind_telegram,
    add_task,
    get_next_task,
    update_task_status,
    decrease_minutes,
    get_task_by_id
)

from services import (
    extract_audio,
    translate_text,
    generate_cloned_audio,
    assemble_video
)

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEB_APP_URL = os.getenv("WEB_APP_URL")
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL")
UPLOAD_DIR = os.getenv("UPLOAD_DIR")
FINAL_DIR = os.getenv("FINAL_DIR")
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", 180))

CRYPTOMUS_API_KEY = os.getenv("CRYPTOMUS_API_KEY")
CRYPTOMUS_MERCHANT_ID = os.getenv("CRYPTOMUS_MERCHANT_ID")
SUBSCRIPTION_AMOUNT = os.getenv("SUBSCRIPTION_AMOUNT")
SUBSCRIPTION_CREDITS = int(os.getenv("SUBSCRIPTION_CREDITS"))

VITE_OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")

# -------------------------------
# FASTAPI INIT
# -------------------------------
app = FastAPI()
app.mount("/media", StaticFiles(directory=FINAL_DIR), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# -------------------------------
# OPENROUTER WHISPER
# -------------------------------
async def transcribe_audio_remote(audio_path: str) -> tuple[str, str]:
    """Транскрибация через OpenRouter Whisper API"""
    url = "https://openrouter.ai/api/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {VITE_OPENROUTER_KEY}"}

    # Открываем файл как бинарный
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/mp3")}
        data = {"model": "whisper-1"}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, files=files, data=data)
            resp.raise_for_status()
            result = resp.json()
            text = result.get("text", "")
            language = result.get("language", "unknown")
            return text, language

# -------------------------------
# AUTH / CODE
# -------------------------------
@app.get("/generate-code")
def generate_code():
    code = str(uuid.uuid4())[:6].upper()
    create_user(code)
    return {"code": code}


@app.get("/status")
def status(code: str):
    user = get_user_by_code(code)
    if not user:
        return {"authorized": False}

    return {
        "authorized": user["telegram_id"] is not None,
        "minutes_left": user["minutes_left"]
    }

# -------------------------------
# CRYPTOMUS PAYMENT
# -------------------------------
@app.post("/create-payment")
def create_payment(code: str):
    user = get_user_by_code(code)
    if not user:
        return {"error": "user not found"}

    payload = {
        "amount": SUBSCRIPTION_AMOUNT,
        "currency": "USD",
        "merchant_id": CRYPTOMUS_MERCHANT_ID,
        "order_id": str(uuid.uuid4()),
        "description": f"Subscription for {SUBSCRIPTION_CREDITS} videos",
        "callback_url": f"{SERVER_BASE_URL}/cryptomus-callback?code={code}",
        "success_url": f"{WEB_APP_URL}?status=success",
        "fail_url": f"{WEB_APP_URL}?status=fail"
    }

    headers = {
        "Authorization": f"Bearer {CRYPTOMUS_API_KEY}",
        "Content-Type": "application/json"
    }

    import requests
    response = requests.post(os.getenv("CRYPTOMUS_CREATE_URL"), json=payload, headers=headers)
    return response.json()


@app.get("/cryptomus-callback")
def cryptomus_callback(code: str, status: str, order_id: str):
    user = get_user_by_code(code)
    if not user:
        return {"error": "user not found"}

    if status == "success":
        decrease_minutes(user["id"], -SUBSCRIPTION_CREDITS)  # минус для пополнения
        return {"status": "success", "message": f"{SUBSCRIPTION_CREDITS} videos added"}

    return {"status": "fail", "message": "Payment failed"}

# -------------------------------
# UPLOAD & TASK
# -------------------------------
@app.post("/translate")
async def translate_video(
    video: UploadFile,
    code: str = Form(...),
    target_language: str = Form(...)
):
    user = get_user_by_code(code)
    if not user or user["minutes_left"] <= 0:
        return {"error": "limit reached"}

    filename = f"{uuid.uuid4()}.mp4"
    video_path = os.path.join(UPLOAD_DIR, filename)

    with open(video_path, "wb") as f:
        f.write(await video.read())

    task_id = add_task(
        user_id=user["id"],
        video_path=video_path,
        language=target_language
    )

    return {"task_id": task_id}


@app.get("/task-status")
def task_status(task_id: int):
    task = get_task_by_id(task_id)
    if not task:
        return {"error": "not found"}

    return {
        "status": task["status"],
        "video_url": f"/media/{os.path.basename(task['result_path'])}" if task["result_path"] else None
    }

# -------------------------------
# WORKER (Очередь)
# -------------------------------
async def worker():
    while True:
        task = get_next_task()
        if not task:
            await asyncio.sleep(2)
            continue

        try:
            update_task_status(task["id"], "processing")

            # 1. Extract audio
            audio_path = extract_audio(task["video_path"])

            # 2. Transcribe через OpenRouter API
            text, source_lang = await transcribe_audio_remote(audio_path)

            # 3. Translate
            translated_text = translate_text(
                text=text,
                source_language_code=source_lang,
                target_language=task["language"]
            )

            # 4. Generate dubbed audio
            dubbed_audio = generate_cloned_audio(translated_text, audio_path)

            # 5. Assemble final video
            final_video = assemble_video(task["video_path"], dubbed_audio)

            # 6. Update task status
            update_task_status(task["id"], "done", final_video)

            # 7. Decrease user minutes
            decrease_minutes(task["user_id"], 1)

        except Exception as e:
            update_task_status(task["id"], "error")
            print("TASK ERROR:", e)

        await asyncio.sleep(1)


@app.on_event("startup")
async def startup():
    # Запускаем worker асинхронно
    asyncio.create_task(worker())
