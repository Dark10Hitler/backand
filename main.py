import os
import uuid
import asyncio
import httpx
import requests
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

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

from openai import OpenAI

# ===============================
# LOAD ENV
# ===============================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEB_APP_URL = os.getenv("WEB_APP_URL")
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
FINAL_DIR = os.getenv("FINAL_DIR", "/tmp/final_videos")
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", 180))

CRYPTOMUS_API_KEY = os.getenv("CRYPTOMUS_API_KEY")
CRYPTOMUS_MERCHANT_ID = os.getenv("CRYPTOMUS_MERCHANT_ID")
CRYPTOMUS_CREATE_URL = os.getenv("CRYPTOMUS_CREATE_URL")
SUBSCRIPTION_AMOUNT = os.getenv("SUBSCRIPTION_AMOUNT")
SUBSCRIPTION_CREDITS = int(os.getenv("SUBSCRIPTION_CREDITS", 10))

VITE_OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")

if not BOT_TOKEN or not WEB_APP_URL or not SERVER_BASE_URL:
    raise RuntimeError("BOT_TOKEN, WEB_APP_URL, SERVER_BASE_URL are required")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ===============================
# FASTAPI
# ===============================
app = FastAPI()

app.mount("/media", StaticFiles(directory=FINAL_DIR), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# ===============================
# OPENROUTER CLIENT
# ===============================
client_router = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=VITE_OPENROUTER_KEY
)

# ===============================
# TELEGRAM BOT (WEBHOOK)
# ===============================
telegram_app = Application.builder().token(BOT_TOKEN).build()

async def tg_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 Open SmartDub", web_app={"url": WEB_APP_URL})]
    ])

    await update.message.reply_text(
        "🎬 Welcome to SmartDub\n\n"
        "AI-powered video dubbing directly inside Telegram.",
        reply_markup=keyboard
    )

telegram_app.add_handler(CommandHandler("start", tg_start))

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}

# ===============================
# OPENROUTER WHISPER
# ===============================
async def transcribe_audio_remote(audio_path: str) -> tuple[str, str]:
    url = "https://openrouter.ai/api/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {VITE_OPENROUTER_KEY}"}

    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/mp3")}
        data = {"model": "whisper-1"}

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=headers, files=files, data=data)
            resp.raise_for_status()
            result = resp.json()

            return result.get("text", ""), result.get("language", "unknown")

# ===============================
# AUTH / CODE
# ===============================
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

# ===============================
# PAYMENT (CRYPTOMUS)
# ===============================
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
        "description": f"{SUBSCRIPTION_CREDITS} video credits",
        "callback_url": f"{SERVER_BASE_URL}/cryptomus-callback?code={code}",
        "success_url": f"{WEB_APP_URL}?status=success",
        "fail_url": f"{WEB_APP_URL}?status=fail"
    }

    headers = {
        "Authorization": f"Bearer {CRYPTOMUS_API_KEY}",
        "Content-Type": "application/json"
    }

    return requests.post(CRYPTOMUS_CREATE_URL, json=payload, headers=headers).json()

@app.get("/cryptomus-callback")
def cryptomus_callback(code: str, status: str):
    user = get_user_by_code(code)
    if not user:
        return {"error": "user not found"}

    if status == "success":
        decrease_minutes(user["id"], -SUBSCRIPTION_CREDITS)
        return {"status": "success"}

    return {"status": "fail"}

# ===============================
# UPLOAD & TASK
# ===============================
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
        "video_url": f"/media/{os.path.basename(task['result_path'])}"
        if task["result_path"] else None
    }

# ===============================
# WORKER
# ===============================
async def worker():
    while True:
        task = get_next_task()
        if not task:
            await asyncio.sleep(2)
            continue

        try:
            update_task_status(task["id"], "processing")

            audio_path = extract_audio(task["video_path"])
            text, source_lang = await transcribe_audio_remote(audio_path)

            translated_text = translate_text(
                text=text,
                source_language_code=source_lang,
                target_language=task["language"],
                client_router=client_router
            )

            dubbed_audio = generate_cloned_audio(translated_text, audio_path)
            final_video = assemble_video(task["video_path"], dubbed_audio)

            update_task_status(task["id"], "done", final_video)
            decrease_minutes(task["user_id"], 1)

        except Exception as e:
            update_task_status(task["id"], "error")
            print("TASK ERROR:", e)

        await asyncio.sleep(1)

# ===============================
# STARTUP
# ===============================
@app.on_event("startup")
async def startup():
    asyncio.create_task(worker())

    await telegram_app.initialize()
    await telegram_app.bot.set_webhook(
        f"{SERVER_BASE_URL}/telegram/webhook"
    )

    print("✅ Telegram webhook set")

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def root():
    return {"status": "ok"}
