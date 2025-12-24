# main.py — PRODUCTION READY
# FastAPI + aiogram 3.x + Telegram Web App Auth (FIXED)

import os
import uuid
import asyncio
import json
import urllib.parse
import httpx

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from openai import OpenAI

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

# ===============================
# ENV
# ===============================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEB_APP_URL = os.getenv("WEB_APP_URL")
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
FINAL_DIR = os.getenv("FINAL_DIR", "/tmp/final_videos")

VITE_OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")

if not BOT_TOKEN or not WEB_APP_URL or not SERVER_BASE_URL:
    raise RuntimeError("BOT_TOKEN, WEB_APP_URL, SERVER_BASE_URL are required")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ===============================
# APP
# ===============================
app = FastAPI()

app.mount("/media", StaticFiles(directory=FINAL_DIR), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# OPENROUTER
# ===============================
client_router = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=VITE_OPENROUTER_KEY
)

# ===============================
# TELEGRAM BOT
# ===============================
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🚀 Open SmartDub",
                    web_app={"url": WEB_APP_URL}
                )
            ]
        ]
    )
    await message.answer(
        "🎬 Welcome to SmartDub\n\nAI-powered video dubbing inside Telegram.",
        reply_markup=keyboard
    )

# ===============================
# WEBHOOK
# ===============================
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = types.Update(**await request.json())
    await dp.feed_update(bot, update)
    return {"ok": True}

# ===============================
# USER CODE
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
        return {"authorized": False, "minutes_left": 0}

    return {
        "authorized": bool(user["telegram_id"]),
        "minutes_left": user["minutes_left"]
    }

# ===============================
# TELEGRAM AUTH (FIXED)
# ===============================
class TelegramAuthRequest(BaseModel):
    code: str
    init_data: str

@app.post("/auth-telegram")
def auth_telegram(req: TelegramAuthRequest):
    user = get_user_by_code(req.code)
    if not user:
        return {"success": False, "message": "User not found"}

    # Parse initData
    parsed = dict(urllib.parse.parse_qsl(req.init_data))
    user_raw = parsed.get("user")

    if not user_raw:
        return {"success": False, "message": "Invalid initData"}

    try:
        tg_user = json.loads(user_raw)
        telegram_id = tg_user.get("id")
    except Exception:
        return {"success": False, "message": "Invalid Telegram user data"}

    if not telegram_id:
        return {"success": False, "message": "Telegram ID missing"}

    # IMPORTANT: bind by CODE (as in your db.py)
    bind_telegram(req.code, telegram_id)

    return {
        "success": True,
        "minutes_left": user["minutes_left"]
    }

# ===============================
# VIDEO TRANSLATION
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

    task_id = add_task(user["id"], video_path, target_language)
    if not task_id:
        return {"error": "no credits"}

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
# WHISPER (REMOTE)
# ===============================
async def transcribe_audio_remote(audio_path: str):
    url = "https://openrouter.ai/api/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {VITE_OPENROUTER_KEY}"}

    with open(audio_path, "rb") as f:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                url,
                headers=headers,
                files={"file": (os.path.basename(audio_path), f, "audio/mp3")},
                data={"model": "whisper-1"}
            )
            resp.raise_for_status()
            r = resp.json()
            return r.get("text", ""), r.get("language", "unknown")

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
            text, src_lang = await transcribe_audio_remote(audio_path)

            translated = translate_text(
                text=text,
                source_language_code=src_lang,
                target_language=task["language"],
                client_router=client_router
            )

            dubbed_audio = generate_cloned_audio(translated, audio_path)
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
    await bot.set_webhook(f"{SERVER_BASE_URL}/telegram/webhook")
    print("✅ Telegram webhook set")

# ===============================
# HEALTH
# ===============================
@app.get("/")
def root():
    return {"status": "ok"}
