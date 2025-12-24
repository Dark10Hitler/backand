# main.py — FINAL / Render FREE compatible

import os
import uuid
import json
import urllib.parse
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
    transcribe_audio,
    translate_text,
    generate_cloned_audio,
    assemble_video
)

# ================= ENV =================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEB_APP_URL = os.getenv("WEB_APP_URL")
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL")
OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")

UPLOAD_DIR = "/tmp/uploads"
FINAL_DIR = "/tmp/final_videos"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ================= APP =================
app = FastAPI()

app.mount("/media", StaticFiles(directory=FINAL_DIR), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= TELEGRAM =================
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(
                text="🚀 Open SmartDub",
                web_app={"url": WEB_APP_URL}
            )
        ]]
    )
    await message.answer(
        "🎬 SmartDub — AI Video Dubbing",
        reply_markup=keyboard
    )

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = types.Update(**await request.json())
    await dp.feed_update(bot, update)
    return {"ok": True}

# ================= OPENROUTER =================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY
)

# ================= EXECUTOR =================
executor = ThreadPoolExecutor(max_workers=1)
processing_lock = asyncio.Lock()

# ================= USER =================
@app.get("/generate-code")
def generate_code():
    code = str(uuid.uuid4())[:6].upper()
    create_user(code)
    return {"code": code}

@app.get("/status")
def status(code: str):
    user = get_user_by_code(code)
    return {
        "authorized": bool(user and user["telegram_id"]),
        "minutes_left": user["minutes_left"] if user else 0
    }

# ================= AUTH =================
class TelegramAuth(BaseModel):
    code: str
    init_data: str

@app.post("/auth-telegram")
def auth(data: TelegramAuth):
    user = get_user_by_code(data.code)
    if not user:
        return {"success": False}

    parsed = dict(urllib.parse.parse_qsl(data.init_data))
    tg_user = json.loads(parsed.get("user", "{}"))

    if not tg_user.get("id"):
        return {"success": False}

    bind_telegram(data.code, tg_user["id"])
    return {"success": True}

# ================= TRANSLATE =================
@app.post("/translate")
async def translate(
    video: UploadFile,
    code: str = Form(...),
    target_language: str = Form(...)
):
    user = get_user_by_code(code)
    if not user or user["minutes_left"] <= 0:
        return {"error": "limit"}

    video_path = f"{UPLOAD_DIR}/{uuid.uuid4()}.mp4"
    with open(video_path, "wb") as f:
        f.write(await video.read())

    task_id = add_task(user["id"], video_path, target_language)

    # 🔥 запускаем обработку очереди
    asyncio.create_task(process_queue())

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

# ================= QUEUE PROCESSOR =================
async def process_queue():
    async with processing_lock:
        task = get_next_task()
        if not task:
            return

        update_task_status(task["id"], "processing")
        loop = asyncio.get_running_loop()

        try:
            audio = await loop.run_in_executor(
                executor, extract_audio, task["video_path"]
            )

            text, src_lang = await loop.run_in_executor(
                executor, transcribe_audio, audio
            )

            translated = await loop.run_in_executor(
                executor,
                translate_text,
                text,
                src_lang,
                task["language"],
                client
            )

            dubbed_audio = await loop.run_in_executor(
                executor,
                generate_cloned_audio,
                translated,
                audio
            )

            final_video = await loop.run_in_executor(
                executor,
                assemble_video,
                task["video_path"],
                dubbed_audio
            )

            update_task_status(task["id"], "done", final_video)
            decrease_minutes(task["user_id"], 1)

        except Exception as e:
            print("❌ PROCESS ERROR:", repr(e))
            update_task_status(task["id"], "error")

# ================= STARTUP =================
@app.on_event("startup")
async def startup():
    await bot.set_webhook(f"{SERVER_BASE_URL}/telegram/webhook")

@app.get("/")
def health():
    return {"ok": True}
