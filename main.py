# main.py — API + Telegram (NO heavy tasks)

import os
import uuid
import json
import urllib.parse

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from db import (
    create_user,
    get_user_by_code,
    bind_telegram,
    add_task,
    get_task_by_id
)

# ================= ENV =================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEB_APP_URL = os.getenv("WEB_APP_URL")
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL")

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
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text="🚀 Open SmartDub",
                web_app={"url": WEB_APP_URL}
            )]
        ]
    )
    await message.answer(
        "🎬 Welcome to SmartDub\nAI video dubbing inside Telegram",
        reply_markup=kb
    )

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = types.Update(**await request.json())
    await dp.feed_update(bot, update)
    return {"ok": True}

# ================= USER =================
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
        "authorized": bool(user["telegram_id"]),
        "minutes_left": user["minutes_left"]
    }

# ================= AUTH =================
class TelegramAuth(BaseModel):
    code: str
    init_data: str

@app.post("/auth-telegram")
def auth_telegram(data: TelegramAuth):
    user = get_user_by_code(data.code)
    if not user:
        return {"success": False}

    parsed = dict(urllib.parse.parse_qsl(data.init_data))
    tg_user = json.loads(parsed.get("user", "{}"))
    telegram_id = tg_user.get("id")

    if not telegram_id:
        return {"success": False}

    bind_telegram(data.code, telegram_id)
    return {"success": True, "minutes_left": user["minutes_left"]}

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

    path = f"{UPLOAD_DIR}/{uuid.uuid4()}.mp4"
    with open(path, "wb") as f:
        f.write(await video.read())

    task_id = add_task(user["id"], path, target_language)
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

# ================= STARTUP =================
@app.on_event("startup")
async def startup():
    await bot.set_webhook(f"{SERVER_BASE_URL}/telegram/webhook")

@app.get("/")
def health():
    return {"ok": True}
