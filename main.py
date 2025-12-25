import os
import uuid
import json
import urllib.parse
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from db import (
    create_user, get_user_by_code, bind_telegram,
    add_task, get_next_task, update_task_status,
    decrease_minutes, get_task_by_id
)

from services import (
    extract_audio, transcribe_audio, translate_text,
    generate_cloned_audio, assemble_video, cleanup_files, 
    FINAL_DIR, UPLOAD_DIR
)

load_dotenv()

# Настройки
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEB_APP_URL = os.getenv("WEB_APP_URL")
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL").rstrip('/')

app = FastAPI()

# Раздача статики (финальных видео)
app.mount("/media", StaticFiles(directory=FINAL_DIR), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация Telegram
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="🚀 Open SmartDub", web_app={"url": WEB_APP_URL})
        ]]
    )
    await message.answer("🎬 SmartDub AI: Озвучка видео через нейросети.", reply_markup=keyboard)

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = types.Update(**data)
    await dp.feed_update(bot, update)
    return {"ok": True}

# Обработка задач в 1 поток для стабильности
executor = ThreadPoolExecutor(max_workers=1)
processing_lock = asyncio.Lock()

@app.get("/generate-code")
def handle_generate_code():
    code = str(uuid.uuid4())[:6].upper()
    create_user(code)
    return {"code": code}

@app.get("/status")
def handle_status(code: str):
    user = get_user_by_code(code)
    return {
        "authorized": bool(user and user.get("telegram_id")),
        "minutes_left": user["minutes_left"] if user else 0
    }

class TelegramAuth(BaseModel):
    code: str
    init_data: str

@app.post("/auth-telegram")
def handle_auth(data: TelegramAuth):
    user = get_user_by_code(data.code)
    if not user: return {"success": False}
    try:
        parsed = dict(urllib.parse.parse_qsl(data.init_data))
        tg_user = json.loads(parsed.get("user", "{}"))
        bind_telegram(data.code, tg_user["id"])
        return {"success": True}
    except:
        return {"success": False}

@app.post("/translate")
async def handle_translate(
    background_tasks: BackgroundTasks,
    video: UploadFile,
    code: str = Form(...),
    target_language: str = Form(...)
):
    user = get_user_by_code(code)
    if not user or user["minutes_left"] <= 0:
        return {"error": "limit"}

    v_path = f"{UPLOAD_DIR}/{uuid.uuid4()}.mp4"
    with open(v_path, "wb") as f:
        f.write(await video.read())

    task_id = add_task(user["id"], v_path, target_language)
    background_tasks.add_task(run_queue) # Запуск воркера
    
    return {"task_id": task_id}

@app.get("/task-status")
def handle_task_status(task_id: int):
    task = get_task_by_id(task_id)
    if not task: return {"error": "not found"}
    
    url = f"/media/{os.path.basename(task['result_path'])}" if task["result_path"] else None
    return {"status": task["status"], "video_url": url}

async def run_queue():
    if processing_lock.locked(): return
    async with processing_lock:
        while True:
            task = get_next_task()
            if not task: break
            
            update_task_status(task["id"], "processing")
            loop = asyncio.get_running_loop()
            
            temp_files = []
            try:
                # 1. Звук
                audio = await loop.run_in_executor(executor, extract_audio, task["video_path"])
                temp_files.append(audio)
                
                # 2. Текст (OpenRouter)
                text, _ = await loop.run_in_executor(executor, transcribe_audio, audio)
                
                # 3. Перевод (OpenRouter)
                translated = await loop.run_in_executor(executor, translate_text, text, task["language"])
                
                # 4. Клон голоса (ElevenLabs)
                dubbed = await loop.run_in_executor(executor, generate_cloned_audio, translated)
                temp_files.append(dubbed)
                
                # 5. Сборка видео
                final = await loop.run_in_executor(executor, assemble_video, task["video_path"], dubbed)
                
                update_task_status(task["id"], "done", final)
                decrease_minutes(task["user_id"], 1)
                
            except Exception as e:
                print(f"Queue Error: {e}")
                update_task_status(task["id"], "error")
            finally:
                # Удаляем оригинал и временные аудио
                cleanup_files(task["video_path"], *temp_files)

@app.on_event("startup")
async def on_startup():
    await bot.set_webhook(f"{SERVER_BASE_URL}/telegram/webhook")

@app.get("/")
def health():
    return {"status": "ok"}
