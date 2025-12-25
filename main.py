import os
import uuid
import json
import asyncio
import urllib.parse
from fastapi import FastAPI, UploadFile, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from db import (
    create_user, get_user_by_code, bind_telegram, 
    add_task, get_next_task, update_task_status, 
    decrease_minutes, get_task_by_id
)
from services import (
    extract_audio, transcribe_audio, translate_text, 
    generate_cloned_audio, assemble_video, cleanup_files, FINAL_DIR
)

load_dotenv()

app = FastAPI(title="Global Voice Ads API")

# Настройка CORS для работы с Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Раздача готовых видео файлов
app.mount("/media", StaticFiles(directory=FINAL_DIR), name="media")

# Глобальный замок для обработки задач по одной (важно для Render Free)
processing_lock = asyncio.Lock()

@app.get("/")
async def health_check():
    return {"status": "working", "provider": "Render Free"}

@app.get("/generate-code")
def handle_generate_code():
    code = str(uuid.uuid4())[:6].upper()
    create_user(code)
    return {"code": code}

@app.get("/status")
def handle_status(code: str):
    user = get_user_by_code(code)
    if not user:
        return {"authorized": False, "minutes_left": 0}
    return {
        "authorized": bool(user.get("telegram_id")),
        "minutes_left": user.get("minutes_left", 0)
    }

@app.post("/translate")
async def handle_translate(
    background_tasks: BackgroundTasks,
    video: UploadFile,
    code: str = Form(...),
    target_language: str = Form(...) # Здесь может быть страна, например "Brazil"
):
    user = get_user_by_code(code)
    if not user or user["minutes_left"] <= 0:
        return JSONResponse(status_code=403, content={"error": "no_minutes"})

    # Сохраняем видео
    temp_video_path = f"/tmp/{uuid.uuid4()}_{video.filename}"
    with open(temp_video_path, "wb") as f:
        f.write(await video.read())

    # Добавляем задачу в БД
    task_id = add_task(user["id"], temp_video_path, target_language)
    
    # Запускаем фоновый процесс обработки
    background_tasks.add_task(process_task_logic)

    return {"task_id": task_id, "status": "queued"}

@app.get("/task-status")
def handle_task_status(task_id: int):
    task = get_task_by_id(task_id)
    if not task:
        return {"error": "not_found"}
    
    video_url = None
    if task["status"] == "done" and task["result_path"]:
        video_url = f"/media/{os.path.basename(task['result_path'])}"
        
    return {
        "status": task["status"],
        "video_url": video_url
    }

async def process_task_logic():
    """Фоновая логика обработки очереди"""
    if processing_lock.locked():
        return

    async with processing_lock:
        while True:
            task = get_next_task()
            if not task:
                break

            update_task_status(task["id"], "processing")
            
            temp_audio = None
            dubbed_audio = None
            
            try:
                # 1. Извлечение
                temp_audio = extract_audio(task["video_path"])
                
                # 2. Транскрибация (Groq)
                original_text, _ = transcribe_audio(temp_audio)
                
                # 3. Перевод (OpenRouter)
                localized_text = translate_text(original_text, task["language"])
                
                # 4. Озвучка (ElevenLabs)
                dubbed_audio = generate_cloned_audio(localized_text)
                
                # 5. Сборка видео
                final_video_path = assemble_video(task["video_path"], dubbed_audio)
                
                # Завершение
                update_task_status(task["id"], "done", final_video_path)
                decrease_minutes(task["user_id"], 1)
                
            except Exception as e:
                print(f"Critical Task Error ID {task['id']}: {str(e)}")
                update_task_status(task["id"], "error")
            finally:
                # Очистка исходного видео и временного аудио
                cleanup_files(task["video_path"], temp_audio, dubbed_audio)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
