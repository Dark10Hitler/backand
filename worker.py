# worker.py — background processing ONLY

import asyncio
import os
from dotenv import load_dotenv
from openai import OpenAI

from db import (
    get_next_task,
    update_task_status,
    decrease_minutes
)

from services import (
    extract_audio,
    translate_text,
    generate_cloned_audio,
    assemble_video
)

load_dotenv()

VITE_OPENROUTER_KEY = os.getenv("VITE_OPENROUTER_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=VITE_OPENROUTER_KEY
)

async def worker():
    print("🟢 WORKER STARTED")

    while True:
        task = get_next_task()

        if not task:
            await asyncio.sleep(2)
            continue

        try:
            update_task_status(task["id"], "processing")

            audio = extract_audio(task["video_path"])
            text, src = translate_text.transcribe(audio)
            translated = translate_text(
                text=text,
                source_language_code=src,
                target_language=task["language"],
                client_router=client
            )

            voice = generate_cloned_audio(translated, audio)
            final = assemble_video(task["video_path"], voice)

            update_task_status(task["id"], "done", final)
            decrease_minutes(task["user_id"], 1)

            print(f"✅ DONE {task['id']}")

        except Exception as e:
            update_task_status(task["id"], "error")
            print("❌ ERROR:", e)

        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(worker())
