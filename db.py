import sqlite3
import os

DB_PATH = "users.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Создаём таблицы
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id TEXT,
    code TEXT,
    plan TEXT DEFAULT 'free',
    minutes_left INTEGER DEFAULT 3,
    video_credits INTEGER DEFAULT 10,
    active_tasks INTEGER DEFAULT 0
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    status TEXT,
    video_path TEXT,
    result_path TEXT,
    language TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()


# -------------------------------
# Функции для работы с пользователями
# -------------------------------
def create_user(code, plan="free"):
    """Создаём пользователя с тарифом и настройками по умолчанию"""
    default_credits = {
        "free_trial": 1,
        "free": 1,
        "starter": 10,
        "pro": 50,
        "advanced": 200
    }
    default_minutes = {
        "free_trial": 3,
        "free": 3,
        "starter": 10*10,
        "pro": 50*60,
        "advanced": 200*120
    }
    credits = default_credits.get(plan, 1)
    minutes = default_minutes.get(plan, 3)

    cursor.execute(
        "INSERT INTO users (code, plan, minutes_left, video_credits) VALUES (?, ?, ?, ?)",
        (code, plan, minutes, credits)
    )
    conn.commit()


def bind_telegram(code, telegram_id):
    cursor.execute(
        "UPDATE users SET telegram_id=? WHERE code=?",
        (telegram_id, code)
    )
    conn.commit()


def get_user_by_code(code):
    cursor.execute(
        "SELECT id, telegram_id, code, plan, minutes_left, video_credits FROM users WHERE code=?",
        (code,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "telegram_id": row[1],
        "code": row[2],
        "plan": row[3],
        "minutes_left": row[4],
        "video_credits": row[5],
    }


def decrease_minutes(user_id, minutes):
    cursor.execute(
        "UPDATE users SET minutes_left = minutes_left - ? WHERE id=?",
        (minutes, user_id)
    )
    conn.commit()


def decrease_video_credits(user_id, credits=1):
    cursor.execute(
        "UPDATE users SET video_credits = video_credits - ? WHERE id=?",
        (credits, user_id)
    )
    conn.commit()


# -------------------------------
# Функции для работы с задачами
# -------------------------------
def add_task(user_id, video_path, language):
    """Добавляем задачу только если есть видео-кредиты"""
    user = cursor.execute("SELECT video_credits FROM users WHERE id=?", (user_id,)).fetchone()
    if not user or user[0] <= 0:
        return None  # Нет кредитов

    cursor.execute(
        "INSERT INTO tasks (user_id, video_path, language, status) VALUES (?, ?, ?, ?)",
        (user_id, video_path, language, "queued")
    )
    conn.commit()

    decrease_video_credits(user_id, 1)  # Уменьшаем кредит на одно видео

    return cursor.lastrowid


def get_next_task():
    cursor.execute("""
        SELECT id, user_id, status, video_path, result_path, language
        FROM tasks
        WHERE status = 'queued'
        ORDER BY created_at
        LIMIT 1
    """)
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "user_id": row[1],
        "status": row[2],
        "video_path": row[3],
        "result_path": row[4],
        "language": row[5],
    }


def update_task_status(task_id, status, result_path=None):
    if result_path:
        cursor.execute(
            "UPDATE tasks SET status=?, result_path=? WHERE id=?",
            (status, result_path, task_id)
        )
    else:
        cursor.execute(
            "UPDATE tasks SET status=? WHERE id=?",
            (status, task_id)
        )
    conn.commit()


def get_task_by_id(task_id: int):
    cursor.execute(
        "SELECT id, user_id, status, video_path, result_path FROM tasks WHERE id = ?",
        (task_id,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "user_id": row[1],
        "status": row[2],
        "video_path": row[3],
        "result_path": row[4],
    }

