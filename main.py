#!/usr/bin/env python3
"""
Telegram-бот с командой из 3 AI-агентов на базе Groq API.

Агенты:
  🗓 Ассистент  — задачи, расписание, напоминания
  🔍 Ресёрчер   — поиск и анализ информации
  ✍️ Копирайтер — тексты, структура презентаций

Ключевая фишка: команда /profile сохраняет описание пользователя,
и все агенты используют его для персонализированных ответов.
"""

import os
import logging
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

load_dotenv()

TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY_MESSAGES = 20
RESTART_DELAY = 5

PROFILES_DIR = Path("profiles")
PROFILES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Определения агентов
# ---------------------------------------------------------------------------

AGENTS: dict[str, dict] = {
    "assistant": {
        "name": "🗓 Ассистент",
        "description": "Задачи, расписание, напоминания",
        "system_prompt": (
            "Ты персональный ассистент, специализирующийся на управлении задачами, "
            "планировании расписания и напоминаниях. "
            "Помогаешь пользователю организовать день, расставить приоритеты "
            "и не забыть важные дела. "
            "Давай конкретные, структурированные ответы с чёткими шагами "
            "и временными рамками. Используй списки и чёткую структуру.\n"
            "{profile_section}"
        ),
    },
    "researcher": {
        "name": "🔍 Ресёрчер",
        "description": "Поиск и анализ информации",
        "system_prompt": (
            "Ты эксперт по поиску и глубокому анализу информации. "
            "Умеешь исследовать темы, находить закономерности, "
            "сравнивать данные и делать обоснованные выводы. "
            "Структурируй информацию логично: факты → анализ → выводы → "
            "направления для дальнейшего изучения. "
            "Указывай, где данные надёжны, а где требуют проверки.\n"
            "{profile_section}"
        ),
    },
    "copywriter": {
        "name": "✍️ Копирайтер",
        "description": "Тексты, структура презентаций",
        "system_prompt": (
            "Ты профессиональный копирайтер и контент-стратег. "
            "Создаёшь убедительные тексты, структуры презентаций, "
            "посты для соцсетей, статьи и маркетинговые материалы. "
            "Адаптируешь стиль под аудиторию и платформу. "
            "Всегда предлагаешь несколько вариантов, объясняешь выбор формата "
            "и даёшь рекомендации по улучшению.\n"
            "{profile_section}"
        ),
    },
}

# ---------------------------------------------------------------------------
# In-memory хранилище сессий
# ---------------------------------------------------------------------------

sessions: dict[int, dict] = {}


def get_session(user_id: int) -> dict:
    if user_id not in sessions:
        sessions[user_id] = {
            "agent": "assistant",
            "history": [],
            "awaiting_profile": False,
        }
    return sessions[user_id]


# ---------------------------------------------------------------------------
# Профили пользователей
# ---------------------------------------------------------------------------


def get_profile(user_id: int) -> str | None:
    profile_path = PROFILES_DIR / f"{user_id}.txt"
    if profile_path.exists():
        return profile_path.read_text(encoding="utf-8").strip()
    return None


def save_profile(user_id: int, profile_text: str) -> None:
    profile_path = PROFILES_DIR / f"{user_id}.txt"
    profile_path.write_text(profile_text.strip(), encoding="utf-8")
    logger.info("Профиль сохранён: %s", profile_path)


# ---------------------------------------------------------------------------
# Сборка промптов и клавиатуры
# ---------------------------------------------------------------------------


def build_system_prompt(agent_key: str, user_id: int) -> str:
    profile = get_profile(user_id)
    if profile:
        profile_section = (
            "\n\n--- ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ ---\n"
            f"{profile}\n"
            "--- КОНЕЦ ПРОФИЛЯ ---\n\n"
            "Обязательно учитывай эту информацию: адаптируй стиль, глубину "
            "и примеры под конкретного человека."
        )
    else:
        profile_section = (
            "\n\nПрофиль пользователя пока не заполнен. "
            "При первой возможности мягко предложи использовать команду /profile — "
            "это сделает ответы значительно более персонализированными."
        )
    return AGENTS[agent_key]["system_prompt"].format(profile_section=profile_section)


def build_agent_keyboard() -> dict:
    buttons = [
        [{"text": f"{data['name']} — {data['description']}",
          "callback_data": f"switch_agent:{key}"}]
        for key, data in AGENTS.items()
    ]
    return {"inline_keyboard": buttons}


# ---------------------------------------------------------------------------
# Telegram API
# ---------------------------------------------------------------------------

_TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def _tg(method: str, payload: dict | None = None) -> dict:
    resp = requests.post(f"{_TG_BASE}/{method}", json=payload or {}, timeout=35)
    resp.raise_for_status()
    return resp.json()


def tg_send(chat_id: int, text: str, parse_mode: str | None = None,
            reply_markup: dict | None = None) -> None:
    payload: dict = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup
    _tg("sendMessage", payload)


def tg_edit(chat_id: int, message_id: int, text: str,
            parse_mode: str | None = None) -> None:
    payload: dict = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    _tg("editMessageText", payload)


def tg_answer_callback(callback_query_id: str) -> None:
    _tg("answerCallbackQuery", {"callback_query_id": callback_query_id})


def tg_typing(chat_id: int) -> None:
    try:
        _tg("sendChatAction", {"chat_id": chat_id, "action": "typing"})
    except Exception:
        pass


def tg_get_updates(offset: int | None, timeout: int = 30) -> list[dict]:
    params: dict = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
    if offset is not None:
        params["offset"] = offset
    resp = requests.post(f"{_TG_BASE}/getUpdates", json=params, timeout=timeout + 5)
    resp.raise_for_status()
    return resp.json().get("result", [])


def send_long_message(chat_id: int, text: str) -> None:
    limit = 4096
    for i in range(0, len(text), limit):
        tg_send(chat_id, text[i: i + limit])


# ---------------------------------------------------------------------------
# Groq API
# ---------------------------------------------------------------------------


def ask_groq(agent_key: str, user_id: int, history: list[dict],
             user_message: str) -> str:
    system_prompt = build_system_prompt(agent_key, user_id)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.7,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error("Ошибка Groq API: %s", exc)
        return f"⚠️ Произошла ошибка при обращении к AI: {exc}"


# ---------------------------------------------------------------------------
# Обработчики команд
# ---------------------------------------------------------------------------


def handle_start(chat_id: int, user_id: int, first_name: str) -> None:
    session = get_session(user_id)
    current = AGENTS[session["agent"]]["name"]
    text = (
        f"👋 Привет, {first_name}!\n\n"
        "Я — команда из трёх AI-агентов, каждый заточен под свою задачу:\n\n"
        "🗓 *Ассистент* — задачи, расписание, напоминания\n"
        "🔍 *Ресёрчер* — поиск и анализ информации\n"
        "✍️ *Копирайтер* — тексты, структура презентаций\n\n"
        "💡 Команда /profile позволяет рассказать о себе — "
        "тогда все агенты будут давать персонализированные ответы.\n\n"
        f"Сейчас активен: *{current}*\n"
        "Выбери агента или сразу задай вопрос:"
    )
    tg_send(chat_id, text, parse_mode="Markdown", reply_markup=build_agent_keyboard())


def handle_menu(chat_id: int, user_id: int) -> None:
    session = get_session(user_id)
    current = AGENTS[session["agent"]]["name"]
    tg_send(
        chat_id,
        f"🤖 Активный агент: *{current}*\n\nВыбери агента:",
        parse_mode="Markdown",
        reply_markup=build_agent_keyboard(),
    )


def handle_profile(chat_id: int, user_id: int) -> None:
    session = get_session(user_id)
    existing = get_profile(user_id)
    current_block = f"📋 *Твой текущий профиль:*\n{existing}\n\n---\n\n" if existing else ""
    text = (
        f"{current_block}"
        "✏️ *Напиши свой профиль* следующим сообщением.\n\n"
        "Включи любую полезную информацию:\n"
        "• Кто ты — профессия, роль, область\n"
        "• Твои цели и текущие задачи\n"
        "• Предпочтения по стилю ответов\n"
        "• Примеры хороших ответов или требования\n\n"
        "*Пример:*\n"
        "_«Я — продакт-менеджер в B2B SaaS стартапе. "
        "Предпочитаю краткие ответы с bullet points. "
        "Нужны практические советы без воды. "
        "Аудитория — технические основатели.»_\n\n"
        "📝 Отправь текст профиля:"
    )
    session["awaiting_profile"] = True
    tg_send(chat_id, text, parse_mode="Markdown")


def handle_clear(chat_id: int, user_id: int) -> None:
    get_session(user_id)["history"] = []
    tg_send(chat_id, "🗑 История диалога очищена. Начинаем с чистого листа!")


def handle_status(chat_id: int, user_id: int) -> None:
    session = get_session(user_id)
    profile = get_profile(user_id)
    profile_status = "✅ Заполнен" if profile else "❌ Не заполнен — используй /profile"
    pairs = len(session["history"]) // 2
    text = (
        "📊 *Статус сессии*\n\n"
        f"🤖 Агент: *{AGENTS[session['agent']]['name']}*\n"
        f"💬 История: *{pairs}* пар вопрос-ответ\n"
        f"👤 Профиль: *{profile_status}*"
    )
    tg_send(chat_id, text, parse_mode="Markdown")


def handle_text_message(chat_id: int, user_id: int, text: str) -> None:
    session = get_session(user_id)

    if session["awaiting_profile"]:
        session["awaiting_profile"] = False
        save_profile(user_id, text)
        tg_send(
            chat_id,
            "✅ *Профиль сохранён!*\n\n"
            "Все агенты теперь будут учитывать информацию о тебе.\n"
            f"Активный агент: *{AGENTS[session['agent']]['name']}*\n\n"
            "Задай вопрос!",
            parse_mode="Markdown",
        )
        return

    logger.info("Сообщение от user_id=%s агент=%s: %s", user_id, session["agent"], text[:60])
    tg_typing(chat_id)

    agent_key = session["agent"]
    response = ask_groq(agent_key, user_id, session["history"], text)

    session["history"].append({"role": "user", "content": text})
    session["history"].append({"role": "assistant", "content": response})

    if len(session["history"]) > MAX_HISTORY_MESSAGES:
        session["history"] = session["history"][-MAX_HISTORY_MESSAGES:]

    send_long_message(chat_id, f"{AGENTS[agent_key]['name']}:\n\n{response}")


# ---------------------------------------------------------------------------
# Обработчик callback-кнопок
# ---------------------------------------------------------------------------


def handle_callback(callback_query: dict) -> None:
    query_id = callback_query["id"]
    user_id = callback_query["from"]["id"]
    chat_id = callback_query["message"]["chat"]["id"]
    message_id = callback_query["message"]["message_id"]
    data = callback_query.get("data", "")

    try:
        tg_answer_callback(query_id)
    except Exception:
        pass

    if not data.startswith("switch_agent:"):
        return

    agent_key = data.split(":", 1)[1]
    if agent_key not in AGENTS:
        tg_edit(chat_id, message_id, "⚠️ Неизвестный агент.")
        return

    session = get_session(user_id)
    session["agent"] = agent_key
    session["history"] = []

    agent = AGENTS[agent_key]
    profile_hint = "" if get_profile(user_id) else "\n\n💡 Заполни /profile для персонализации ответов."

    tg_edit(
        chat_id,
        message_id,
        f"✅ Переключился на *{agent['name']}*\n"
        f"_{agent['description']}_\n\n"
        f"История сброшена. Задавай вопросы!{profile_hint}",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Диспетчер входящих обновлений
# ---------------------------------------------------------------------------


def process_update(update: dict) -> None:
    try:
        if "callback_query" in update:
            handle_callback(update["callback_query"])
            return

        message = update.get("message")
        if not message:
            return

        text: str = message.get("text", "")
        if not text:
            return

        chat_id: int = message["chat"]["id"]
        user_id: int = message["from"]["id"]
        first_name: str = message["from"].get("first_name", "")

        if text.startswith("/start"):
            handle_start(chat_id, user_id, first_name)
        elif text.startswith("/menu"):
            handle_menu(chat_id, user_id)
        elif text.startswith("/profile"):
            handle_profile(chat_id, user_id)
        elif text.startswith("/clear"):
            handle_clear(chat_id, user_id)
        elif text.startswith("/status"):
            handle_status(chat_id, user_id)
        elif not text.startswith("/"):
            handle_text_message(chat_id, user_id, text)

    except Exception as exc:
        logger.error("Ошибка обработки update %s: %s", update.get("update_id"), exc)


# ---------------------------------------------------------------------------
# Главный цикл long polling
# ---------------------------------------------------------------------------


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN не задан в .env файле")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY не задан в .env файле")

    offset: int | None = None
    attempt = 0

    while True:
        attempt += 1
        logger.info("Запуск polling (попытка #%d)...", attempt)
        try:
            while True:
                updates = tg_get_updates(offset=offset, timeout=30)
                for update in updates:
                    offset = update["update_id"] + 1
                    process_update(update)
        except KeyboardInterrupt:
            logger.info("Бот остановлен пользователем (Ctrl+C).")
            break
        except Exception as exc:
            logger.error(
                "Ошибка polling (попытка #%d): %s. Перезапуск через %d сек...",
                attempt, exc, RESTART_DELAY,
            )
            time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    main()
