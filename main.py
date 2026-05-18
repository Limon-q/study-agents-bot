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

import asyncio
import os
import logging
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

load_dotenv()

TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Модель Groq, которую используют все агенты
GROQ_MODEL = "llama-3.3-70b-versatile"

# Максимальное количество сообщений в истории диалога (1 пара = 2 сообщения)
MAX_HISTORY_MESSAGES = 20

# Папка для хранения профилей пользователей (по одному файлу на user_id)
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
# Каждый агент имеет:
#   name          — отображаемое имя с эмодзи
#   description   — краткое описание для меню
#   system_prompt — системный промпт; {profile_section} заменяется профилем юзера

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
# sessions[user_id] = {
#   "agent": str,               — ключ текущего агента
#   "history": list[dict],      — история диалога (роль + контент)
#   "awaiting_profile": bool,   — ожидаем ли следующее сообщение как профиль
# }

sessions: dict[int, dict] = {}


def get_session(user_id: int) -> dict:
    """Вернуть сессию пользователя, создав новую если её нет."""
    if user_id not in sessions:
        sessions[user_id] = {
            "agent": "assistant",
            "history": [],
            "awaiting_profile": False,
        }
    return sessions[user_id]


# ---------------------------------------------------------------------------
# Работа с профилями пользователей
# ---------------------------------------------------------------------------


def get_profile(user_id: int) -> str | None:
    """Прочитать профиль из файла. Вернуть None если файла нет."""
    profile_path = PROFILES_DIR / f"{user_id}.txt"
    if profile_path.exists():
        return profile_path.read_text(encoding="utf-8").strip()
    return None


def save_profile(user_id: int, profile_text: str) -> None:
    """Сохранить профиль пользователя в файл profiles/<user_id>.txt."""
    profile_path = PROFILES_DIR / f"{user_id}.txt"
    profile_path.write_text(profile_text.strip(), encoding="utf-8")
    logger.info("Профиль сохранён: %s", profile_path)


# ---------------------------------------------------------------------------
# Сборка промптов и клавиатуры
# ---------------------------------------------------------------------------


def build_system_prompt(agent_key: str, user_id: int) -> str:
    """
    Собрать финальный системный промпт агента.

    Подставляет профиль пользователя в {profile_section}.
    Если профиля нет — вставляет подсказку использовать /profile.
    """
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


def build_agent_keyboard() -> InlineKeyboardMarkup:
    """Создать inline-клавиатуру для переключения между агентами."""
    buttons = [
        [
            InlineKeyboardButton(
                f"{data['name']} — {data['description']}",
                callback_data=f"switch_agent:{key}",
            )
        ]
        for key, data in AGENTS.items()
    ]
    return InlineKeyboardMarkup(buttons)


# ---------------------------------------------------------------------------
# Запрос к Groq API
# ---------------------------------------------------------------------------


async def ask_groq(
    agent_key: str,
    user_id: int,
    history: list[dict],
    user_message: str,
) -> str:
    """
    Отправить запрос в Groq и вернуть текст ответа.

    Передаёт системный промпт + историю диалога + новое сообщение.
    """
    system_prompt = build_system_prompt(agent_key, user_id)

    # Формируем список сообщений: system → история → новый запрос
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    def _call() -> str:
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

    try:
        return await asyncio.to_thread(_call)
    except Exception as exc:
        logger.error("Ошибка Groq API: %s", exc)
        return f"⚠️ Произошла ошибка при обращении к AI: {exc}"


# ---------------------------------------------------------------------------
# Утилита: отправка длинных сообщений
# ---------------------------------------------------------------------------


async def send_long_message(update: Update, text: str) -> None:
    """Разбить текст на части по 4096 символов и отправить последовательно."""
    limit = 4096
    for i in range(0, len(text), limit):
        await update.message.reply_text(text[i : i + limit])


# ===========================================================================
# ОБРАБОТЧИКИ КОМАНД
# ===========================================================================


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start — приветствие и главное меню.

    Показывает список агентов и подсказку про /profile.
    """
    user = update.effective_user
    session = get_session(user.id)
    current = AGENTS[session["agent"]]["name"]

    text = (
        f"👋 Привет, {user.first_name}!\n\n"
        "Я — команда из трёх AI-агентов, каждый заточен под свою задачу:\n\n"
        "🗓 *Ассистент* — задачи, расписание, напоминания\n"
        "🔍 *Ресёрчер* — поиск и анализ информации\n"
        "✍️ *Копирайтер* — тексты, структура презентаций\n\n"
        "💡 Команда /profile позволяет рассказать о себе — "
        "тогда все агенты будут давать персонализированные ответы.\n\n"
        f"Сейчас активен: *{current}*\n"
        "Выбери агента или сразу задай вопрос:"
    )

    await update.message.reply_text(
        text,
        reply_markup=build_agent_keyboard(),
        parse_mode="Markdown",
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /menu — показать меню выбора агента.

    Полезно когда нужно быстро переключиться без /start.
    """
    user = update.effective_user
    session = get_session(user.id)
    current = AGENTS[session["agent"]]["name"]

    await update.message.reply_text(
        f"🤖 Активный агент: *{current}*\n\nВыбери агента:",
        reply_markup=build_agent_keyboard(),
        parse_mode="Markdown",
    )


async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /profile — создать или обновить профиль пользователя.

    Переводит сессию в режим ожидания: следующее сообщение
    будет сохранено как профиль, а не передано агенту.
    """
    user = update.effective_user
    session = get_session(user.id)
    existing = get_profile(user.id)

    # Показываем текущий профиль если он есть
    current_block = ""
    if existing:
        current_block = f"📋 *Твой текущий профиль:*\n{existing}\n\n---\n\n"

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

    # Устанавливаем флаг — следующее сообщение идёт в профиль
    session["awaiting_profile"] = True

    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /clear — очистить историю диалога текущей сессии.

    Профиль при этом сохраняется, удаляется только история переписки.
    """
    session = get_session(update.effective_user.id)
    session["history"] = []

    await update.message.reply_text(
        "🗑 История диалога очищена. Начинаем с чистого листа!"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /status — показать текущее состояние сессии.

    Отображает: активный агент, длину истории, наличие профиля.
    """
    user = update.effective_user
    session = get_session(user.id)
    profile = get_profile(user.id)

    profile_status = "✅ Заполнен" if profile else "❌ Не заполнен — используй /profile"
    history_count = len(session["history"])
    pairs = history_count // 2  # количество пар вопрос-ответ

    text = (
        "📊 *Статус сессии*\n\n"
        f"🤖 Агент: *{AGENTS[session['agent']]['name']}*\n"
        f"💬 История: *{pairs}* пар вопрос-ответ\n"
        f"👤 Профиль: *{profile_status}*"
    )

    await update.message.reply_text(text, parse_mode="Markdown")


# ===========================================================================
# ОБРАБОТЧИК INLINE-КНОПОК
# ===========================================================================


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик нажатий на inline-кнопки меню агентов.

    Переключает агента и сбрасывает историю диалога,
    чтобы новый агент начинал без контекста предыдущего.
    """
    query = update.callback_query
    await query.answer()  # убираем анимацию загрузки на кнопке

    user_id = query.from_user.id
    session = get_session(user_id)

    if not query.data.startswith("switch_agent:"):
        return

    agent_key = query.data.split(":", 1)[1]

    if agent_key not in AGENTS:
        await query.edit_message_text("⚠️ Неизвестный агент.")
        return

    # Переключаем агента и сбрасываем историю (новый агент — новый контекст)
    session["agent"] = agent_key
    session["history"] = []

    agent = AGENTS[agent_key]
    profile = get_profile(user_id)
    profile_hint = "" if profile else "\n\n💡 Заполни /profile для персонализации ответов."

    await query.edit_message_text(
        f"✅ Переключился на *{agent['name']}*\n"
        f"_{agent['description']}_\n\n"
        f"История сброшена. Задавай вопросы!{profile_hint}",
        parse_mode="Markdown",
    )


# ===========================================================================
# ОБРАБОТЧИК ТЕКСТОВЫХ СООБЩЕНИЙ
# ===========================================================================


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Главный обработчик входящих текстовых сообщений.

    Два режима:
    1. Режим профиля (awaiting_profile=True) — сохраняет текст как профиль.
    2. Обычный режим — передаёт сообщение текущему агенту через Groq API.
    """
    user = update.effective_user
    session = get_session(user.id)
    text = update.message.text

    # -------------------------------------------------------------------
    # Режим сохранения профиля
    # -------------------------------------------------------------------
    if session["awaiting_profile"]:
        session["awaiting_profile"] = False
        save_profile(user.id, text)

        await update.message.reply_text(
            "✅ *Профиль сохранён!*\n\n"
            "Все агенты теперь будут учитывать информацию о тебе.\n"
            f"Активный агент: *{AGENTS[session['agent']]['name']}*\n\n"
            "Задай вопрос!",
            parse_mode="Markdown",
        )
        return

    # -------------------------------------------------------------------
    # Обычный диалог с агентом
    # -------------------------------------------------------------------
    logger.info("Сообщение от user_id=%s агент=%s: %s", user.id, session["agent"], text[:60])

    # send_chat_action — только косметика, не критично если упадёт
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
        )
    except Exception:
        pass

    agent_key = session["agent"]
    response = await ask_groq(agent_key, user.id, session["history"], text)

    # Обновляем историю диалога (сначала запрос, потом ответ)
    session["history"].append({"role": "user", "content": text})
    session["history"].append({"role": "assistant", "content": response})

    # Обрезаем историю до MAX_HISTORY_MESSAGES чтобы не раздувать контекст
    if len(session["history"]) > MAX_HISTORY_MESSAGES:
        session["history"] = session["history"][-MAX_HISTORY_MESSAGES:]

    # Добавляем подпись агента к ответу
    agent_name = AGENTS[agent_key]["name"]
    full_response = f"{agent_name}:\n\n{response}"

    await send_long_message(update, full_response)


# ===========================================================================
# ГЛОБАЛЬНЫЙ ОБРАБОТЧИК ОШИБОК
# ===========================================================================


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Ловит все необработанные исключения из handlers и логирует их.

    Без этого исключения в handlers молча проглатываются фреймворком,
    и пользователь не получает никакого ответа.
    """
    logger.error("Необработанное исключение:", exc_info=context.error)

    # Сообщаем пользователю об ошибке если есть активное сообщение
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Что-то пошло не так. Попробуй ещё раз или используй /clear."
            )
        except Exception:
            pass


# ===========================================================================
# ТОЧКА ВХОДА
# ===========================================================================

RESTART_DELAY = 5  # секунд между перезапусками после краша


def build_app() -> Application:
    """Собрать и настроить Application. Вызывается заново при каждом перезапуске."""
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # --- Команды ---
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("profile", cmd_profile))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("status", cmd_status))

    # --- Кнопки ---
    app.add_handler(CallbackQueryHandler(handle_callback))

    # --- Обычные сообщения (не команды) ---
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # --- Глобальный обработчик ошибок ---
    app.add_error_handler(error_handler)

    return app


def main() -> None:
    """
    Главная точка входа с авто-перезапуском.

    Если polling падает с любым исключением (кроме Ctrl+C),
    бот ждёт RESTART_DELAY секунд и запускается заново.
    Application нельзя переиспользовать после остановки,
    поэтому при каждом перезапуске создаётся новый экземпляр.
    """
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN не задан в .env файле")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY не задан в .env файле")

    attempt = 0
    while True:
        attempt += 1
        logger.info("Запуск бота (попытка #%d)...", attempt)
        try:
            app = build_app()
            app.run_polling(
                allowed_updates=Update.ALL_TYPES,
                # drop_pending_updates=False — подбираем накопленные сообщения
            )
            # run_polling вернулся нормально — значит Ctrl+C или явная остановка
            logger.info("Бот остановлен штатно.")
            break
        except KeyboardInterrupt:
            logger.info("Бот остановлен пользователем (Ctrl+C).")
            break
        except Exception as exc:
            logger.error(
                "Критическая ошибка (попытка #%d): %s. Перезапуск через %d сек...",
                attempt, exc, RESTART_DELAY,
            )
            time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    main()
