# telegram_notifier.py
import requests

def send_telegram_message(bot_token: str, chat_id: str, text: str, timeout: int = 10) -> bool:
    """
    Envia uma mensagem simples via Telegram.
    Retorna True/False.
    """
    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False
