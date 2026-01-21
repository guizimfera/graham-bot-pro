# notifier_worker.py
import os
import time
from datetime import datetime

from dotenv import load_dotenv
import yfinance as yf

import db
from telegram_notifier import send_telegram_message

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def ticker_yfinance(ticker: str) -> str:
    t = ticker.upper().strip().replace(".SA", "")
    return t if t.endswith(".SA") else f"{t}.SA"

def fmt_brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_current_price(ticker: str) -> float | None:
    try:
        stock = yf.Ticker(ticker_yfinance(ticker))
        info = stock.info or {}
        price = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
            or info.get("open")
        )
        return float(price) if price else None
    except Exception:
        return None

def check_once() -> None:
    alerts = db.list_alerts()
    if not alerts:
        return

    for a in alerts:
        note = a.get("note") if isinstance(a, dict) else None
        note_line = f"\n📌 {note}" if note else ""  
        ticker = a["ticker"]
        alvo = float(a["preco_alvo"])
        tipo = a["tipo"]  # 'menor' ou 'maior'

        price = get_current_price(ticker)
        if price is None:
            continue

        triggered = (tipo == "menor" and price <= alvo) or (tipo == "maior" and price >= alvo)
        if not triggered:
            continue

        op = "≤" if tipo == "menor" else "≥"
        msg = (
    f"🔔 *Alerta atingido!* \n"
    f"*{ticker}* está em *{fmt_brl(price)}* (alvo: {op} {fmt_brl(alvo)})"
    f"{note_line}\n"
    f"_Hora:_ {datetime.now().strftime('%d/%m %H:%M')}"
)

        ok = send_telegram_message(BOT_TOKEN, CHAT_ID, msg)

        # Anti-spam simples: remove o alerta depois que disparar
        if ok:
            db.remove_alert(a["id"])

def main():
    db.init_db()
    interval_seconds = 300  # 5 min
    while True:
        check_once()
        time.sleep(interval_seconds)

if __name__ == "__main__":
    main()
