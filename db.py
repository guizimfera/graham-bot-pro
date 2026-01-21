import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

DB_FILE = Path("data.db")

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS historico (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,                -- ISO: YYYY-MM-DD HH:MM
    ticker TEXT NOT NULL,
    preco REAL NOT NULL,
    valor_justo REAL NOT NULL,
    margem REAL NOT NULL,
    score REAL,
    veredito TEXT NOT NULL,
    fonte_dy TEXT,
    dy REAL,
    dy_yahoo REAL,
    dy_fundamentus REAL
);

CREATE INDEX IF NOT EXISTS idx_historico_ticker_ts ON historico(ticker, ts);

CREATE TABLE IF NOT EXISTS posicoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    qtd REAL NOT NULL,
    preco_medio REAL NOT NULL,
    atualizado_em TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_posicoes_ticker ON posicoes(ticker);

"""

def connect(db_file: Path = DB_FILE) -> sqlite3.Connection:
    con = sqlite3.connect(db_file.as_posix(), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db() -> None:
    con = connect()
    try:
        con.executescript(SCHEMA)
        con.commit()
    finally:
        con.close()

def count_rows(table: str) -> int:
    con = connect()
    try:
        cur = con.execute(f"SELECT COUNT(*) AS n FROM {table}")
        return int(cur.fetchone()["n"])
    finally:
        con.close()

def add_history(entry: Dict[str, Any], dedupe_minutes: int = 2) -> None:
    """
    Dedupe: if there's an entry for same ticker with same veredito within the last `dedupe_minutes`, skip.
    """
    con = connect()
    try:
        ts = entry["ts"]
        ticker = entry["ticker"]
        veredito = entry["veredito"]
        # Check recent
        cur = con.execute("PRAGMA table_info(alertas)")
        cols = {row[1] for row in cur.fetchall()}  # row[1] = name
        if "note" not in cols:
            con.execute("ALTER TABLE alertas ADD COLUMN note TEXT")
        
        row = cur.fetchone()
        if row is not None:
            # Compare times in minutes
            from datetime import datetime, timedelta
            try:
                last_dt = datetime.strptime(row["ts"], "%Y-%m-%d %H:%M")
                new_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
                if abs((new_dt - last_dt).total_seconds()) <= dedupe_minutes * 60:
                    return
            except Exception:
                pass

        con.execute(
            """
            INSERT INTO historico (ts, ticker, preco, valor_justo, margem, score, veredito, fonte_dy, dy, dy_yahoo, dy_fundamentus)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["ts"],
                entry["ticker"],
                float(entry["preco"]),
                float(entry["valor_justo"]),
                float(entry["margem"]),
                float(entry.get("score")) if entry.get("score") is not None else None,
                entry["veredito"],
                entry.get("fonte_dy"),
                float(entry.get("dy")) if entry.get("dy") is not None else None,
                float(entry.get("dy_yahoo")) if entry.get("dy_yahoo") is not None else None,
                float(entry.get("dy_fundamentus")) if entry.get("dy_fundamentus") is not None else None,
            ),
        )
        # keep last 200 rows (SaaS-ready but bounded)
        con.execute(
            """
            DELETE FROM historico
            WHERE id NOT IN (SELECT id FROM historico ORDER BY id DESC LIMIT 200)
            """
        )
        con.commit()
    finally:
        con.close()

def list_history(limit: int = 200) -> List[Dict[str, Any]]:
    con = connect()
    try:
        cur = con.execute(
            "SELECT * FROM historico ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

def clear_history() -> None:
    con = connect()
    try:
        con.execute("DELETE FROM historico")
        con.commit()
    finally:
        con.close()

def add_alert(ticker: str, tipo: str, preco_alvo: float, criado_em: str, note: Optional[str] = None) -> None:
    con = connect()
    try:
        # compatível com DB antigo (sem note) por garantia
        cur = con.execute("PRAGMA table_info(alertas)")
        cols = {row[1] for row in cur.fetchall()}
        if "note" in cols:
            con.execute(
                """
                INSERT INTO alertas (criado_em, ticker, tipo, preco_alvo, note)
                VALUES (?, ?, ?, ?, ?)
                """,
                (criado_em, ticker, tipo, float(preco_alvo), note),
            )
        else:
            con.execute(
                """
                INSERT INTO alertas (criado_em, ticker, tipo, preco_alvo)
                VALUES (?, ?, ?, ?)
                """,
                (criado_em, ticker, tipo, float(preco_alvo)),
            )
        con.commit()
    finally:
        con.close()


def remove_alert(alert_id: int) -> None:
    con = connect()
    try:
        con.execute("DELETE FROM alertas WHERE id = ?", (int(alert_id),))
        con.commit()
    finally:
        con.close()

def list_alerts() -> List[Dict[str, Any]]:
    con = connect()
    try:
        cur = con.execute("SELECT * FROM alertas ORDER BY id DESC")
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()
        
def get_position(ticker: str) -> Optional[Dict[str, Any]]:
    con = connect()
    try:
        cur = con.execute("SELECT * FROM posicoes WHERE ticker = ?", (ticker,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        con.close()

def upsert_position(ticker: str, qtd: float, preco: float, atualizado_em: str) -> Dict[str, Any]:
    """
    Registra compra e recalcula PREÇO MÉDIO:
    novo_pm = (pm_antigo*qtd_antiga + preco*qtd_nova) / (qtd_total)
    """
    con = connect()
    try:
        t = ticker.upper().strip().replace(".SA", "")
        qtd = float(qtd)
        preco = float(preco)

        cur = con.execute("SELECT qtd, preco_medio FROM posicoes WHERE ticker = ?", (t,))
        row = cur.fetchone()

        if row:
            qtd_old = float(row["qtd"])
            pm_old = float(row["preco_medio"])
            qtd_total = qtd_old + qtd
            pm_new = (pm_old * qtd_old + preco * qtd) / qtd_total if qtd_total > 0 else preco
            con.execute(
                "UPDATE posicoes SET qtd = ?, preco_medio = ?, atualizado_em = ? WHERE ticker = ?",
                (qtd_total, pm_new, atualizado_em, t),
            )
        else:
            con.execute(
                "INSERT INTO posicoes (ticker, qtd, preco_medio, atualizado_em) VALUES (?, ?, ?, ?)",
                (t, qtd, preco, atualizado_em),
            )

        con.commit()
        return get_position(t) or {"ticker": t, "qtd": qtd, "preco_medio": preco, "atualizado_em": atualizado_em}
    finally:
        con.close()
