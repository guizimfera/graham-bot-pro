import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

import google.generativeai as genai

import db  # local sqlite layer

# ======================================================
#                    CONFIG / BOOT
# ======================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="Graham-Bot Pro", page_icon="🧠", layout="wide")

# ======================================================
# UI / ESTILO (SaaS-ready) — CSS + micro-animações
# ======================================================
def _inject_global_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        }

        .stApp {
            background: radial-gradient(1200px 800px at 10% 0%, rgba(99, 102, 241, 0.18), transparent 60%),
                        radial-gradient(1000px 700px at 90% 10%, rgba(16, 185, 129, 0.14), transparent 55%),
                        radial-gradient(900px 700px at 50% 100%, rgba(244, 63, 94, 0.10), transparent 55%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.96) 0%, rgba(17, 24, 39, 0.90) 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div {
            color: rgba(255,255,255,0.92);
        }

        header[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
        .block-container { padding-top: 3.4rem; }

        .gb-hero {
            margin-top: 6px;
            padding: 18px 18px 14px 18px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(16, 185, 129, 0.14));
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 16px 40px rgba(0,0,0,0.25);
            backdrop-filter: blur(10px);
        }
        .gb-hero h1 { margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.02em; }
        .gb-hero p  { margin: 6px 0 0 0; opacity: 0.86; font-size: 14px; }

        .gb-card {
            border-radius: 22px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 14px 34px rgba(0,0,0,0.20);
            backdrop-filter: blur(10px);
        }

        .gb-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(0,0,0,0.18);
            font-size: 12px;
            line-height: 1;
            opacity: .92;
        }
        .gb-chip b { font-weight: 800; }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 12px 14px;
            border-radius: 20px;
            box-shadow: 0 12px 26px rgba(0,0,0,0.18);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 38px rgba(0,0,0,0.24);
            border-color: rgba(255,255,255,0.14);
        }

        .stButton > button {
            border-radius: 14px !important;
            padding: 10px 14px !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.82), rgba(16, 185, 129, 0.72)) !important;
            color: white !important;
            font-weight: 800 !important;
            transition: transform .14s ease, filter .14s ease;
        }
        .stButton > button:hover { transform: translateY(-1px); filter: brightness(1.03); }

        input, textarea { border-radius: 14px !important; }
        div[data-baseweb="select"] > div { border-radius: 14px !important; }
        button[data-baseweb="tab"] { border-radius: 999px !important; padding: 10px 14px !important; }

        div[data-testid="stPlotlyChart"] > div {
            border-radius: 22px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.06);
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .gb-anim { animation: fadeUp .45s ease both; }

        .gb-footer { opacity: 0.72; font-size: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _hero(title: str, subtitle: str, right_html: str = ""):
    st.markdown(
        f"""
        <div class="gb-hero gb-anim">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;">
                <div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;">
                    {right_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

_inject_global_css()

# ======================================================
# App constants
# ======================================================
HISTORICO_JSON = "historico_analises.json"
ALERTAS_JSON = "alertas_precos.json"
CACHE_TTL_SECONDS = 60 * 30  # 30 min
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

db.init_db()

# ======================================================
# JSON -> SQLITE migration (best-effort)
# ======================================================
def _safe_load_json(path: str) -> List[Dict[str, Any]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def migrar_json_para_sqlite() -> None:
    try:
        if db.count_rows("historico") == 0:
            hist = _safe_load_json(HISTORICO_JSON)
            for item in hist:
                ts = item.get("data") or item.get("ts")
                if not ts:
                    continue
                entry = {
                    "ts": ts,
                    "ticker": str(item.get("ticker", "")).upper().replace(".SA", ""),
                    "preco": float(item.get("preco", 0) or 0),
                    "valor_justo": float(item.get("valor_justo", 0) or 0),
                    "margem": float(item.get("margem", 0) or 0),
                    "score": float(item.get("score")) if item.get("score") is not None else None,
                    "veredito": str(item.get("veredito", "AGUARDAR")).upper(),
                    "fonte_dy": item.get("fonte_dy"),
                    "dy": item.get("dy"),
                    "dy_yahoo": item.get("dy_yahoo"),
                    "dy_fundamentus": item.get("dy_fundamentus"),
                }
                db.add_history(entry, dedupe_minutes=0)

        if db.count_rows("alertas") == 0:
            alerts = _safe_load_json(ALERTAS_JSON)
            for a in alerts:
                ticker = str(a.get("ticker", "")).upper().replace(".SA", "")
                tipo = a.get("tipo", "menor")
                preco_alvo = float(a.get("preco_alvo", 0) or 0)
                criado = a.get("criado_em") or datetime.now().strftime("%Y-%m-%d %H:%M")
                if ticker and preco_alvo > 0:
                    db.add_alert(ticker, tipo, preco_alvo, criado)
    except Exception:
        pass

migrar_json_para_sqlite()

# ======================================================
# Utils
# ======================================================
def normalizar_ticker(ticker: str) -> str:
    return ticker.upper().strip().replace(".SA", "")

def ticker_yfinance(ticker: str) -> str:
    t = normalizar_ticker(ticker)
    return t if t.endswith(".SA") else f"{t}.SA"

def fmt_brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def calcular_graham(info: Dict[str, Any]) -> float:
    lpa = float(info.get("trailingEps", 0) or 0)
    vpa = float(info.get("bookValue", 0) or 0)
    if lpa > 0 and vpa > 0:
        return float(np.sqrt(22.5 * lpa * vpa))
    return 0.0

def calcular_margem(preco_justo: float, preco_atual: float) -> float:
    if preco_justo > 0:
        return float(((preco_justo - preco_atual) / preco_justo) * 100)
    return 0.0

def calcular_p_vp(preco: float, vpa: float) -> float:
    if vpa and vpa > 0:
        return float(preco / vpa)
    return 0.0

# ======================================================
# Data collection (cache)
# ======================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def pegar_dividend_yield_fundamentus(ticker: str) -> Optional[float]:
    try:
        t = normalizar_ticker(ticker)
        url = f"https://www.fundamentus.com.br/detalhes.php?papel={t}"
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.content, "html.parser")
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2 and "Div. Yield" in cells[0].get_text(strip=True):
                txt = cells[1].get_text(strip=True)
                txt = txt.replace("%", "").replace(".", "").replace(",", ".").strip()
                if txt and txt != "-":
                    return float(txt) / 100.0
        return None
    except Exception:
        return None

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def pegar_dados_yfinance(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Puxa dados do Yahoo e tenta enriquecer com DY do Fundamentus.
    Também tenta trazer proxies úteis para 'risk': debtToEquity e beta (quando existir).
    """
    try:
        t_yf = ticker_yfinance(ticker)
        stock = yf.Ticker(t_yf)
        info = stock.info or {}

        current_price = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
            or info.get("open")
        )
        if not current_price:
            return None

        t_clean = normalizar_ticker(ticker)
        dy_fund = pegar_dividend_yield_fundamentus(t_clean)

        dy_yahoo = float(
            info.get("trailingAnnualDividendYield", 0.0)
            or info.get("dividendYield", 0.0)
            or 0.0
        )
        dy_final = dy_fund if dy_fund is not None else dy_yahoo
        fonte_dy = "Fundamentus" if dy_fund is not None else "Yahoo"

        dados = {
            "ticker": t_clean,
            "longName": info.get("longName", t_clean),
            "currentPrice": float(current_price),
            "trailingEps": float(info.get("trailingEps", 0.0) or 0.0),
            "bookValue": float(info.get("bookValue", 0.0) or 0.0),

            "dividendYield": float(dy_final or 0.0),
            "dividendYieldYahoo": float(dy_yahoo or 0.0),
            "dividendYieldFundamentus": dy_fund,
            "fonteDY": fonte_dy,

            "trailingPE": float(info.get("trailingPE", 0.0) or 0.0),
            "profitMargins": float(info.get("profitMargins", 0.0) or 0.0),

            # Risk-ish fields (best-effort)
            "debtToEquity": float(info.get("debtToEquity", 0.0) or 0.0),  # often "percent" in Yahoo (e.g. 120 = 1.2x)
            "beta": float(info.get("beta", 0.0) or 0.0),

            "sector": info.get("sector", "Desconhecido"),
            "industry": info.get("industry", "Desconhecida"),
            "marketCap": float(info.get("marketCap", 0) or 0),
            "sharesOutstanding": float(info.get("sharesOutstanding", 0) or 0),
        }

        # Completar bookValue via priceToBook
        if dados["bookValue"] == 0:
            p_vp = info.get("priceToBook")
            if p_vp and p_vp > 0:
                dados["bookValue"] = dados["currentPrice"] / p_vp

        # Completar P/L via EPS
        if dados["trailingPE"] == 0 and dados["trailingEps"] > 0:
            dados["trailingPE"] = dados["currentPrice"] / dados["trailingEps"]

        return dados
    except Exception:
        return None

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def pegar_historico_preco(ticker: str, period: str = "2y") -> pd.DataFrame:
    stock = yf.Ticker(ticker_yfinance(ticker))
    hist = stock.history(period=period)
    if hist is None or hist.empty:
        return pd.DataFrame()
    return hist

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def calcular_volatilidade_1y(ticker: str) -> Optional[float]:
    """
    Retorna volatilidade anualizada aproximada (std diário * sqrt(252)).
    Saída em decimal: 0.35 = 35% a.a.
    """
    try:
        hist = pegar_historico_preco(ticker, period="1y")
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        rets = hist["Close"].pct_change().dropna()
        if rets.empty:
            return None
        vol = float(rets.std() * np.sqrt(252))
        return max(0.0, vol)
    except Exception:
        return None

def debt_ratio_from_info(info: Dict[str, Any]) -> float:
    """
    Usa debtToEquity como proxy (Yahoo costuma fornecer em 'percent' ex: 120 = 1.2x).
    Retorna decimal em 'x do PL': 1.2 = dívida 1.2x PL.
    """
    dte = float(info.get("debtToEquity", 0.0) or 0.0)
    if dte <= 0:
        return 0.0
    # heurística: se vier grande (>= 5), provavelmente é percent (50, 120, 300...)
    if dte >= 5:
        return dte / 100.0
    return dte

# ======================================================
#  PERFIS (NÚCLEO ÚNICO) — SEM IFS ESPALHADOS
# ======================================================
PROFILES: Dict[str, Dict[str, Any]] = {
    "Defensivo": {
        "objective": "Preservar capital com mínimo risco",
        "chips": ["Valuation", "Qualidade", "Risco"],
        "weights": {"valuation": 0.45, "quality": 0.25, "income": 0.10, "risk": 0.20},
        "buy_rules": {"min_margin": 35, "min_score": 80},
        "guards": {"max_pe": 15, "max_pvp": 1.5, "eps_positive": True, "bv_positive": True},
        "penalties": {"high_volatility": 0.85, "high_debt": 0.85},
        "alerts": [
            {"type": "margin_ge", "value": 35},
            {"type": "score_ge", "value": 80},
            {"type": "price_le", "multiplier": 0.65},
        ],
        "position_plan": [0.5, 0.5],
    },
    "Equilibrado": {
        "objective": "Crescer com disciplina e risco controlado",
        "chips": ["Valuation", "Qualidade", "Renda", "Risco"],
        "weights": {"valuation": 0.35, "quality": 0.25, "income": 0.15, "risk": 0.25},
        "buy_rules": {"min_margin": 25, "min_score": 74},
        "guards": {"max_pe": 20, "max_pvp": 2.2, "eps_positive": True, "bv_positive": True},
        "penalties": {"high_volatility": 0.88, "high_debt": 0.88},
        "alerts": [
            {"type": "margin_ge", "value": 25},
            {"type": "score_ge", "value": 74},
            {"type": "price_le", "multiplier": 0.72},
        ],
        "position_plan": [0.4, 0.3, 0.3],
    },
    "Oportunista": {
        "objective": "Buscar assimetrias e upside com mais tolerância",
        "chips": ["Valuation", "Qualidade", "Risco"],
        "weights": {"valuation": 0.40, "quality": 0.20, "income": 0.05, "risk": 0.35},
        "buy_rules": {"min_margin": 15, "min_score": 66},
        "guards": {"max_pe": 35, "max_pvp": 4.0, "eps_positive": False, "bv_positive": False},
        "penalties": {"high_volatility": 0.92, "high_debt": 0.92},
        "alerts": [
            {"type": "margin_ge", "value": 15},
            {"type": "score_ge", "value": 66},
            {"type": "price_le", "multiplier": 0.80},
        ],
        "position_plan": [0.35, 0.35, 0.30],
    },
}

# ======================================================
# SCORE POR PILAR (explicável, vendável)
# ======================================================
def score_valuation(info: Dict[str, Any], preco_justo: float) -> Tuple[float, Dict[str, float]]:
    """
    Retorna score 0..40 e também métricas para explicar.
    """
    preco = float(info.get("currentPrice", 0) or 0)
    margem = calcular_margem(preco_justo, preco)
    pl = float(info.get("trailingPE", 0) or 0)
    pvp = calcular_p_vp(preco, float(info.get("bookValue", 0) or 0))

    s_margem = min(40.0, max(0.0, float(margem)))
    s_pl = max(0.0, 20.0 - (pl - 10.0) * 2.0) if pl > 0 else 0.0
    s_pvp = max(0.0, 20.0 - (pvp - 1.0) * 10.0) if pvp > 0 else 0.0

    raw = min(40.0, s_margem) + s_pl + s_pvp
    score = float(clamp(raw, 0.0, 40.0))

    details = {
        "margem_%": float(margem),
        "pl": float(pl),
        "pvp": float(pvp),
        "s_margem": float(min(40.0, s_margem)),
        "s_pl": float(clamp(s_pl, 0.0, 20.0)),
        "s_pvp": float(clamp(s_pvp, 0.0, 20.0)),
    }
    return score, details

def score_quality(info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    0..20 (proxy simples via profitMargins).
    """
    pm = float(info.get("profitMargins", 0) or 0)
    score = float(min(20.0, max(0.0, pm * 100.0)))
    return score, {"profitMargins_%": float(pm * 100.0)}

def score_income(info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    0..10 com penalidade se DY suspeito (diferença entre fontes).
    """
    dy = float(info.get("dividendYield", 0) or 0)
    dy_y = float(info.get("dividendYieldYahoo", 0) or 0)
    dy_f = info.get("dividendYieldFundamentus")

    penalty = 1.0
    if dy_f is not None:
        dy_f = float(dy_f or 0)
        if abs(dy_f - dy_y) > 0.02:
            penalty = 0.6

    score = float(min(10.0, dy * 100.0) * penalty)
    return score, {
        "dy_%": float(dy * 100.0),
        "dy_yahoo_%": float(dy_y * 100.0),
        "dy_fund_%": float(dy_f * 100.0) if dy_f is not None else float("nan"),
        "penalty": float(penalty),
    }

def score_risk(volatility: float, debt_ratio: float) -> Tuple[float, Dict[str, float]]:
    """
    0..30 (15 vol + 15 dívida) — quanto menor o risco, maior score.
    volatility: 0.35 = 35% a.a.
    debt_ratio: 1.2 = dívida 1.2x PL (aprox)
    """
    vol = float(volatility or 0.0)
    debt = float(debt_ratio or 0.0)

    s_vol = max(0.0, 15.0 - vol * 100.0)      # 15 - 35 = 0 em 35% a.a.
    s_debt = max(0.0, 15.0 - debt * 10.0)     # 15 - 12 = 3 em 1.2x

    score = float(clamp(s_vol + s_debt, 0.0, 30.0))
    return score, {"vol_%aa": float(vol * 100.0), "debt_x_pl": float(debt)}

def apply_penalties(pillars: Dict[str, float], profile: Dict[str, Any], volatility: float, debt_ratio: float) -> Tuple[Dict[str, float], List[str]]:
    """
    Penalidades suaves (multiplicadores) para casos extremos.
    Retorna pillars ajustados + lista de razões.
    """
    penalties = profile.get("penalties", {}) or {}
    reasons: List[str] = []
    out = dict(pillars)

    # thresholds heurísticos (pode evoluir depois)
    if volatility is not None and volatility >= 0.45:  # 45% a.a.
        mult = float(penalties.get("high_volatility", 1.0))
        out["risk"] = out.get("risk", 0.0) * mult
        reasons.append(f"Volatilidade alta (≈ {volatility*100:.0f}% a.a.) → x{mult:.2f}")

    if debt_ratio is not None and debt_ratio >= 2.0:  # dívida >= 2x PL
        mult = float(penalties.get("high_debt", 1.0))
        out["risk"] = out.get("risk", 0.0) * mult
        reasons.append(f"Dívida alta (≈ {debt_ratio:.1f}x PL) → x{mult:.2f}")

    return out, reasons

def composite_score(pillars: Dict[str, float], profile: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Score final 0..100 (normaliza internamente).
    A ideia: pilares têm escalas diferentes (valuation 0..40, quality 0..20, income 0..10, risk 0..30).
    Primeiro convertemos para 0..100 por pilar, depois aplicamos weights.
    """
    max_by_pillar = {"valuation": 40.0, "quality": 20.0, "income": 10.0, "risk": 30.0}

    total = 0.0
    breakdown: Dict[str, float] = {}

    for p, w in (profile.get("weights") or {}).items():
        raw = float(pillars.get(p, 0.0))
        maxv = float(max_by_pillar.get(p, 100.0))
        normalized = (raw / maxv) * 100.0 if maxv > 0 else 0.0
        weighted = normalized * float(w)
        breakdown[p] = round(weighted, 1)
        total += weighted

    return round(total, 1), breakdown

def decide(profile: Dict[str, Any], info: Dict[str, Any], score: float, margem: float) -> Tuple[str, str]:
    g = profile.get("guards", {}) or {}

    if g.get("eps_positive") and float(info.get("trailingEps", 0) or 0) <= 0:
        return "EVITAR", "Lucro (LPA) negativo"

    if g.get("bv_positive") and float(info.get("bookValue", 0) or 0) <= 0:
        return "EVITAR", "Patrimônio (VPA) negativo"

    max_pe = g.get("max_pe")
    if max_pe is not None:
        pe = float(info.get("trailingPE", 0) or 0)
        if pe > 0 and pe > float(max_pe):
            return "EVITAR", f"P/L acima do guardrail ({pe:.1f} > {float(max_pe):.1f})"

    max_pvp = g.get("max_pvp")
    if max_pvp is not None:
        pvp = calcular_p_vp(float(info.get("currentPrice", 0) or 0), float(info.get("bookValue", 0) or 0))
        if pvp > 0 and pvp > float(max_pvp):
            return "EVITAR", f"P/VP acima do guardrail ({pvp:.2f} > {float(max_pvp):.2f})"

    br = profile.get("buy_rules", {}) or {}
    min_margin = float(br.get("min_margin", 0))
    min_score = float(br.get("min_score", 0))

    if margem >= min_margin and score >= min_score:
        return "COMPRA", "Atende critérios do perfil"

    if score >= min_score * 0.85:
        return "AGUARDAR", "Boa empresa; preço ainda não ideal"

    return "EVITAR", "Risco ou preço inadequado"

def suggested_action(veredito: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    if veredito == "COMPRA":
        return {"label": "Montar posição", "steps": profile.get("position_plan", [1.0])}
    if veredito == "AGUARDAR":
        return {"label": "Esperar gatilho", "suggest": "Criar alertas automáticos"}
    return {"label": "Evitar ativo"}

def build_alerts(profile: Dict[str, Any], info: Dict[str, Any], preco_justo: float, score: float, margem: float) -> List[Dict[str, Any]]:
    """
    Retorna alertas recomendados (mistos). Para persistência no DB, só criamos os de preço.
    """
    out: List[Dict[str, Any]] = []
    for a in (profile.get("alerts") or []):
        if a.get("type") == "price_le":
            target = float(preco_justo) * float(a.get("multiplier", 0.7))
            out.append({"kind": "price", "op": "<=", "target": target, "label": f"Preço ≤ {fmt_brl(target)}"})
        elif a.get("type") == "margin_ge":
            v = float(a.get("value", 0))
            out.append({"kind": "logic", "label": f"Margem ≥ {v:.0f}% (atual: {margem:.1f}%)"})
        elif a.get("type") == "score_ge":
            v = float(a.get("value", 0))
            out.append({"kind": "logic", "label": f"Score ≥ {v:.0f} (atual: {score:.1f})"})
    return out

# ======================================================
# IA (opcional)
# ======================================================
def analise_ia_gemini(ticker: str, info: Dict[str, Any], preco_justo: float, profile_name: str, score: float, veredito: str) -> str:
    if not GOOGLE_API_KEY:
        return "⚠️ IA desabilitada: configure `GOOGLE_API_KEY` no arquivo `.env`."
    try:
        generation_config = {
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        modelo = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

        margem_seguranca = calcular_margem(preco_justo, info["currentPrice"])
        pvp = calcular_p_vp(info["currentPrice"], info["bookValue"])

        prompt = f"""
Aja como **Benjamin Graham** e como um analista de risco.
Empresa: **{ticker}** (Setor: {info['sector']})

Perfil aplicado: **{profile_name}**
Score do sistema: **{score:.1f}/100**
Veredito do sistema: **{veredito}**

### DADOS:
- Preço Atual: R$ {info['currentPrice']:.2f}
- Valor Justo (Graham): R$ {preco_justo:.2f}
- Margem de Segurança: {margem_seguranca:.1f}%
- P/L: {info['trailingPE']:.2f}
- P/VP: {pvp:.2f}
- Dividend Yield (12m): {info['dividendYield']*100:.2f}% (Fonte: {info.get('fonteDY','?')})

### ENTREGUE (Markdown):
1) Uma **tabela curta**: Indicador | Atual | Referência | Status ✅/⚠️/❌
2) 3 pontos de **qualidade** e 3 pontos de **risco**
3) Uma explicação do porquê o veredito do sistema faz sentido (ou não)
4) Uma seção final: "**O que eu faria agora**" com 3 bullets (bem objetivo)
"""
        resp = modelo.generate_content(prompt)
        return resp.text or ""
    except Exception as e:
        return f"Erro na IA: {e}"

# ======================================================
# Backtest (mantido do seu projeto, com leve ajuste no texto)
# ======================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _tentar_fundamentos_anuais_yf(ticker: str) -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker_yfinance(ticker))
        income_df = getattr(stock, "financials", None)
        bal = getattr(stock, "balance_sheet", None)

        if income_df is None or bal is None:
            return None
        if not isinstance(income_df, pd.DataFrame) or not isinstance(bal, pd.DataFrame):
            return None
        if income_df.empty or bal.empty:
            return None

        def pick_row(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
            for n in names:
                for idx in df.index:
                    if str(idx).strip().lower() == n.lower():
                        return df.loc[idx]
            return None

        net_income = pick_row(income_df, ["Net Income", "NetIncome"])
        equity = pick_row(bal, ["Total Stockholder Equity", "Total Stockholders Equity", "Stockholders Equity"])

        if net_income is None or equity is None:
            return None

        shares = float((stock.info or {}).get("sharesOutstanding", 0) or 0)
        if shares <= 0:
            return None

        out = []
        for col in net_income.index:
            try:
                year = pd.to_datetime(col).year
            except Exception:
                continue
            ni = float(net_income[col] or 0)
            eq = float(equity[col] or 0)
            eps = ni / shares if shares > 0 else np.nan
            vpa = eq / shares if shares > 0 else np.nan
            out.append({"ano": int(year), "eps": eps, "vpa": vpa})

        df = pd.DataFrame(out).dropna()
        if df.empty:
            return None
        return df.sort_values("ano")
    except Exception:
        return None

def calcular_backtest(ticker: str, anos: int = 5) -> Optional[pd.DataFrame]:
    try:
        hist = pegar_historico_preco(ticker, period=f"{anos}y")
        if hist.empty:
            return None

        fund_anuais = _tentar_fundamentos_anuais_yf(ticker)
        info_atual = pegar_dados_yfinance(ticker)
        if not info_atual:
            return None

        resultados = []
        for dt in hist.index:
            preco = float(hist.loc[dt, "Close"])
            ano = int(pd.to_datetime(dt).year)

            if fund_anuais is not None:
                row = fund_anuais[fund_anuais["ano"] <= ano].tail(1)
                if row.empty:
                    eps = info_atual["trailingEps"]
                    vpa = info_atual["bookValue"]
                    modo = "fallback_atual"
                else:
                    eps = float(row.iloc[0]["eps"])
                    vpa = float(row.iloc[0]["vpa"])
                    modo = "anual_yahoo"
            else:
                eps = info_atual["trailingEps"]
                vpa = info_atual["bookValue"]
                modo = "simplificado"

            valor_justo = float(np.sqrt(22.5 * eps * vpa)) if eps and vpa and eps > 0 and vpa > 0 else 0.0
            margem = calcular_margem(valor_justo, preco)
            sinal = "COMPRA" if (valor_justo > 0 and margem > 20) else "AGUARDAR"

            resultados.append(
                {
                    "data": pd.to_datetime(dt),
                    "preco": preco,
                    "valor_justo": valor_justo,
                    "margem": margem,
                    "sinal": sinal,
                    "modo": modo,
                }
            )

        return pd.DataFrame(resultados)
    except Exception:
        return None

# ======================================================
# Alert checker (db)
# ======================================================
def verificar_alertas() -> List[str]:
    alertas = db.list_alerts()
    atingidos: List[str] = []
    for a in alertas:
        info = pegar_dados_yfinance(a["ticker"])
        if not info:
            continue
        preco_atual = float(info["currentPrice"])
        alvo = float(a["preco_alvo"])
        if a["tipo"] == "menor" and preco_atual <= alvo:
            atingidos.append(f"🔔 **{a['ticker']}** atingiu {fmt_brl(preco_atual)} (alvo: {fmt_brl(alvo)})")
        if a["tipo"] == "maior" and preco_atual >= alvo:
            atingidos.append(f"🔔 **{a['ticker']}** atingiu {fmt_brl(preco_atual)} (alvo: {fmt_brl(alvo)})")
    return atingidos

# ======================================================
# UI helpers (profile card + breakdown)
# ======================================================
def render_profile_card(profile_name: str, profile: Dict[str, Any]) -> None:
    chips_html = "".join([f'<span class="gb-chip">🧩 <b>{c}</b></span>' for c in profile.get("chips", [])])
    st.markdown(
        f"""
        <div class="gb-card gb-anim">
            <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:flex-start;">
                <div>
                    <div style="font-size:12px;opacity:.85;">PERFIL</div>
                    <div style="font-size:18px;font-weight:900;letter-spacing:-0.01em;">🛡️ {profile_name.upper()}</div>
                    <div style="margin-top:6px;opacity:.88;">{profile.get("objective","")}</div>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;">
                    {chips_html}
                </div>
            </div>
            <div style="margin-top:10px;opacity:.75;font-size:12px;">
                Compra apenas com <b>margem</b> ≥ {profile.get("buy_rules",{}).get("min_margin","?")}%
                e <b>score</b> ≥ {profile.get("buy_rules",{}).get("min_score","?")}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_breakdown(score_total: float, breakdown: Dict[str, float]) -> None:
    st.markdown("### 📌 Score explicável")
    st.metric("Score", f"{score_total:.1f} / 100")

    order = ["valuation", "quality", "income", "risk"]
    labels = {
        "valuation": "Valuation",
        "quality": "Qualidade",
        "income": "Renda",
        "risk": "Risco",
    }

    for k in order:
        if k in breakdown:
            v = float(breakdown[k])
            # barra simples
            bar = int(clamp(v, 0, 100) / 4)  # 25 blocos
            st.write(f"**{labels[k]}:** " + "█" * bar + f"  {v:.1f}")

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("⚙️ Menu")

    profile_name = st.selectbox("Perfil", list(PROFILES.keys()), index=1)
    profile = PROFILES[profile_name]

    st.caption(
        f"Compra: margem ≥ {profile['buy_rules']['min_margin']}% | "
        f"score ≥ {profile['buy_rules']['min_score']} | "
        f"guards: P/L ≤ {profile['guards'].get('max_pe','—')} | P/VP ≤ {profile['guards'].get('max_pvp','—')}"
    )

    usar_ia = st.checkbox("Usar IA (Gemini) na análise individual", value=bool(GOOGLE_API_KEY))
    if usar_ia and not GOOGLE_API_KEY:
        st.warning("Configure `GOOGLE_API_KEY` no `.env` para habilitar a IA.")

    modo = st.radio(
        "Escolha o modo:",
        [
            "📊 Análise Individual",
            "🔄 Comparador de Ações",
            "🏆 Ranking (Perfil-aware)",
            "📈 Simulação Histórica",
            "🔔 Alertas de Preço",
            "📚 Histórico de Análises",
        ],
    )

    st.divider()

    if "alertas_checked" not in st.session_state:
        st.session_state["alertas_checked"] = True
        st.session_state["alertas_atingidos"] = verificar_alertas()

    with st.expander("🔔 Alertas atingidos nesta sessão", expanded=bool(st.session_state.get("alertas_atingidos"))):
        atingidos = st.session_state.get("alertas_atingidos", [])
        if atingidos:
            for msg in atingidos:
                st.success(msg)
        else:
            st.info("Nenhum alerta atingido no momento.")

    if st.button("🔁 Rechecar alertas agora"):
        st.session_state["alertas_atingidos"] = verificar_alertas()
        st.rerun()

# ======================================================
# HERO
# ======================================================
chips = []
chips.append(f'<span class="gb-chip">📌 Perfil: <b>{profile_name}</b></span>')
chips.append(f'<span class="gb-chip">🧩 Modo: <b>{modo}</b></span>')
chips.append('<span class="gb-chip">📡 Dados: <b>Yahoo + Fundamentus</b></span>')
chips.append(f'<span class="gb-chip">🤖 IA: <b>{"ON" if usar_ia else "OFF"}</b></span>')
_hero("🧠 Graham-Bot Pro", "Perfis premium + score explicável por pilares (arquitetura SaaS).", " ".join(chips))

st.markdown('<div style="height: 10px"></div>', unsafe_allow_html=True)

# ======================================================
# MODO: ANÁLISE INDIVIDUAL
# ======================================================
if modo == "📊 Análise Individual":
    colA, colB = st.columns([3, 1])
    ticker_input = colA.text_input("Ticker (Ex: BBAS3, ITSA4, VALE3):", "BBAS3")
    ticker_clean = normalizar_ticker(ticker_input)  
    btn_analisar = colB.button("🔍 Analisar", use_container_width=True)

    render_profile_card(profile_name, profile)
    st.markdown('<div style="height: 10px"></div>', unsafe_allow_html=True)

    if btn_analisar or ticker_clean:
        with st.spinner(f"Buscando dados de {ticker_clean}..."):
            info = pegar_dados_yfinance(ticker_clean)

        if not info:
            st.error("Não foi possível obter dados. Verifique o ticker.")
        else:
            preco_justo = calcular_graham(info)
            preco_atual = float(info["currentPrice"])
            margem = calcular_margem(preco_justo, preco_atual)

            vol = calcular_volatilidade_1y(ticker_clean) or 0.0
            debt = debt_ratio_from_info(info)

            # Pillars
            val_s, val_d = score_valuation(info, preco_justo)
            qua_s, qua_d = score_quality(info)
            inc_s, inc_d = score_income(info)
            ris_s, ris_d = score_risk(vol, debt)

            pillars = {"valuation": val_s, "quality": qua_s, "income": inc_s, "risk": ris_s}

            # Apply penalties (only touches risk pillar)
            pillars_adj, penalty_reasons = apply_penalties(pillars, profile, vol, debt)

            score_total, breakdown = composite_score(pillars_adj, profile)
            veredito, motivo = decide(profile, info, score_total, margem)
            cta = suggested_action(veredito, profile)
            rec_alerts = build_alerts(profile, info, preco_justo, score_total, margem)

            # Persist history
            db.add_history(
                {
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "ticker": ticker_clean,
                    "preco": preco_atual,
                    "valor_justo": preco_justo,
                    "margem": margem,
                    "score": score_total,
                    "veredito": veredito,
                    "fonte_dy": info.get("fonteDY"),
                    "dy": info.get("dividendYield"),
                    "dy_yahoo": info.get("dividendYieldYahoo"),
                    "dy_fundamentus": info.get("dividendYieldFundamentus"),
                }
            )

            # Tabs
            aba1, aba2, aba3 = st.tabs(["📊 Indicadores", "🧠 IA Graham", "📈 Gráfico"])

            with aba1:
                st.subheader(f"{info['longName']} ({ticker_clean})")

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Preço Atual", fmt_brl(preco_atual))
                c2.metric("Valor Justo (Graham)", fmt_brl(preco_justo) if preco_justo else "—")
                c3.metric("Margem de Segurança", f"{margem:.1f}%", delta_color="normal" if margem > 0 else "off")
                c4.metric("Score (perfil-aware)", f"{score_total:.1f}/100")
                c5.metric("Veredito", veredito)

                st.caption(f"🧠 Motivo: **{motivo}**")

                if penalty_reasons:
                    st.warning("Penalidades aplicadas: " + " • ".join(penalty_reasons))

                st.divider()

                # Explainer
                render_breakdown(score_total, breakdown)

                st.markdown("### 🧾 Registrar compra (preço médio — estilo Graham)")

                pos = db.get_position(ticker_clean)
                if pos:
                    st.info(
                        f"📌 Posição atual em **{ticker_clean}**: "
                        f"qtd **{pos['qtd']:.4g}** | preço médio **{fmt_brl(pos['preco_medio'])}** | "
                        f"atualizado em {pos['atualizado_em']}"
                    )

                cA, cB, cC = st.columns([1,1,1])
                qtd_buy = cA.number_input("Quantidade", min_value=0.0001, value=1.0, step=1.0)
                preco_buy = cB.number_input(
                    "Preço da compra (R$)",
                    min_value=0.01,
                    value=float(preco_atual),
                    step=0.10
                )
                criar_alertas_etapas = cC.checkbox("Criar alertas automáticos de Etapas", value=True)

                if st.button("✅ Salvar compra"):
                    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M")
                    pos2 = db.upsert_position(
                        ticker_clean,
                        float(qtd_buy),
                        float(preco_buy),
                        ts_now
                    )

                    st.success(
                        f"Compra registrada. Novo preço médio de **{ticker_clean}**: "
                        f"**{fmt_brl(float(pos2['preco_medio']))}**"
                    )

                    if criar_alertas_etapas:
                        pm = float(pos2["preco_medio"])
                        drops = [0.10] if len(profile.get("position_plan", [])) == 2 else [0.07, 0.15]

                        existing = db.list_alerts()
                        existing_keys = {(a["ticker"], a["tipo"], float(a["preco_alvo"])) for a in existing}

                        created = 0
                        for idx, d in enumerate(drops, start=2):
                            alvo = pm * (1 - d)
                            key = (ticker_clean, "menor", float(alvo))
                            if key in existing_keys:
                                continue

                            db.add_alert(
                                ticker=ticker_clean,
                                tipo="menor",
                                preco_alvo=float(alvo),
                                criado_em=ts_now,
                                note=f"Etapa {idx} (≈ -{d*100:.0f}% do preço médio)"
                            )
                            created += 1

                        st.success(f"{created} alerta(s) de etapas criado(s) com base no seu preço médio.")

                    st.rerun()
  

                st.markdown("### 🔎 Fundamentais rápidos")
                pvp = calcular_p_vp(preco_atual, float(info.get("bookValue", 0) or 0))
                dy_display = float(info.get("dividendYield", 0) or 0) * 100
                st.info(
                    f"**DY (12m):** {dy_display:.2f}% — **Fonte:** {info.get('fonteDY','?')}  \n"
                    f"**P/L:** {float(info.get('trailingPE',0) or 0):.2f} | **P/VP:** {pvp:.2f} | "
                    f"**Volatilidade (1y):** {vol*100:.0f}% a.a. | **Dívida/PL (proxy):** {debt:.2f}x  \n"
                    f"**Setor:** {info['sector']} / {info['industry']}"
                )

                with st.expander("🧱 Detalhes dos pilares (debug explicável)"):
                    st.json(
                        {
                            "valuation": {"score_0_40": val_s, "details": val_d},
                            "quality": {"score_0_20": qua_s, "details": qua_d},
                            "income": {"score_0_10": inc_s, "details": inc_d},
                            "risk": {"score_0_30": ris_s, "details": ris_d},
                            "pillars_adj": pillars_adj,
                            "weights": profile.get("weights"),
                            "breakdown_weighted": breakdown,
                        }
                    )

                # CTA
                st.markdown("### ✅ Ação sugerida")
                if veredito == "COMPRA":
                    steps = cta.get("steps", [1.0])
                    st.success(f"✔️ **{veredito}** — {cta.get('label','')}")
                    st.write("Plano de posição:")
                    for i, s in enumerate(steps, 1):
                        st.write(f"• Etapa {i}: **{s*100:.0f}%**")
                elif veredito == "AGUARDAR":
                    st.info(f"⏳ **{veredito}** — {cta.get('label','')}")
                    st.write(f"Sugestão: {cta.get('suggest','')}")
                else:
                    st.error(f"⛔ **{veredito}** — {cta.get('label','')}")

                st.divider()

                st.markdown("### 🔔 Alertas recomendados (1 clique)")
                if rec_alerts:
                    for a in rec_alerts:
                        st.write("• " + a["label"])

                    # Criar automaticamente só os alertas de preço (persistentes)
                    price_alerts = [a for a in rec_alerts if a.get("kind") == "price" and float(a.get("target", 0)) > 0]
                    colx, coly = st.columns([1, 2])
                    with colx:
                        if st.button("🔔 Criar alertas de preço recomendados", use_container_width=True):
                            for a in price_alerts:
                                db.add_alert(
                                    ticker=ticker_clean,
                                    tipo="menor",
                                    preco_alvo=float(a["target"]),
                                    criado_em=datetime.now().strftime("%Y-%m-%d %H:%M"),
                                )
                            st.success(f"{len(price_alerts)} alerta(s) de preço criado(s) para {ticker_clean}.")
                            st.rerun()
                    with coly:
                        st.caption("Obs.: alertas de Score/Margem aparecem como recomendação (lógica). Persistência automática hoje é só por preço.")
                else:
                    st.info("Este perfil não definiu alertas recomendados.")

                if preco_justo == 0:
                    st.warning("⚠️ Graham exige **LPA e VPA positivos**. Sem isso, o valor justo pode ficar zerado.")

            with aba2:
                if usar_ia:
                    with st.spinner("Gerando análise com IA (Gemini)..."):
                        st.write(analise_ia_gemini(ticker_clean, info, preco_justo, profile_name, score_total, veredito))
                else:
                    st.info("IA desativada. Ative na barra lateral se quiser uma análise textual.")

            with aba3:
                hist = pegar_historico_preco(ticker_clean, "2y")
                if hist.empty:
                    st.warning("Gráfico indisponível para este ativo.")
                else:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Candlestick(
                            x=hist.index,
                            open=hist["Open"],
                            high=hist["High"],
                            low=hist["Low"],
                            close=hist["Close"],
                            name="Cotação",
                        )
                    )
                    if preco_justo > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=hist.index,
                                y=[preco_justo] * len(hist),
                                mode="lines",
                                name="Valor Justo (Graham)",
                                line=dict(width=2, dash="dash"),
                                hoverinfo="y+name",
                            )
                        )
                    fig.update_layout(
                        height=520,
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis_rangeslider_visible=False,
                        yaxis=dict(side="right", tickprefix="R$ "),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# MODO: COMPARADOR
# ======================================================
elif modo == "🔄 Comparador de Ações":
    st.subheader("🔄 Comparador de Ações (perfil-aware)")
    st.caption("Compare até 5 ações com score por pilares + decisão baseada no perfil atual.")

    tickers_input = st.text_input("Tickers separados por vírgula:", "BBAS3, ITSA4, PETR4")

    if st.button("⚖️ Comparar"):
        tickers = [normalizar_ticker(t) for t in tickers_input.split(",") if t.strip()]
        if len(tickers) > 5:
            st.warning("Máximo de 5 ações por comparação.")
            tickers = tickers[:5]

        rows = []
        for t in tickers:
            with st.spinner(f"Analisando {t}..."):
                info = pegar_dados_yfinance(t)
            if not info:
                continue

            preco_justo = calcular_graham(info)
            preco_atual = float(info["currentPrice"])
            margem = calcular_margem(preco_justo, preco_atual)

            vol = calcular_volatilidade_1y(t) or 0.0
            debt = debt_ratio_from_info(info)

            val_s, _ = score_valuation(info, preco_justo)
            qua_s, _ = score_quality(info)
            inc_s, _ = score_income(info)
            ris_s, _ = score_risk(vol, debt)

            pillars = {"valuation": val_s, "quality": qua_s, "income": inc_s, "risk": ris_s}
            pillars_adj, _ = apply_penalties(pillars, profile, vol, debt)

            score_total, _ = composite_score(pillars_adj, profile)
            veredito, _motivo = decide(profile, info, score_total, margem)

            pvp = calcular_p_vp(preco_atual, float(info.get("bookValue", 0) or 0))

            rows.append(
                {
                    "Ticker": t,
                    "Empresa": str(info.get("longName", t))[:38],
                    "Preço": preco_atual,
                    "Valor Justo": preco_justo,
                    "Margem (%)": margem,
                    "Score": score_total,
                    "Veredito": veredito,
                    "P/L": float(info.get("trailingPE", 0) or 0),
                    "P/VP": pvp,
                    "DY (%)": float(info.get("dividendYield", 0) or 0) * 100,
                    "Vol (1y, %a.a.)": vol * 100,
                    "Dívida/PL (x)": debt,
                    "Setor": info.get("sector", "—"),
                }
            )

        if not rows:
            st.warning("Nenhum dado disponível para os tickers informados.")
        else:
            df = pd.DataFrame(rows)

            order = {"COMPRA": 0, "AGUARDAR": 1, "EVITAR": 2}
            df["ord"] = df["Veredito"].map(order).fillna(9)
            df = df.sort_values(["ord", "Score", "Margem (%)"], ascending=[True, False, False]).drop(columns=["ord"])

            st.dataframe(df, use_container_width=True)

            st.subheader("📊 Comparação Visual (Score)")
            fig = go.Figure()
            for _, row in df.iterrows():
                fig.add_trace(go.Bar(name=row["Ticker"], x=["Score"], y=[row["Score"]]))
            fig.update_layout(height=420, yaxis_title="Score", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# ======================================================
# MODO: RANKING
# ======================================================
elif modo == "🏆 Ranking (Perfil-aware)":
    st.subheader("🏆 Ranking (perfil-aware)")
    st.caption("A lista é analisada com o perfil atual (weights + guards + buy_rules).")

    blue_chips = [
        "BBAS3", "ITSA4", "BBDC4", "PETR4", "VALE3", "ITUB4",
        "ABEV3", "B3SA3", "WEGE3", "RENT3", "EGIE3", "TAEE11",
        "CPLE6", "CMIG4", "ENBR3", "BBSE3", "SANB11", "VIVT3",
        "RADL3", "HAPV3", "FLRY3", "SUZB3", "KLBN11", "CSAN3"
    ]

    st.markdown("Você pode alterar a lista base (separada por vírgula).")
    lista_input = st.text_area("Lista de tickers", ", ".join(blue_chips), height=90)
    tickers = [normalizar_ticker(t) for t in lista_input.split(",") if t.strip()]

    col1, col2, col3 = st.columns(3)
    with col1:
        min_margem = st.slider("Margem mínima (%)", -50, 100, 10)
    with col2:
        min_score = st.slider("Score mínimo", 0, 100, int(profile["buy_rules"]["min_score"] * 0.75))
    with col3:
        max_pe = st.slider("P/L máximo (filtro)", 0, 80, int(profile["guards"].get("max_pe", 30) or 30))

    if st.button("🚀 Gerar Ranking"):
        rows = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, t in enumerate(tickers):
            status_text.text(f"Analisando {t}... ({idx+1}/{len(tickers)})")
            progress_bar.progress((idx + 1) / max(1, len(tickers)))

            info = pegar_dados_yfinance(t)
            if not info:
                continue

            preco_justo = calcular_graham(info)
            preco_atual = float(info["currentPrice"])
            margem = calcular_margem(preco_justo, preco_atual)

            vol = calcular_volatilidade_1y(t) or 0.0
            debt = debt_ratio_from_info(info)

            val_s, _ = score_valuation(info, preco_justo)
            qua_s, _ = score_quality(info)
            inc_s, _ = score_income(info)
            ris_s, _ = score_risk(vol, debt)

            pillars = {"valuation": val_s, "quality": qua_s, "income": inc_s, "risk": ris_s}
            pillars_adj, _ = apply_penalties(pillars, profile, vol, debt)

            score_total, _ = composite_score(pillars_adj, profile)
            veredito, _motivo = decide(profile, info, score_total, margem)

            pvp = calcular_p_vp(preco_atual, float(info.get("bookValue", 0) or 0))

            rows.append(
                {
                    "Ticker": t,
                    "Empresa": str(info.get("longName", t))[:38],
                    "Preço": preco_atual,
                    "Valor Justo": preco_justo,
                    "Margem (%)": margem,
                    "Score": score_total,
                    "Veredito": veredito,
                    "P/L": float(info.get("trailingPE", 0) or 0),
                    "P/VP": pvp,
                    "DY (%)": float(info.get("dividendYield", 0) or 0) * 100,
                    "Vol (1y, %a.a.)": vol * 100,
                    "Dívida/PL (x)": debt,
                    "Setor": info.get("sector", "—"),
                }
            )

        progress_bar.empty()
        status_text.empty()

        if not rows:
            st.warning("Nenhuma ação retornou dados no momento.")
        else:
            df = pd.DataFrame(rows)

            df = df[
                (df["Margem (%)"] >= min_margem)
                & (df["Score"] >= min_score)
                & ((df["P/L"] <= max_pe) | (df["P/L"] == 0))
            ].copy()

            st.success(f"✅ Encontradas {len(df)} ações dentro dos filtros (perfil: {profile_name}).")

            order = {"COMPRA": 0, "AGUARDAR": 1, "EVITAR": 2}
            df["ord"] = df["Veredito"].map(order).fillna(9)
            df = df.sort_values(["ord", "Score", "Margem (%)"], ascending=[True, False, False]).drop(columns=["ord"])

            def color_score(v):
                if v >= 80: return "background-color: rgba(46, 204, 113, 0.25)"
                if v >= 60: return "background-color: rgba(241, 196, 15, 0.25)"
                return "background-color: rgba(231, 76, 60, 0.20)"

            st.dataframe(df.style.applymap(color_score, subset=["Score"]), use_container_width=True, height=650)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Baixar Ranking (CSV)",
                data=csv,
                file_name=f"ranking_{profile_name.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

# ======================================================
# MODO: SIMULAÇÃO HISTÓRICA
# ======================================================
elif modo == "📈 Simulação Histórica":
    st.subheader("📈 Simulação Histórica (Estratégia Graham)")
    st.caption("Mantido do seu projeto. (Resultados educacionais; dados anuais podem faltar no Yahoo.)")

    ticker_bt = normalizar_ticker(st.text_input("Ticker:", "BBAS3"))
    anos_bt = st.slider("Período (anos)", 1, 10, 5)

    if st.button("🔍 Executar Simulação"):
        with st.spinner(f"Calculando simulação de {ticker_bt} ({anos_bt} anos)..."):
            df_bt = calcular_backtest(ticker_bt, anos_bt)

        if df_bt is None or df_bt.empty:
            st.error("Não foi possível calcular. Verifique o ticker.")
        else:
            modo_used = df_bt["modo"].value_counts().idxmax()
            if modo_used == "simplificado":
                st.warning("⚠️ Modo simplificado: Yahoo não forneceu fundamentos anuais suficientes. Resultado educacional.")
            else:
                st.success(f"✅ Fundamentos anuais detectados (modo: {modo_used}).")

            compras = df_bt[df_bt["sinal"] == "COMPRA"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Sinais de Compra", len(compras))

            if len(compras) > 0:
                preco_medio = float(compras["preco"].mean())
                preco_atual = float(df_bt.iloc[-1]["preco"])
                retorno = ((preco_atual - preco_medio) / preco_medio) * 100
                col2.metric("Preço Médio Compra", fmt_brl(preco_medio))
                col3.metric("Retorno Simulado", f"{retorno:.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_bt["data"], y=df_bt["preco"], mode="lines", name="Preço"))
            fig.add_trace(go.Scatter(x=df_bt["data"], y=df_bt["valor_justo"], mode="lines", name="Valor Justo", line=dict(dash="dash")))
            if len(compras) > 0:
                fig.add_trace(go.Scatter(x=compras["data"], y=compras["preco"], mode="markers", name="Compras", marker=dict(size=10, symbol="triangle-up")))
            fig.update_layout(height=520, hovermode="x unified", yaxis_title="Preço (R$)")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_bt.tail(200), use_container_width=True)

# ======================================================
# MODO: ALERTAS
# ======================================================
elif modo == "🔔 Alertas de Preço":
    st.subheader("🔔 Gerenciador de Alertas (Persistente)")
    st.caption("Persistência via SQLite (robusto para SaaS).")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ➕ Criar Alerta")
        t = normalizar_ticker(st.text_input("Ticker", "BBAS3"))
        tipo_label = st.selectbox("Tipo", ["Menor ou igual a", "Maior ou igual a"])
        preco_alvo = st.number_input("Preço alvo (R$)", min_value=0.01, value=10.00, step=0.50)

        if st.button("✅ Criar alerta"):
            db.add_alert(
                ticker=t,
                tipo="menor" if "Menor" in tipo_label else "maior",
                preco_alvo=float(preco_alvo),
                criado_em=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
            st.success(f"Alerta criado para {t}.")
            st.rerun()

    with col2:
        st.markdown("### 📋 Alertas Ativos")
        alertas = db.list_alerts()
        if not alertas:
            st.info("Nenhum alerta configurado.")
        else:
            for a in alertas:
                tipo_txt = "≤" if a["tipo"] == "menor" else "≥"
                st.info(f"**{a['ticker']}** {tipo_txt} {fmt_brl(a['preco_alvo'])}  \nCriado em: {a['criado_em']}")
                if st.button("🗑️ Remover", key=f"rm_{a['id']}"):
                    db.remove_alert(a["id"])
                    st.rerun()

# ======================================================
# MODO: HISTÓRICO
# ======================================================
elif modo == "📚 Histórico de Análises":
    st.subheader("📚 Histórico de Análises (SQLite)")
    hist = db.list_history(200)

    if not hist:
        st.info("Nenhuma análise registrada ainda.")
    else:
        df = pd.DataFrame(hist)

        col1, col2, col3 = st.columns(3)
        with col1:
            tickers = sorted(df["ticker"].unique().tolist())
            filtro_ticker = st.multiselect("Ticker", options=tickers)
        with col2:
            veredictos = sorted(df["veredito"].unique().tolist())
            filtro_veredito = st.multiselect("Veredito", options=veredictos)
        with col3:
            min_score = st.slider("Score mínimo", 0, 100, 0)

        df_f = df.copy()
        if filtro_ticker:
            df_f = df_f[df_f["ticker"].isin(filtro_ticker)]
        if filtro_veredito:
            df_f = df_f[df_f["veredito"].isin(filtro_veredito)]
        df_f = df_f[df_f["score"].fillna(0) >= min_score]

        st.dataframe(df_f, use_container_width=True, height=650)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total (DB)", len(df))
        c2.metric("COMPRA", int((df["veredito"] == "COMPRA").sum()))
        c3.metric("Ações únicas", df["ticker"].nunique())

        colA, colB = st.columns(2)
        with colA:
            csv = df_f.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Baixar (CSV)", data=csv, file_name=f"historico_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        with colB:
            if st.button("🗑️ Limpar histórico"):
                db.clear_history()
                st.success("Histórico removido.")
                st.rerun()

st.markdown("---")
st.markdown('<div class="gb-footer">💡 Streamlit • Perfis: PROFILES (núcleo SaaS) • Dados: Yahoo + Fundamentus • IA: Gemini (opcional) • Persistência: SQLite</div>', unsafe_allow_html=True)
