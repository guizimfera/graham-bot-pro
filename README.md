# 🧠 Graham-Bot Pro

Sistema de análise fundamentalista inspirado em **Benjamin Graham**, com:

- 📊 Score explicável por pilares
- 🛡️ Perfis de investimento (Defensivo, Equilibrado, Oportunista)
- 🔔 Alertas persistentes de preço
- 📈 Backtest histórico
- 🤖 Análise opcional com IA (Google Gemini)
- 💾 Persistência em SQLite
- 🌐 Interface SaaS-ready com Streamlit

---

## 🚀 Funcionalidades

- Cálculo de **Valor Justo (Fórmula de Graham)**
- Margem de segurança automática
- Score composto por:
  - Valuation
  - Qualidade
  - Renda
  - Risco
- Veredito claro: **COMPRA / AGUARDAR / EVITAR**
- Ranking e comparador de ações
- Alertas de preço persistentes
- Histórico de análises

---

## 🛠️ Tecnologias

- Python 3.10+
- Streamlit
- Yahoo Finance (yfinance)
- Fundamentus (web scraping)
- SQLite
- Plotly
- Google Gemini (opcional)

---

## 📦 Instalação

```bash
git clone https://github.com/SEU_USUARIO/graham-bot-pro.git
cd graham-bot-pro
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r Requirements.txt
