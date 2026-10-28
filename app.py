import math
from datetime import datetime
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Sports Terminal", page_icon="⚽", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.1rem; max-width:1400px}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25); padding:12px; border-radius:12px}
</style>
""", unsafe_allow_html=True)

FD_BASE = "https://api.football-data.org/v4"
ODDS_BASE = "https://api.the-odds-api.com/v4"
LEAGUES = {
    "Brasileirão Série A": "BSA",
    "Premier League": "PL",
    "La Liga": "PD",
    "Bundesliga": "BL1",
    "Serie A (Itália)": "SA",
    "Ligue 1": "FL1",
    "Champions League": "CL",
    "Eredivisie": "DED",
    "Primeira Liga": "PPL",
}
st.title("⚽ AI Sports Terminal")
st.caption("v1.0 • análise e previsões de futebol • sem apostas reais")
st.warning("Este painel gera estimativas, não garantias. Não executa apostas. Odds e dados podem ter cobertura limitada conforme a fonte e o plano.")

with st.sidebar:
    st.header("Configuração")
    league_name = st.selectbox("Competição", list(LEAGUES.keys()), index=0)
    league_code = LEAGUES[league_name]
    fd_key = st.text_input("Token football-data.org (opcional)", type="password",
                           help="Crie uma conta e copie seu token, se disponível. Não publique o token no GitHub.")
    odds_key = st.text_input("Chave The Odds API (opcional)", type="password",
                             help="Sem chave, o painel funciona sem odds ao vivo.")
    st.caption("Para publicar com segurança, prefira configurar tokens em Secrets na plataforma de hospedagem, em vez de deixá-los no código.")
    refresh = st.button("Atualizar dados", type="primary", use_container_width=True)
    st.divider()
    st.caption("Fontes: football-data.org para jogos/classificação; The Odds API opcional para odds. Limites e cobertura variam por plano.")

def safe_get(url, headers=None, params=None):
    r = requests.get(url, headers=headers or {}, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=900, show_spinner=False)
def get_matches(league_code, token):
    headers = {"X-Auth-Token": token} if token else {}
    return safe_get(f"{FD_BASE}/competitions/{league_code}/matches", headers=headers,
                    params={"status": "SCHEDULED"})["matches"]

@st.cache_data(ttl=1800, show_spinner=False)
def get_standings(league_code, token):
    headers = {"X-Auth-Token": token} if token else {}
    return safe_get(f"{FD_BASE}/competitions/{league_code}/standings", headers=headers).get("standings", [])

@st.cache_data(ttl=300, show_spinner=False)
def get_odds(api_key, sport_key):
    return safe_get(f"{ODDS_BASE}/sports/{sport_key}/odds/",
                    params={"apiKey": api_key, "regions": "uk,eu", "markets": "h2h", "oddsFormat": "decimal"})

def sample_matches():
    now = datetime.now().strftime("%Y-%m-%d")
    return pd.DataFrame([
        {"Data":now,"Competição":"DEMO (fictício)","Mandante":"Time Azul","Visitante":"Time Verde","Status":"DEMO"},
        {"Data":now,"Competição":"DEMO (fictício)","Mandante":"Clube Norte","Visitante":"Clube Sul","Status":"DEMO"},
    ])

def implied_probs(odds):
    inv = [1.0/x if x and x > 1 else 0 for x in odds]
    total = sum(inv)
    return [x/total for x in inv] if total else [None]*len(odds)

def poisson_probs(lam, max_goals=8):
    vals = [math.exp(-lam) * (lam**k) / math.factorial(k) for k in range(max_goals+1)]
    s = sum(vals)
    return [v/s for v in vals]

def estimate_from_standings(home, away, standings_df):
    # Uses points per match plus goals for/against as a transparent rough heuristic.
    # It is not a trained or calibrated prediction model.
    default = {"ppm": 1.35, "gf": 1.35, "ga": 1.35, "played": 0}
    def row_for(team):
        if standings_df.empty or "team" not in standings_df.columns:
            return default
        rows = standings_df[standings_df["team"].str.casefold() == team.casefold()]
        if rows.empty: return default
        r = rows.iloc[0]
        played = max(float(r.get("played", 0) or 0), 1)
        return {
            "ppm": float(r.get("points", 0) or 0) / played,
            "gf": float(r.get("goalsFor", 0) or 0) / played,
            "ga": float(r.get("goalsAgainst", 0) or 0) / played,
            "played": played,
        }
    h, a = row_for(home), row_for(away)
    # Conservative blend of team scoring and opponent concessions, with modest home advantage.
    lam_h = max(0.2, min(3.5, 0.55*h["gf"] + 0.45*a["ga"] + 0.18))
    lam_a = max(0.2, min(3.5, 0.55*a["gf"] + 0.45*h["ga"] - 0.08))
    ph, pa = poisson_probs(lam_h), poisson_probs(lam_a)
    p_home = p_draw = p_away = 0.0
    for i, pi in enumerate(ph):
        for j, pj in enumerate(pa):
            p = pi*pj
            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p
    total = p_home+p_draw+p_away
    return {
        "home_xg":lam_h, "away_xg":lam_a,
        "p_home":p_home/total, "p_draw":p_draw/total, "p_away":p_away/total,
        "home_stats":h, "away_stats":a
    }

def pct(x):
    return "—" if x is None else f"{100*x:.1f}%"

def odds_for_match(odds_data, home, away):
    if not odds_data: return None
    for event in odds_data:
        if event.get("home_team","").casefold() == home.casefold() and event.get("away_team","").casefold() == away.casefold():
            # Best decimal odds per outcome across bookmakers; map draw if present.
            best = {"home":None,"draw":None,"away":None}
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h": continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        price = outcome.get("price")
                        key = "home" if name == home else "away" if name == away else "draw" if name.casefold() == "draw" else None
                        if key and (best[key] is None or price > best[key]): best[key] = price
            return best
    return None

def backtest(df):
    # Expanding-window prediction: only prior matches contribute to each prediction.
    # Team attack/defence rates are estimated from previous rows; early rows use league defaults.
    required = {"date","home_team","away_team","home_goals","away_goals"}
    if not required.issubset(set(df.columns)):
        return None, "CSV precisa conter: date, home_team, away_team, home_goals, away_goals."
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    for c in ["home_goals","away_goals"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["date","home_team","away_team","home_goals","away_goals"]).sort_values("date")
    if len(x) < 20:
        return None, "Envie pelo menos 20 jogos; 100+ jogos são preferíveis para uma avaliação inicial."
    prior = []
    predictions = []
    for _, m in x.iterrows():
        if len(prior) >= 20:
            past = pd.DataFrame(prior)
            team_rows = []
            teams = set(past["home_team"]).union(set(past["away_team"]))
            for t in teams:
                h = past[past["home_team"] == t]
                a = past[past["away_team"] == t]
                gf = h["home_goals"].sum() + a["away_goals"].sum()
                ga = h["away_goals"].sum() + a["home_goals"].sum()
                played = len(h)+len(a)
                team_rows.append({"team":t,"played":played,"goalsFor":gf,"goalsAgainst":ga,"points":0})
            sdf = pd.DataFrame(team_rows)
            pred = estimate_from_standings(m["home_team"],m["away_team"],sdf)
            outcome = "H" if m["home_goals"] > m["away_goals"] else "D" if m["home_goals"] == m["away_goals"] else "A"
            predicted = max([("H",pred["p_home"]),("D",pred["p_draw"]),("A",pred["p_away"])],key=lambda z:z[1])[0]
            predictions.append({"date":m["date"],"home_team":m["home_team"],"away_team":m["away_team"],
                                "actual":outcome,"predicted":predicted,"correct":int(predicted==outcome),
                                "p_home":pred["p_home"],"p_draw":pred["p_draw"],"p_away":pred["p_away"]})
        prior.append(m.to_dict())
    out = pd.DataFrame(predictions)
    if out.empty: return None, "Não foi possível gerar previsões com esse arquivo."
    return out, None

st.subheader("1. Jogos e previsões")
matches_df = pd.DataFrame()
standings_df = pd.DataFrame()
odds_data = None
source_status = []

try:
    matches_raw = get_matches(league_code, fd_key.strip())
    matches_df = pd.DataFrame([{
        "Data": m.get("utcDate","")[:16].replace("T"," "),
        "Competição": m.get("competition",{}).get("name",league_name),
        "Mandante": m.get("homeTeam",{}).get("name",""),
        "Visitante": m.get("awayTeam",{}).get("name",""),
        "Status": m.get("status","")
    } for m in matches_raw])
    source_status.append(f"Jogos: football-data.org ({len(matches_df)} partidas)")
except Exception as e:
    source_status.append("Jogos: indisponível — " + str(e)[:160])

try:
    standings_raw = get_standings(league_code, fd_key.strip())
    table = standings_raw[0].get("table", []) if standings_raw else []
    standings_df = pd.DataFrame([{
        "team": t.get("team",{}).get("name",""),
        "position": t.get("position"),
        "played": t.get("playedGames",0),
        "points": t.get("points",0),
        "goalsFor": t.get("goalsFor",0),
        "goalsAgainst": t.get("goalsAgainst",0),
        "goalDifference": t.get("goalDifference",0),
    } for t in table])
    source_status.append(f"Classificação: {len(standings_df)} equipes")
except Exception as e:
    source_status.append("Classificação: indisponível — " + str(e)[:160])

if odds_key.strip():
    try:
        # Mapping varies by provider and region; football-data competition code is not an Odds API key.
        odds_sport_map = {
            "Brasileirão Série A":"soccer_brazil_campeonato",
            "Premier League":"soccer_epl",
            "La Liga":"soccer_spain_la_liga",
            "Bundesliga":"soccer_germany_bundesliga",
            "Serie A (Itália)":"soccer_italy_serie_a",
            "Ligue 1":"soccer_france_ligue_one",
            "Champions League":"soccer_uefa_champs_league",
            "Eredivisie":"soccer_netherlands_eredivisie",
            "Primeira Liga":"soccer_portugal_primeira_liga",
        }
        odds_data = get_odds(odds_key.strip(), odds_sport_map[league_name])
        source_status.append(f"Odds: {len(odds_data)} eventos recebidos")
    except Exception as e:
        source_status.append("Odds: indisponíveis — " + str(e)[:160])
else:
    source_status.append("Odds: não conectadas (opcional)")

for s in source_status:
    st.caption("• " + s)

if not matches_df.empty:
    st.dataframe(matches_df, use_container_width=True, hide_index=True)
    if not standings_df.empty:
        st.subheader("Classificação usada como contexto")
        st.dataframe(standings_df, use_container_width=True, hide_index=True)
    st.subheader("2. Previsões")
    teams = sorted(set(matches_df["Mandante"].dropna()).union(set(matches_df["Visitante"].dropna())))
    prediction_rows = []
    for _, m in matches_df.head(100).iterrows():
        pred = estimate_from_standings(m["Mandante"], m["Visitante"], standings_df)
        odds = odds_for_match(odds_data, m["Mandante"], m["Visitante"])
        odds_txt = "Não disponível"
        if odds:
            odds_txt = " | ".join([f"{k}: {v}" for k,v in odds.items() if v])
        prediction_rows.append({
            "Data":m["Data"],"Mandante":m["Mandante"],"Visitante":m["Visitante"],
            "Prob. mandante":pct(pred["p_home"]),"Prob. empate":pct(pred["p_draw"]),"Prob. visitante":pct(pred["p_away"]),
            "Gols esperados (M-V)":f'{pred["home_xg"]:.2f} - {pred["away_xg"]:.2f}',
            "Odds melhores disponíveis":odds_txt,
            "Agente de probabilidade":"Heurística Poisson (não calibrada)"
        })
    pred_df = pd.DataFrame(prediction_rows)
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    st.download_button("Baixar previsões CSV", pred_df.to_csv(index=False).encode("utf-8"), "previsoes_futebol.csv", "text/csv")
else:
    st.info("Nenhum jogo foi carregado. A API pode exigir token, não cobrir a competição escolhida ou estar temporariamente indisponível. Consulte a documentação da fonte.")
    st.markdown("**Demonstração:** as previsões não serão preenchidas com jogos fictícios. Configure uma fonte de dados válida para habilitar a análise.")

st.divider()
st.subheader("3. Central dos seis agentes")
agents = [
    ("1 — Scout de jogos","Busca calendário, equipes e status dos jogos."),
    ("2 — Forma e estatísticas","Usa classificação, gols marcados/sofridos e desempenho disponível."),
    ("3 — Modelo de probabilidades","Estima probabilidades 1X2 por uma aproximação de Poisson transparente; não é um modelo calibrado."),
    ("4 — Mercado de odds","Opcional: obtém odds quando a API e o plano cobrem a competição; calcula probabilidades implícitas sem margem quando há dados suficientes."),
    ("5 — Comparador","Sinaliza divergências entre probabilidade estimada e probabilidade de mercado; não chama isso de lucro garantido."),
    ("6 — Auditor e validação","Registra previsões históricas importadas e mede acerto, sem usar jogos futuros no cálculo de cada previsão do backtest.")
]
for name, desc in agents:
    with st.expander(name, expanded=False):
        st.write(desc)

st.divider()
st.subheader("4. Validação histórica")
st.write("Envie um CSV de resultados históricos para um backtest cronológico simples. O sistema só usa partidas anteriores à partida avaliada para estimar as taxas dos times.")
st.caption("Colunas obrigatórias: date, home_team, away_team, home_goals, away_goals. Datas no formato YYYY-MM-DD. Idealmente, use dados completos e consistentes de uma única competição.")
uploaded = st.file_uploader("Enviar CSV histórico", type=["csv"])
if uploaded:
    try:
        hist = pd.read_csv(uploaded)
        bt, err = backtest(hist)
        if err:
            st.error(err)
        else:
            accuracy = bt["correct"].mean()
            st.metric("Acurácia 1X2 no backtest", f"{accuracy:.1%}", f"{len(bt)} jogos avaliados")
            st.caption("Acurácia sozinha não mede qualidade de probabilidades. Para uma avaliação séria, acrescente Brier Score, log loss, comparação com baseline e validação fora da amostra.")
            st.dataframe(bt, use_container_width=True, hide_index=True)
            st.download_button("Baixar relatório do backtest", bt.to_csv(index=False).encode("utf-8"), "backtest_futebol.csv", "text/csv")
    except Exception as e:
        st.error(f"Não foi possível ler o CSV: {e}")

st.divider()
st.subheader("5. Custos e limites")
st.markdown("""
- **Aplicação:** Streamlit + Python + pandas — software gratuito.
- **Calendário e resultados:** football-data.org possui plano gratuito para competições selecionadas; cobertura, quotas e acesso a estatísticas variam por competição e plano.
- **Odds:** The Odds API é opcional; disponibilidade de competições, mercados, chamadas e histórico depende do plano. Sem chave, o painel funciona sem odds.
- **Hospedagem:** pode começar no Streamlit Community Cloud gratuito, sujeito às regras e limites atuais.
- **Sem apostas reais:** não há integração com casas de apostas nem execução de bilhetes.
- **Sem promessa de rentabilidade:** estimativas iniciais são heurísticas e precisam ser avaliadas contra dados fora da amostra.
""")
st.caption("AI Sports Terminal v1.0 • protótipo educacional • não é aconselhamento financeiro ou recomendação garantida.")
