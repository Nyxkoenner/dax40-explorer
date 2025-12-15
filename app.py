import streamlit as st
import pandas as pd
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt
from typing import List, Dict, Optional
import feedparser
from dateutil import parser as dateparser
import altair as alt
from datetime import datetime, timedelta


# ---------- Daten laden ----------

def load_companies(csv_path: str = "dax40_companies.csv") -> pd.DataFrame:
    """
    Lädt die DAX40-Unternehmen aus der CSV.
    Erkennt automatisch Komma/Semikolon als Trennzeichen.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    st.sidebar.write("CSV-Spalten:", list(df.columns))
    if "sector" not in df.columns:
        st.error(
            "In der CSV wurde keine Spalte namens **'sector'** gefunden.\n\n"
            f"Aktuelle Spalten sind: {list(df.columns)}"
        )
        st.stop()
    return df


# ---------- Helper für Prozentwerte ----------

def to_percent(val) -> Optional[float]:
    """
    Konvertiert einen Wert konsistent in Prozent.
    - Wenn |val| <= 1 -> Interpret als Anteil (0.12 => 12 %)
    - Wenn |val| >  1 -> Interpret als bereits Prozent
    """
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if abs(v) <= 1:
        return v * 100.0
    return v


# ---------- Kennzahlen & Kursdaten laden ----------

def infer_dividend_frequency(div_series: pd.Series) -> Optional[str]:
    """
    Heuristik zur Dividendenfrequenz: Anzahl Ausschüttungen im letzten Jahr.
    """
    if div_series is None or div_series.empty:
        return None
    last_date = div_series.index.max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    recent = div_series[div_series.index >= one_year_ago]
    n = len(recent)
    if n == 0:
        return None
    if n >= 4:
        return "quarterly"
    if 2 <= n <= 3:
        return "semiannual"
    if n == 1:
        return "annual"
    return "irregular"


def calc_dividend_growth_5y(div_series: pd.Series) -> Optional[float]:
    """
    Dividenden-CAGR über bis zu 5 Jahre (in %).
    """
    if div_series is None or div_series.empty:
        return None
    yearly = div_series.groupby(div_series.index.year).sum().sort_index()
    if len(yearly) < 2:
        return None

    first_year = yearly.index[0]
    last_year = yearly.index[-1]
    n_years = last_year - first_year
    if n_years <= 0:
        return None
    if n_years > 5:
        first_year = last_year - 5
        yearly = yearly[yearly.index >= first_year]
        if len(yearly) < 2:
            return None

    first_val = float(yearly.iloc[0])
    last_val = float(yearly.iloc[-1])
    if first_val <= 0 or last_val <= 0:
        return None

    n_years = yearly.index[-1] - yearly.index[0]
    if n_years <= 0:
        return None

    cagr = (last_val / first_val) ** (1 / n_years) - 1
    return cagr * 100.0


def fetch_metrics(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Holt Kursdaten (1y Historie) + Fundamentaldaten für jeden Ticker.
    Alle Prozentgrößen werden in % gespeichert.
    """
    result: Dict[str, Dict[str, Optional[float]]] = {}
    errors: List[str] = []

    for ticker in tickers:
        metrics: Dict[str, Optional[float]] = {
            "last_price": None,
            "change_1d": None,
            "change_5d": None,
            "change_1y": None,
            "vol_30d": None,
            "vol_1y": None,
            "market_cap": None,
            "pe_ratio": None,
            "forward_pe": None,
            "pb_ratio": None,
            "ps_ratio": None,
            "ev_ebitda": None,
            "net_margin": None,
            "operating_margin": None,
            "roe": None,
            "roa": None,
            "dividend_yield": None,
            "dividend_per_share": None,
            "payout_ratio": None,
            "dividend_growth_5y": None,
            "dividend_frequency": None,
            "debt_to_equity": None,  # Faktor (x)
            "net_debt_ebitda": None,
            "history": None,
        }

        try:
            t = yf.Ticker(ticker)

            # --- Kurs-Historie (1 Jahr) ---
            hist = t.history(period="1y")
            if not hist.empty:
                close = hist["Close"]
                metrics["history"] = close

                last_price = float(close.iloc[-1])
                metrics["last_price"] = last_price

                rets = close.pct_change().dropna()

                def pct_change_from(idx_offset: int) -> Optional[float]:
                    if len(close) <= idx_offset:
                        return None
                    base = float(close.iloc[-1 - idx_offset])
                    if base == 0:
                        return None
                    return (last_price - base) / base * 100.0

                metrics["change_1d"] = pct_change_from(1)
                metrics["change_5d"] = pct_change_from(5)
                metrics["change_1y"] = pct_change_from(len(close) - 1)

                # Volatilitäten in %
                if len(rets) >= 30:
                    metrics["vol_30d"] = float(rets.tail(30).std() * sqrt(252) * 100.0)
                if len(rets) > 0:
                    metrics["vol_1y"] = float(rets.std() * sqrt(252) * 100.0)

            # --- Fundamentaldaten ---
            info = t.info
            metrics["market_cap"] = info.get("marketCap")
            metrics["pe_ratio"] = info.get("trailingPE")
            metrics["forward_pe"] = info.get("forwardPE")
            metrics["pb_ratio"] = info.get("priceToBook")
            metrics["ps_ratio"] = info.get("priceToSalesTrailing12Months")
            metrics["ev_ebitda"] = info.get("enterpriseToEbitda")

            metrics["net_margin"] = to_percent(info.get("profitMargins"))
            metrics["operating_margin"] = to_percent(info.get("operatingMargins"))
            metrics["roe"] = to_percent(info.get("returnOnEquity"))
            metrics["roa"] = to_percent(info.get("returnOnAssets"))

            metrics["dividend_yield"] = to_percent(info.get("dividendYield"))
            metrics["dividend_per_share"] = info.get("dividendRate")
            metrics["payout_ratio"] = to_percent(info.get("payoutRatio"))

            # Debt-to-Equity: Faktor (x) aus Prozent
            dte_raw = info.get("debtToEquity")
            if dte_raw is not None:
                dte = float(dte_raw)
                if dte > 0:
                    metrics["debt_to_equity"] = dte / 100.0

            total_debt = info.get("totalDebt")
            total_cash = info.get("totalCash")
            ebitda = info.get("ebitda")
            if (
                total_debt is not None
                and total_cash is not None
                and ebitda not in (None, 0)
            ):
                net_debt = float(total_debt) - float(total_cash)
                metrics["net_debt_ebitda"] = float(net_debt) / float(ebitda)

            # --- Dividendenhistorie ---
            divs = t.dividends
            if divs is not None and not divs.empty:
                metrics["dividend_frequency"] = infer_dividend_frequency(divs)
                metrics["dividend_growth_5y"] = calc_dividend_growth_5y(divs)

        except Exception as e:
            errors.append(f"{ticker}: {e}")

        result[ticker] = metrics

    if errors:
        st.sidebar.error("Probleme beim Laden der Daten:")
        for msg in errors:
            st.sidebar.write("- ", msg)

    return result


# ---------- News / Sentiment (Google News RSS) ----------

POSITIVE_WORDS = {
    "beat", "record", "strong", "upgrades", "buy", "outperform",
    "bullish", "profit", "growth", "better-than-expected", "raised"
}
NEGATIVE_WORDS = {
    "warning", "profit warning", "miss", "downgrade", "sell",
    "weak", "loss", "fraud", "investigation", "lawsuit", "cut"
}


def simple_sentiment(text: str) -> float:
    """
    Sehr einfache Sentiment-Heuristik auf Wortbasis.
    """
    if not text:
        return 0.0
    t = text.lower()
    score = 0
    for w in POSITIVE_WORDS:
        if w in t:
            score += 1
    for w in NEGATIVE_WORDS:
        if w in t:
            score -= 1
    return float(score)


def fetch_news_for_ticker(
    ticker: str,
    days_back: int = 30,
    max_entries: int = 50,
) -> pd.DataFrame:
    """
    Holt News über Google News RSS für einen Ticker.
    Gibt DataFrame mit published (datetime), title, summary, link, sentiment_* zurück.
    """
    query = f'"{ticker}" stock OR Aktie'
    url = (
        "https://news.google.com/rss/search?"
        f"q={query}&hl=de&gl=DE&ceid=DE:de"
    )

    try:
        feed = feedparser.parse(url)
    except Exception:
        return pd.DataFrame(columns=["published", "title", "summary", "link",
                                     "sentiment_score", "sentiment_label"])

    if not feed.entries:
        return pd.DataFrame(columns=["published", "title", "summary", "link",
                                     "sentiment_score", "sentiment_label"])

    cutoff = datetime.utcnow() - timedelta(days=days_back)
    rows = []

    for entry in feed.entries:
        title = entry.get("title", "") or ""
        summary = entry.get("summary", "") or ""
        link = entry.get("link", "") or ""
        published_raw = entry.get("published") or entry.get("updated") or ""
        try:
            published = dateparser.parse(published_raw)
        except Exception:
            continue

        if published < cutoff:
            continue

        sent_score = simple_sentiment(f"{title} {summary}")
        if sent_score > 0:
            sent_label = "positiv"
        elif sent_score < 0:
            sent_label = "negativ"
        else:
            sent_label = "neutral"

        rows.append(
            {
                "published": published,
                "title": title,
                "summary": summary,
                "link": link,
                "sentiment_score": sent_score,
                "sentiment_label": sent_label,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["published", "title", "summary", "link",
                                     "sentiment_score", "sentiment_label"])

    df = (
        pd.DataFrame(rows)
        .sort_values("published", ascending=False)
        .head(max_entries)
        .reset_index(drop=True)
    )
    return df


# ---------- Graph aufbauen & zeichnen ----------

def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in df.iterrows():
        ticker = row["ticker_yahoo"]
        sector = row["sector"]
        price = row.get("last_price", None)
        label_price = "n/a" if pd.isna(price) else f"{price:.2f} €"
        G.add_node(
            ticker,
            label=f"{ticker}\n{label_price}",
            sector=sector,
        )
    for sector in df["sector"].dropna().unique():
        same_sector = df[df["sector"] == sector]["ticker_yahoo"].tolist()
        for i in range(len(same_sector)):
            for j in range(i + 1, len(same_sector)):
                G.add_edge(same_sector[i], same_sector[j], relation="same_sector")
    return G


def draw_graph(G: nx.Graph):
    if len(G.nodes) == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------- Score & Farben ----------

def compute_total_score(row: pd.Series) -> Optional[float]:
    """
    Einfacher Gesamt-Score 0–100 basierend auf:
    - ROE (%)
    - Nettomarge (%)
    - Dividendenrendite (%)
    - Verschuldung (Debt/Equity als Faktor)
    - Volatilität 1y (%)
    """
    score = 0.0
    max_score = 0.0

    # ROE
    roe = row.get("roe")
    if pd.notna(roe):
        max_score += 25
        if roe >= 20:
            score += 25
        elif roe >= 15:
            score += 20
        elif roe >= 10:
            score += 15
        elif roe >= 5:
            score += 10

    # Nettomarge
    net_margin = row.get("net_margin")
    if pd.notna(net_margin):
        max_score += 20
        if net_margin >= 20:
            score += 20
        elif net_margin >= 15:
            score += 16
        elif net_margin >= 10:
            score += 12
        elif net_margin >= 5:
            score += 8

    # Dividendenrendite (in %)
    div_yield = row.get("dividend_yield")
    if pd.notna(div_yield):
        max_score += 20
        if 2 <= div_yield <= 6:
            score += 20
        elif 1 <= div_yield < 2:
            score += 10
        elif 6 < div_yield <= 10:
            score += 8

    # Verschuldung – Debt/Equity Faktor
    dte = row.get("debt_to_equity")
    if pd.notna(dte):
        max_score += 15
        if dte < 0.5:
            score += 15
        elif dte < 1.0:
            score += 12
        elif dte < 2.0:
            score += 8
        elif dte < 3.0:
            score += 4

    # Volatilität 1y (in %)
    vol_1y = row.get("vol_1y")
    if pd.notna(vol_1y):
        max_score += 20
        if vol_1y <= 20:
            score += 20
        elif vol_1y <= 30:
            score += 15
        elif vol_1y <= 40:
            score += 8

    if max_score == 0:
        return None

    total = score / max_score * 100.0
    return round(total, 1)


def colorize_change(val):
    if pd.isna(val):
        return ""
    color = "green" if val > 0 else "red" if val < 0 else "black"
    return f"color: {color}"


def colorize_score(val):
    if pd.isna(val):
        return ""
    if val >= 70:
        color = "green"
    elif val >= 40:
        color = "orange"
    else:
        color = "red"
    return f"color: {color}"


# ---------- Streamlit App ----------

def main():
    st.title("DAX40 Explorer – Mini-Tool mit Fundamentaldaten")

    df = load_companies()

    # --- Filter ---
    st.sidebar.header("Filter")
    sector_options = ["Alle"] + sorted(df["sector"].dropna().unique().tolist())
    selected_sector = st.sidebar.selectbox("Sektor", sector_options)
    search_text = st.sidebar.text_input(
        "Suche nach Firmenname oder Ticker (optional)"
    ).strip()

    # --- Daten laden ---
    st.subheader("Schritt 1: Kurse, Performance & Fundamentaldaten laden")
    if st.button("Daten laden / aktualisieren"):
        tickers = df["ticker_yahoo"].tolist()
        metrics = fetch_metrics(tickers)

        def map_metric(col_name: str):
            return df["ticker_yahoo"].map(
                lambda t: metrics.get(t, {}).get(col_name)
            )

        # Kurs & Performance
        df["last_price"] = map_metric("last_price")
        df["change_1d"] = map_metric("change_1d")
        df["change_5d"] = map_metric("change_5d")
        df["change_1y"] = map_metric("change_1y")
        df["vol_30d"] = map_metric("vol_30d")
        df["vol_1y"] = map_metric("vol_1y")

        # Bewertung & Profitabilität
        df["market_cap"] = map_metric("market_cap")
        df["pe_ratio"] = map_metric("pe_ratio")
        df["forward_pe"] = map_metric("forward_pe")
        df["pb_ratio"] = map_metric("pb_ratio")
        df["ps_ratio"] = map_metric("ps_ratio")
        df["ev_ebitda"] = map_metric("ev_ebitda")
        df["net_margin"] = map_metric("net_margin")
        df["operating_margin"] = map_metric("operating_margin")
        df["roe"] = map_metric("roe")
        df["roa"] = map_metric("roa")

        # Dividende & Verschuldung
        df["dividend_yield"] = map_metric("dividend_yield")
        df["dividend_per_share"] = map_metric("dividend_per_share")
        df["payout_ratio"] = map_metric("payout_ratio")
        df["dividend_growth_5y"] = map_metric("dividend_growth_5y")
        df["dividend_frequency"] = map_metric("dividend_frequency")
        df["debt_to_equity"] = map_metric("debt_to_equity")
        df["net_debt_ebitda"] = map_metric("net_debt_ebitda")

        # Gesamt-Score
        df["total_score"] = df.apply(compute_total_score, axis=1)

        st.session_state["metrics"] = metrics
        st.session_state["df_with_metrics"] = df.copy()

        st.success("Daten aktualisiert!")
    else:
        if "df_with_metrics" in st.session_state:
            df = st.session_state["df_with_metrics"]
        else:
            for col in [
                "last_price", "change_1d", "change_5d", "change_1y",
                "vol_30d", "vol_1y", "market_cap", "pe_ratio", "forward_pe",
                "pb_ratio", "ps_ratio", "ev_ebitda", "net_margin",
                "operating_margin", "roe", "roa", "dividend_yield",
                "dividend_per_share", "payout_ratio", "dividend_growth_5y",
                "dividend_frequency", "debt_to_equity", "net_debt_ebitda",
                "total_score",
            ]:
                if col not in df.columns:
                    df[col] = None

    # --- Filter anwenden ---
    df_view = df.copy()
    if selected_sector != "Alle":
        df_view = df_view[df_view["sector"] == selected_sector]

    if search_text:
        mask = df_view["name"].str.contains(search_text, case=False, na=False) | \
               df_view["ticker_yahoo"].str.contains(search_text, case=False, na=False)
        df_view = df_view[mask]

    # --- Tabs ---
    tab_overview, tab_funda, tab_risk, tab_sector, tab_news = st.tabs(
        ["Überblick", "Fundamentaldaten", "Risiko-Panel", "Sektor-Übersicht", "News & Events"]
    )

    # --- Überblick ---
    with tab_overview:
        st.subheader("Überblick: Kurs & Performance")

        overview_cols = [
            "name",
            "ticker_yahoo",
            "sector",
            "last_price",
            "change_1d",
            "change_5d",
            "change_1y",
            "vol_30d",
            "vol_1y",
            "total_score",
        ]

        styled_overview = (
            df_view[overview_cols]
            .style.format(
                {
                    "last_price": "{:.2f}",
                    "change_1d": "{:+.2f}%",
                    "change_5d": "{:+.2f}%",
                    "change_1y": "{:+.2f}%",
                    "vol_30d": "{:.2f}%",
                    "vol_1y": "{:.2f}%",
                    "total_score": "{:.1f}",
                },
                na_rep="–",
            )
            .applymap(colorize_change, subset=["change_1d", "change_5d", "change_1y"])
            .applymap(colorize_score, subset=["total_score"])
        )

        st.dataframe(styled_overview, use_container_width=True)

    # --- Fundamentaldaten ---
    with tab_funda:
        st.subheader("Fundamentaldaten")

        df_funda = df_view.copy()
        df_funda["market_cap_billion"] = df_funda["market_cap"] / 1e9

        funda_cols = [
            "name",
            "ticker_yahoo",
            "total_score",
            "market_cap_billion",
            "pe_ratio",
            "forward_pe",
            "pb_ratio",
            "ps_ratio",
            "ev_ebitda",
            "net_margin",
            "operating_margin",
            "roe",
            "roa",
            "dividend_yield",
            "dividend_per_share",
            "payout_ratio",
            "dividend_growth_5y",
            "dividend_frequency",
            "debt_to_equity",
            "net_debt_ebitda",
        ]

        styled_funda = (
            df_funda[funda_cols]
            .style.format(
                {
                    "market_cap_billion": "{:.1f}",
                    "pe_ratio": "{:.2f}",
                    "forward_pe": "{:.2f}",
                    "pb_ratio": "{:.2f}",
                    "ps_ratio": "{:.2f}",
                    "ev_ebitda": "{:.2f}",
                    "net_margin": "{:.2f}%",
                    "operating_margin": "{:.2f}%",
                    "roe": "{:.2f}%",
                    "roa": "{:.2f}%",
                    "dividend_yield": "{:.2f}%",
                    "dividend_per_share": "{:.2f}",
                    "payout_ratio": "{:.2f}%",
                    "dividend_growth_5y": "{:.2f}%",
                    "debt_to_equity": "{:.2f}x",
                    "net_debt_ebitda": "{:.2f}",
                    "total_score": "{:.1f}",
                },
                na_rep="–",
            )
        )

        styled_funda = styled_funda.applymap(
            colorize_change,
            subset=[
                "net_margin",
                "operating_margin",
                "roe",
                "roa",
                "dividend_yield",
                "dividend_growth_5y",
            ],
        ).applymap(
            colorize_score,
            subset=["total_score"],
        )

        st.dataframe(styled_funda, use_container_width=True)
        st.caption(
            "Market Cap in Milliarden (Basiswährung laut Börsenplatz). "
            "Debt/Equity als Faktor (x). Prozentwerte sind bereits skaliert."
        )

    # --- Risiko-Panel ---
    with tab_risk:
        st.subheader("Risiko-Panel für einzelne Aktie")

        if df_view.empty:
            st.info("Keine Firmen für diese Filtereinstellung.")
        else:
            ticker_choice = st.selectbox(
                "Ticker auswählen",
                options=df_view["ticker_yahoo"].tolist(),
            )
            row = df_view[df_view["ticker_yahoo"] == ticker_choice].iloc[0]

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Volatilität 1J",
                f"{row['vol_1y']:.1f}%" if pd.notna(row["vol_1y"]) else "–",
            )
            col2.metric(
                "Debt/Equity (x)",
                f"{row['debt_to_equity']:.2f}x" if pd.notna(row["debt_to_equity"]) else "–",
            )
            col3.metric(
                "Gesamt-Score",
                f"{row['total_score']:.1f}" if pd.notna(row["total_score"]) else "–",
            )

            st.write("**Dividende & Qualität**")
            col4, col5, col6 = st.columns(3)
            dy = row["dividend_yield"]
            dg = row["dividend_growth_5y"]
            roe = row["roe"]
            col4.metric(
                "Div.-Rendite",
                f"{dy:.1f}%" if pd.notna(dy) else "–",
            )
            col5.metric(
                "Div.-Wachstum 5J",
                f"{dg:.1f}%" if pd.notna(dg) else "–",
            )
            col6.metric(
                "ROE",
                f"{roe:.1f}%" if pd.notna(roe) else "–",
            )

    # --- Sektor-Übersicht ---
    with tab_sector:
        st.subheader("Sektor-Übersicht")

        if df_view.empty:
            st.info("Keine Firmen für diese Filtereinstellung.")
        else:
            sector_stats = (
                df_view.groupby("sector")
                .agg(
                    anzahl=("ticker_yahoo", "count"),
                    avg_change_1y=("change_1y", "mean"),
                    avg_score=("total_score", "mean"),
                )
                .sort_values("avg_change_1y", ascending=False)
            )
            st.dataframe(
                sector_stats.style.format(
                    {
                        "avg_change_1y": "{:+.2f}%",
                        "avg_score": "{:.1f}",
                    },
                    na_rep="–",
                ),
                use_container_width=True,
            )
            st.bar_chart(sector_stats["avg_change_1y"])

    # --- News & Events ---
    with tab_news:
        st.subheader("News & Events")

        if df_view.empty:
            st.info("Keine Firmen für diese Filtereinstellung.")
        else:
            news_ticker = st.selectbox(
                "Ticker für News auswählen",
                options=df_view["ticker_yahoo"].tolist(),
                key="news_ticker_select",
            )

            days_back = st.slider(
                "Zeitraum (Tage)",
                min_value=3,
                max_value=90,
                value=30,
                step=1,
            )

            if st.button("News laden", key="load_news_button"):
                news_df = fetch_news_for_ticker(
                    news_ticker,
                    days_back=days_back,
                    max_entries=50,
                )
                st.session_state["news_df"] = news_df

                # Events für den Chart erzeugen
                if not news_df.empty:
                    events_rows = []
                    for _, r in news_df.iterrows():
                        etype = {
                            "positiv": "news_pos",
                            "negativ": "news_neg",
                            "neutral": "news_neutral",
                        }.get(r["sentiment_label"], "news_neutral")
                        events_rows.append(
                            {
                                "date": r["published"],
                                "event_type": etype,
                                "title": r["title"],
                                "link": r["link"],
                            }
                        )
                    st.session_state["events_df"] = pd.DataFrame(events_rows)

            news_df = st.session_state.get("news_df")

            if news_df is None or news_df.empty:
                st.info("Noch keine News geladen oder keine Einträge im Feed.")
            else:
                news_df = news_df.reset_index(drop=True)

                st.write("Neueste Meldungen (mit grober Sentiment-Einschätzung):")
                st.dataframe(
                    news_df[["published", "title", "sentiment_label", "link"]],
                    use_container_width=True,
                )

                df_plot = news_df.dropna(subset=["published"]).copy()
                if not df_plot.empty:
                    df_plot["published_date"] = pd.to_datetime(df_plot["published"]).dt.date

                    chart = (
                        alt.Chart(df_plot)
                        .mark_point(size=70)
                        .encode(
                            x="published_date:T",
                            y=alt.value(0),
                            color=alt.Color(
                                "sentiment_label:N",
                                scale=alt.Scale(
                                    domain=["negativ", "neutral", "positiv"],
                                    range=["red", "gray", "green"],
                                ),
                                legend=alt.Legend(title="Sentiment"),
                            ),
                            tooltip=["published", "title", "sentiment_label", "link"],
                        )
                        .properties(height=120)
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Keine datierten News-Einträge für die Timeline gefunden.")

    # --- Mini-Chart / Sparkline mit Event-Layer ---
    st.subheader("Schritt 3: Mini-Kurscharts (Sparkline)")

    if "metrics" in st.session_state and st.session_state["metrics"]:
        available_tickers = df_view["ticker_yahoo"].tolist()
        if available_tickers:
            selected_ticker = st.selectbox(
                "Wähle einen Ticker für den Mini-Chart:",
                options=available_tickers,
            )

            period_choice = st.selectbox(
                "Zeitraum",
                options=["2 Monate", "6 Monate", "1 Jahr", "5 Jahre"],
                index=0,
            )

            show_smas = st.checkbox("SMA 20/50/200 anzeigen", value=True)

            metrics_store = st.session_state["metrics"]
            base_series = metrics_store.get(selected_ticker, {}).get("history", None)

            series = None
            if period_choice == "5 Jahre":
                try:
                    t = yf.Ticker(selected_ticker)
                    hist5 = t.history(period="5y")
                    if not hist5.empty:
                        series = hist5["Close"]
                    else:
                        series = base_series
                except Exception:
                    series = base_series
            else:
                series = base_series

            if series is not None and not series.empty:
                if period_choice == "2 Monate":
                    series_window = series.tail(min(60, len(series)))
                elif period_choice == "6 Monate":
                    series_window = series.tail(min(130, len(series)))
                elif period_choice == "1 Jahr":
                    series_window = series.tail(min(252, len(series)))
                else:  # 5 Jahre
                    series_window = series

                spark = series_window.to_frame(name="Kurs")

                if show_smas:
                    spark["SMA20"] = spark["Kurs"].rolling(window=20).mean()
                    spark["SMA50"] = spark["Kurs"].rolling(window=50).mean()
                    spark["SMA200"] = spark["Kurs"].rolling(window=200).mean()

                chart_df = spark.reset_index().rename(columns={"index": "Datum"})

                # Basis-Linie
                base_line = alt.Chart(chart_df).mark_line().encode(
                    x=alt.X("Datum:T", title="Datum"),
                    y=alt.Y("Kurs:Q", title="Preis"),
                    tooltip=["Datum", "Kurs"]
                )

                # SMA-Linien
                sma_layers = []
                for sma in ["SMA20", "SMA50", "SMA200"]:
                    if sma in chart_df.columns:
                        sma_layers.append(
                            alt.Chart(chart_df).mark_line(strokeDash=[3, 3]).encode(
                                x="Datum:T",
                                y=f"{sma}:Q",
                                color=alt.value("#888"),
                                tooltip=["Datum", sma],
                            )
                        )

                # Event-Layer (News-Icons)
                events_df = st.session_state.get(
                    "events_df",
                    pd.DataFrame(columns=["date", "event_type", "title", "link"])
                )

                event_layer = None
                if not events_df.empty:
                    icon_map = {
                        "news_pos": "★",
                        "news_neg": "‼",
                        "news_neutral": "•",
                        "earnings": "◆",
                        "upgrade": "⬆",
                        "downgrade": "⬇",
                        "hv": "📅",
                        "report": "📄",
                    }
                    events_df_plot = events_df.copy()
                    events_df_plot["date"] = pd.to_datetime(events_df_plot["date"])
                    events_df_plot["icon"] = events_df_plot["event_type"].map(icon_map).fillna("•")

                    event_layer = alt.Chart(events_df_plot).mark_text(
                        align="center",
                        baseline="middle",
                        fontSize=18,
                    ).encode(
                        x="date:T",
                        y=alt.value(0),
                        text="icon:N",
                        tooltip=["date", "event_type", "title", "link"],
                    )

                all_layers = [base_line] + sma_layers
                if event_layer is not None:
                    all_layers.append(event_layer)

                final_chart = alt.layer(*all_layers).resolve_scale(
                    y="shared"
                ).properties(
                    width="container",
                    height=300,
                )

                st.altair_chart(final_chart, use_container_width=True)
            else:
                st.info("Für diesen Ticker ist keine Kurs-Historie verfügbar.")
        else:
            st.info("Keine Firmen für diese Filtereinstellung.")
    else:
        st.info("Bitte zuerst oben auf **'Daten laden / aktualisieren'** klicken.")

    # --- Graph-Ansicht ---
    st.subheader("Schritt 4: Graph-Ansicht (Mindmap-light)")
    st.write(
        "Hier siehst du die ausgewählten Firmen als Knoten. "
        "Kanten verbinden Firmen im gleichen Sektor."
    )

    if df_view.empty:
        st.info("Keine Firmen für diese Filtereinstellung gefunden.")
        return

    G = build_graph(df_view)
    fig = draw_graph(G)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("Graph konnte nicht gezeichnet werden (keine Knoten).")


if __name__ == "__main__":
    main()
