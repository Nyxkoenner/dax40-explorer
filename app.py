"""
Aktien Explorer – überarbeitete Einzeldatei für Streamlit.

Start:
    python -m streamlit run app.py

Hinweise:
- Die App nutzt yfinance als Datenquelle. Daten können fehlen, verzögert oder
  von Yahoo geändert sein. Sie dienen der Recherche und sind keine Anlageberatung.
- Lege optional eine portfolio.csv in denselben Ordner. Beispiel:
  ticker_yahoo,shares,cost_basis,currency,purchase_date
  ALV.DE,10,245.50,EUR,2024-01-15
  MSFT,5,390.00,USD,2024-02-10
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Optional

import altair as alt
import feedparser
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from dateutil import parser as dateparser


# -----------------------------------------------------------------------------
# App-Konfiguration
# -----------------------------------------------------------------------------

APP_VERSION = "2.1"
APP_TITLE = "Aktien Explorer"
BASE_CURRENCY = "EUR"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "indices"
CACHE_DIR = ROOT_DIR / ".cache"
WATCHLIST_PATH = DATA_DIR / "watchlist.csv"
PORTFOLIO_PATH = ROOT_DIR / "portfolio.csv"

for directory in (DATA_DIR, INDEX_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

DEFAULT_DRAWDOWN_TRIGGER = 25.0
DEFAULT_PAYOUT_MAX = 90.0
DEFAULT_SCORE_MIN = 65.0
DEFAULT_YIELD_MIN = 5.0
MIN_SCORE_COVERAGE_FOR_TRIGGER = 55.0

SMA_COLORS = {
    "SMA 20": "#ef4444",
    "SMA 50": "#f59e0b",
    "SMA 200": "#14b8a6",
}

# Hohe Renditen sind je nach Sektor unterschiedlich normal. Diese Werte sind
# absichtlich konservativer als in der alten Version und im UI anpassbar.
YIELD_TRIGGER_BY_SECTOR = {
    "Utilities": 4.5,
    "Real Estate": 5.5,
    "Financials": 5.5,
    "Energy": 5.5,
    "Consumer Staples": 4.5,
    "Communication Services": 4.5,
    "Telecommunication Services": 4.5,
    "_default": 5.0,
}

FINANCIAL_SECTOR_TERMS = {
    "financial",
    "bank",
    "insurance",
    "asset management",
    "capital markets",
}

POSITIVE_WORDS = {
    "beat", "beats", "record", "strong", "upgrade", "upgrades", "buy",
    "outperform", "bullish", "profit", "growth", "raised", "raise",
    "besser als erwartet", "rekord", "stark", "angehoben", "kaufen",
    "gewinn", "wachstum", "übertrifft", "uebertrifft",
}

NEGATIVE_WORDS = {
    "warning", "profit warning", "miss", "downgrade", "sell", "weak",
    "loss", "fraud", "investigation", "lawsuit", "cut", "cuts",
    "warnung", "gewinnwarnung", "verfehlt", "herabgestuft", "verkaufen",
    "schwach", "verlust", "betrug", "untersuchung", "klage", "senken",
}

RSS_SOURCES = [
    "https://www.tagesschau.de/wirtschaft/unternehmen/index~rss2.xml",
    "https://www.finanzen.net/rss/news",
    "https://www.onvista.de/news/feed/aktien",
    "https://www.handelsblatt.com/contentexport/feed/wirtschaft",
    "https://www.eqs-news.com/de/news/dgap/rss",
]


# -----------------------------------------------------------------------------
# Allgemeine Hilfsfunktionen
# -----------------------------------------------------------------------------


def safe_float(value: Any) -> Optional[float]:
    """Wandelt Werte robust in float um; unbrauchbare Werte werden None."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def to_percent(value: Any) -> Optional[float]:
    """Yahoo liefert einige Quoten als Dezimalzahl, andere bereits in Prozent."""
    number = safe_float(value)
    if number is None:
        return None
    return number * 100.0 if abs(number) <= 1 else number


def clean_ticker(ticker: Any) -> str:
    return str(ticker or "").strip().upper()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def find_col(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Findet Spalten auch bei leicht abweichender Wikipedia-Benennung."""
    available = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in available:
            return available[candidate.lower()]

    for column in columns:
        column_lower = str(column).lower()
        if any(candidate.lower() in column_lower for candidate in candidates):
            return str(column)
    return None


def format_number(value: Any, decimals: int = 2, suffix: str = "") -> str:
    """Formatiert Zahlen im deutschen Stil: 1.234.567,89."""
    number = safe_float(value)
    if number is None:
        return "–"
    return (
        f"{number:,.{decimals}f}{suffix}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def format_percent(value: Any, decimals: int = 1, signed: bool = False) -> str:
    """Formatiert Prozentwerte im deutschen Stil."""
    number = safe_float(value)
    if number is None:
        return "–"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{format_number(number, decimals)} %"


def format_eur(value: Any, decimals: int = 2, signed: bool = False) -> str:
    """Formatiert Beträge, die intern bereits in Euro vorliegen."""
    number = safe_float(value)
    if number is None:
        return "–"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{format_number(number, decimals)} EUR"


def human_market_cap(value: Any) -> str:
    """Kompakte Marktkapitalisierung: Mio., Mrd. oder Bio. mit deutschem Zahlenformat."""
    number = safe_float(value)
    if number is None:
        return "–"

    absolute = abs(number)
    if absolute >= 1_000_000_000_000:
        return f"{format_number(number / 1_000_000_000_000, 1)} Bio."
    if absolute >= 1_000_000_000:
        return f"{format_number(number / 1_000_000_000, 1)} Mrd."
    if absolute >= 1_000_000:
        return f"{format_number(number / 1_000_000, 1)} Mio."
    return format_number(number, 0)


def is_financial_sector(sector: Any) -> bool:
    text = str(sector or "").lower()
    return any(term in text for term in FINANCIAL_SECTOR_TERMS)


def ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result[~result.index.isna()]
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)
    return result.sort_index()


def empty_metrics_frame() -> pd.DataFrame:
    columns = [
        "name", "ticker_yahoo", "sector", "currency", "last_price", "change_1d",
        "change_5d", "change_1y", "vol_30d", "vol_1y", "high_52w", "low_52w",
        "market_cap", "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda",
        "net_margin", "operating_margin", "roe", "roa", "dividend_yield",
        "dividend_per_share", "payout_ratio", "dividend_growth_5y",
        "dividend_frequency", "debt_to_equity", "net_debt_ebitda", "data_updated_at",
    ]
    return pd.DataFrame(columns=columns)


# -----------------------------------------------------------------------------
# Indexdaten: CSV-Fallback und Wikipedia
# -----------------------------------------------------------------------------


def fetch_html(url: str, retries: int = 3, timeout: int = 20) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            response.raise_for_status()
            html = response.text
            if "<table" not in html.lower():
                raise RuntimeError("Die Antwort enthält keine HTML-Tabelle.")
            return html
        except Exception as error:  # externe Quelle: Fehler sind erwartbar
            last_error = error
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"Abruf fehlgeschlagen: {last_error}")


def read_html_tables(html: str) -> list[pd.DataFrame]:
    return pd.read_html(StringIO(html))


def validate_constituents(df: pd.DataFrame) -> pd.DataFrame:
    required = {"name", "ticker_yahoo", "sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Indexdatei enthält nicht alle benötigten Spalten: {sorted(missing)}")

    result = df[["name", "ticker_yahoo", "sector"]].copy()
    result["name"] = result["name"].astype(str).str.strip()
    result["ticker_yahoo"] = result["ticker_yahoo"].map(clean_ticker)
    result["sector"] = result["sector"].fillna("Unbekannt").astype(str).str.strip()
    result = result[result["ticker_yahoo"].ne("")]
    return result.drop_duplicates(subset=["ticker_yahoo"]).reset_index(drop=True)


def parse_dax_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        table = normalize_columns(table)
        columns = list(table.columns)
        name_col = find_col(columns, ["Company", "Unternehmen", "Name", "Constituent", "Security"])
        ticker_col = find_col(columns, ["Ticker symbol", "Ticker", "Symbol"])
        sector_col = find_col(columns, ["Industry", "Sector", "Industrie", "Branche"])

        if not name_col or not ticker_col:
            continue
        if any(word in ticker_col.lower() for word in ("isin", "wkn")):
            continue

        result = table[[name_col, ticker_col]].copy()
        result.columns = ["name", "ticker_yahoo"]
        result["sector"] = table[sector_col] if sector_col else "Unbekannt"
        result["ticker_yahoo"] = (
            result["ticker_yahoo"].astype(str)
            .str.replace(r"\[.*?\]", "", regex=True)
            .str.strip()
        )
        result = result[result["ticker_yahoo"].str.len().gt(0)]
        result["ticker_yahoo"] = result["ticker_yahoo"].apply(
            lambda ticker: ticker if "." in ticker else f"{ticker}.DE"
        )
        result = validate_constituents(result)
        if len(result) >= 30:
            return result
    raise RuntimeError("Keine passende DAX-Tabelle mit Yahoo-Tickern gefunden.")


def parse_sp500_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        table = normalize_columns(table)
        columns = list(table.columns)
        name_col = find_col(columns, ["Security", "Company", "Name"])
        ticker_col = find_col(columns, ["Symbol", "Ticker", "Ticker symbol"])
        sector_col = find_col(columns, ["GICS Sector", "Sector"])
        if not (name_col and ticker_col and sector_col):
            continue

        result = table[[name_col, ticker_col, sector_col]].copy()
        result.columns = ["name", "ticker_yahoo", "sector"]
        result["ticker_yahoo"] = result["ticker_yahoo"].astype(str).str.replace(".", "-", regex=False)
        result = validate_constituents(result)
        if len(result) >= 400:
            return result
    raise RuntimeError("Keine passende S&P-500-Tabelle gefunden.")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_index_constituents(index_name: str) -> pd.DataFrame:
    """Lädt zuerst eine optionale lokale CSV, sonst Wikipedia als Fallback."""
    local_files = {
        "DAX 40": INDEX_DIR / "dax40.csv",
        "S&P 500": INDEX_DIR / "sp500.csv",
    }
    local_path = local_files.get(index_name)
    if local_path and local_path.exists():
        return validate_constituents(pd.read_csv(local_path))

    if index_name == "DAX 40":
        errors: list[str] = []
        for url in ("https://en.wikipedia.org/wiki/DAX", "https://de.wikipedia.org/wiki/DAX"):
            try:
                return parse_dax_tables(read_html_tables(fetch_html(url)))
            except Exception as error:
                errors.append(f"{url}: {error}")
        raise RuntimeError("DAX konnte nicht geladen werden. " + " | ".join(errors))

    if index_name == "S&P 500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        return parse_sp500_tables(read_html_tables(fetch_html(url)))

    raise ValueError(f"Unbekannter Index: {index_name}")


# -----------------------------------------------------------------------------
# Marktdaten und Dividenden
# -----------------------------------------------------------------------------


def _extract_ticker_history(downloaded: pd.DataFrame, ticker: str, ticker_count: int) -> pd.DataFrame:
    """Extrahiert einen Ticker aus yf.download, unabhängig von Spaltenlayout."""
    if downloaded.empty:
        return pd.DataFrame()

    if ticker_count == 1 and not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded.copy()

    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        level1 = downloaded.columns.get_level_values(1)
        if ticker in level0:
            return downloaded[ticker].copy()
        if ticker in level1:
            return downloaded.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_price_histories(tickers: tuple[str, ...], period: str = "5y") -> dict[str, pd.DataFrame]:
    """Lädt Kursdaten gebündelt; das ist schneller als ein Abruf je Ticker."""
    clean_tickers = tuple(dict.fromkeys(ticker for ticker in tickers if ticker))
    if not clean_tickers:
        return {}

    try:
        downloaded = yf.download(
            list(clean_tickers),
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=True,
            progress=False,
            repair=False,
        )
    except Exception:
        downloaded = pd.DataFrame()

    histories: dict[str, pd.DataFrame] = {}
    for ticker in clean_tickers:
        history = _extract_ticker_history(downloaded, ticker, len(clean_tickers))
        if history.empty:
            try:
                history = yf.Ticker(ticker).history(period=period, auto_adjust=False)
            except Exception:
                history = pd.DataFrame()
        if not history.empty:
            history = ensure_datetime_index(history)
            valid_columns = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in history.columns]
            histories[ticker] = history[valid_columns].copy()
        else:
            histories[ticker] = pd.DataFrame()
    return histories


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict[str, Any]:
    """Beschränkt die gespeicherten Yahoo-Felder auf wirklich benötigte Daten."""
    wanted_keys = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
        "enterpriseToEbitda", "profitMargins", "operatingMargins", "returnOnEquity",
        "returnOnAssets", "dividendYield", "dividendRate", "payoutRatio", "debtToEquity",
        "totalDebt", "totalCash", "ebitda", "currency", "financialCurrency", "beta",
    ]
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}
    return {key: info.get(key) for key in wanted_keys}


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_dividends(ticker: str) -> pd.DataFrame:
    try:
        dividends = yf.Ticker(ticker).dividends
    except Exception:
        dividends = pd.Series(dtype=float)

    if dividends is None or dividends.empty:
        return pd.DataFrame(columns=["date", "amount"])
    frame = dividends.rename("amount").to_frame().reset_index()
    frame.columns = ["date", "amount"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_localize(None)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    return frame.dropna(subset=["date", "amount"]).sort_values("date")


def infer_dividend_frequency(dividend_frame: pd.DataFrame) -> Optional[str]:
    """Erkennt Monats-, Quartals-, Halbjahres- und Jahresdividenden."""
    if dividend_frame is None or dividend_frame.empty:
        return None
    dates = pd.to_datetime(dividend_frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None

    cutoff = dates.max() - pd.Timedelta(days=400)
    count = int((dates >= cutoff).sum())
    if count >= 10:
        return "monatlich"
    if count >= 4:
        return "quartalsweise"
    if count in (2, 3):
        return "halbjährlich"
    if count == 1:
        return "jährlich"
    return "unregelmäßig"


def calc_dividend_growth_5y(dividend_frame: pd.DataFrame) -> Optional[float]:
    """CAGR nur aus abgeschlossenen Kalenderjahren, nie aus dem laufenden Jahr."""
    if dividend_frame is None or dividend_frame.empty:
        return None

    frame = dividend_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.dropna(subset=["date", "amount"])
    if frame.empty:
        return None

    current_year = datetime.now().year
    frame = frame[frame["date"].dt.year < current_year]
    if frame.empty:
        return None

    yearly = frame.groupby(frame["date"].dt.year)["amount"].sum().sort_index()
    if len(yearly) < 2:
        return None

    latest_year = int(yearly.index.max())
    start_year = latest_year - 5
    selected = yearly[yearly.index >= start_year]
    if len(selected) < 2:
        return None

    first_value = safe_float(selected.iloc[0])
    last_value = safe_float(selected.iloc[-1])
    years = int(selected.index[-1] - selected.index[0])
    if first_value is None or last_value is None or first_value <= 0 or last_value <= 0 or years <= 0:
        return None
    return ((last_value / first_value) ** (1 / years) - 1) * 100.0


def series_change(close: pd.Series, periods_back: int) -> Optional[float]:
    if close is None or len(close) <= periods_back:
        return None
    last_price = safe_float(close.iloc[-1])
    base_price = safe_float(close.iloc[-1 - periods_back])
    if last_price is None or base_price in (None, 0):
        return None
    return (last_price / base_price - 1) * 100.0


def trailing_dividend_per_share(dividend_frame: pd.DataFrame) -> Optional[float]:
    if dividend_frame is None or dividend_frame.empty:
        return None
    latest_date = pd.to_datetime(dividend_frame["date"], errors="coerce").max()
    if pd.isna(latest_date):
        return None
    cutoff = latest_date - pd.Timedelta(days=365)
    result = pd.to_numeric(
        dividend_frame.loc[dividend_frame["date"] >= cutoff, "amount"], errors="coerce"
    ).sum()
    return safe_float(result) if result > 0 else None


def metrics_from_ticker(
    ticker: str,
    name: str,
    sector: str,
    history: pd.DataFrame,
    info: dict[str, Any],
    dividends: pd.DataFrame,
) -> dict[str, Any]:
    """Erstellt eine flache, tabellenfreundliche Kennzahlenzeile."""
    record: dict[str, Any] = {
        "name": name,
        "ticker_yahoo": ticker,
        "sector": sector or "Unbekannt",
        "currency": info.get("currency") or info.get("financialCurrency") or "–",
        "last_price": None,
        "change_1d": None,
        "change_5d": None,
        "change_1y": None,
        "vol_30d": None,
        "vol_1y": None,
        "high_52w": None,
        "low_52w": None,
        "market_cap": safe_float(info.get("marketCap")),
        "pe_ratio": safe_float(info.get("trailingPE")),
        "forward_pe": safe_float(info.get("forwardPE")),
        "pb_ratio": safe_float(info.get("priceToBook")),
        "ps_ratio": safe_float(info.get("priceToSalesTrailing12Months")),
        "ev_ebitda": safe_float(info.get("enterpriseToEbitda")),
        "net_margin": to_percent(info.get("profitMargins")),
        "operating_margin": to_percent(info.get("operatingMargins")),
        "roe": to_percent(info.get("returnOnEquity")),
        "roa": to_percent(info.get("returnOnAssets")),
        "dividend_yield": to_percent(info.get("dividendYield")),
        "dividend_per_share": safe_float(info.get("dividendRate")),
        "payout_ratio": to_percent(info.get("payoutRatio")),
        "dividend_growth_5y": calc_dividend_growth_5y(dividends),
        "dividend_frequency": infer_dividend_frequency(dividends),
        "debt_to_equity": None,
        "net_debt_ebitda": None,
        "beta": safe_float(info.get("beta")),
        "data_updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    debt_to_equity = safe_float(info.get("debtToEquity"))
    if debt_to_equity is not None and debt_to_equity >= 0:
        record["debt_to_equity"] = debt_to_equity / 100.0

    total_debt = safe_float(info.get("totalDebt"))
    total_cash = safe_float(info.get("totalCash"))
    ebitda = safe_float(info.get("ebitda"))
    if total_debt is not None and total_cash is not None and ebitda not in (None, 0):
        record["net_debt_ebitda"] = (total_debt - total_cash) / ebitda

    if history is not None and not history.empty and "Close" in history.columns:
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if not close.empty:
            one_year = close.tail(253)
            returns = one_year.pct_change().dropna()
            record["last_price"] = safe_float(close.iloc[-1])
            record["change_1d"] = series_change(close, 1)
            record["change_5d"] = series_change(close, 5)
            record["change_1y"] = series_change(close, 252) if len(close) >= 253 else None
            record["high_52w"] = safe_float(one_year.max())
            record["low_52w"] = safe_float(one_year.min())
            if len(returns) >= 30:
                record["vol_30d"] = float(returns.tail(30).std() * (252 ** 0.5) * 100)
            if len(returns) >= 2:
                record["vol_1y"] = float(returns.std() * (252 ** 0.5) * 100)

    trailing_dps = trailing_dividend_per_share(dividends)
    if record["dividend_per_share"] is None:
        record["dividend_per_share"] = trailing_dps

    # Plausibilitäts-Fallback: Yahoo liefert die Yield gelegentlich leer oder in
    # einer unpassenden Skalierung. Erst dann rechnen wir aus DPS / letztem Kurs.
    price = safe_float(record["last_price"])
    dps = safe_float(record["dividend_per_share"])
    estimated_yield = dps / price * 100 if price not in (None, 0) and dps not in (None, 0) else None
    current_yield = safe_float(record["dividend_yield"])
    if estimated_yield is not None and (
        current_yield is None or current_yield < 0 or current_yield > 25 or abs(current_yield - estimated_yield) > 15
    ):
        record["dividend_yield"] = estimated_yield

    return record


def collect_metrics(constituents: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    """Lädt Kurse gebündelt; Fundamentaldaten bleiben pro Ticker gecacht."""
    if constituents.empty:
        return empty_metrics_frame(), {}, []

    tickers = tuple(constituents["ticker_yahoo"].map(clean_ticker).tolist())
    histories = download_price_histories(tickers, period="5y")
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    for _, company in constituents.iterrows():
        ticker = clean_ticker(company["ticker_yahoo"])
        try:
            info = fetch_ticker_info(ticker)
            dividends = fetch_dividends(ticker)
            history = histories.get(ticker, pd.DataFrame())
            records.append(
                metrics_from_ticker(
                    ticker=ticker,
                    name=str(company.get("name", ticker)),
                    sector=str(company.get("sector", "Unbekannt")),
                    history=history,
                    info=info,
                    dividends=dividends,
                )
            )
            if history.empty:
                errors.append(f"{ticker}: keine Kursdaten")
        except Exception as error:
            errors.append(f"{ticker}: {type(error).__name__}")

    return pd.DataFrame(records), histories, errors


# -----------------------------------------------------------------------------
# Score und Value-Scanner
# -----------------------------------------------------------------------------


def score_band(value: Optional[float], thresholds: list[tuple[float, float]], default: float = 0.0) -> float:
    if value is None:
        return default
    for minimum, score in thresholds:
        if value >= minimum:
            return score
    return default


def compute_total_score(row: pd.Series) -> pd.Series:
    """Qualitäts-/Stabilitätsscore mit Datenabdeckungsabschlag.

    Er verhindert, dass wenige zufällig vorhandene Spitzenwerte automatisch zu
    einem 100er-Score führen. Negative KGVs erhalten keine Bewertungs-Punkte.
    """
    components: list[tuple[str, Optional[float], float]] = []

    roe = safe_float(row.get("roe"))
    if roe is not None:
        points = score_band(roe, [(20, 20), (15, 16), (10, 12), (5, 8)], 2)
        components.append(("ROE", points, 20))

    margin = safe_float(row.get("net_margin"))
    if margin is not None:
        points = score_band(margin, [(20, 20), (15, 16), (10, 12), (5, 8)], 2)
        components.append(("Netto-Marge", points, 20))

    pe = safe_float(row.get("pe_ratio"))
    pb = safe_float(row.get("pb_ratio"))
    valuation_points: Optional[float] = None
    if pe is not None:
        if pe <= 0:
            valuation_points = 0.0  # Verlustunternehmen sind nicht "günstig" per KGV.
        else:
            valuation_points = score_band(pe, [(0.0, 0)], 0)
            if pe <= 10:
                valuation_points = 20
            elif pe <= 15:
                valuation_points = 16
            elif pe <= 22:
                valuation_points = 11
            elif pe <= 35:
                valuation_points = 5
            else:
                valuation_points = 1
    elif pb is not None and pb > 0:
        valuation_points = 18 if pb <= 1 else 14 if pb <= 2 else 8 if pb <= 4 else 3
    if valuation_points is not None:
        components.append(("Bewertung", valuation_points, 20))

    # D/E ist für Banken/Versicherungen weniger aussagekräftig und wird dort
    # bewusst aus der Datenabdeckung herausgenommen.
    if not is_financial_sector(row.get("sector")):
        debt_to_equity = safe_float(row.get("debt_to_equity"))
        net_debt_ebitda = safe_float(row.get("net_debt_ebitda"))
        if debt_to_equity is not None or net_debt_ebitda is not None:
            dte_points = None
            nde_points = None
            if debt_to_equity is not None:
                dte_points = 20 if debt_to_equity < 0.5 else 16 if debt_to_equity < 1 else 11 if debt_to_equity < 2 else 5 if debt_to_equity < 3 else 0
            if net_debt_ebitda is not None:
                nde_points = 20 if net_debt_ebitda < 1 else 16 if net_debt_ebitda < 2 else 11 if net_debt_ebitda < 3 else 5 if net_debt_ebitda < 4 else 0
            debt_points = max(point for point in (dte_points, nde_points) if point is not None)
            components.append(("Verschuldung", debt_points, 20))

    volatility = safe_float(row.get("vol_1y"))
    if volatility is not None:
        points = 20 if volatility <= 20 else 16 if volatility <= 30 else 10 if volatility <= 40 else 4
        components.append(("Volatilität", points, 20))

    if not components:
        return pd.Series({
            "total_score": None,
            "score_raw": None,
            "score_coverage": 0.0,
            "score_confidence": "keine Daten",
            "score_components": "Keine ausreichenden Kennzahlen",
        })

    available_weight = sum(weight for _, _, weight in components)
    eligible_weight = 100.0 if not is_financial_sector(row.get("sector")) else 80.0
    raw_score = sum(points for _, points, _ in components) / available_weight * 100
    coverage = min(available_weight / eligible_weight * 100, 100.0)
    # Bei geringer Abdeckung wird der Rohscore sichtbar abgeschwächt.
    total_score = raw_score * (0.60 + 0.40 * coverage / 100)
    confidence = "hoch" if coverage >= 80 else "mittel" if coverage >= 60 else "niedrig"
    component_text = " | ".join(f"{name}: {points:.0f}/{weight:.0f}" for name, points, weight in components)

    return pd.Series({
        "total_score": round(total_score, 1),
        "score_raw": round(raw_score, 1),
        "score_coverage": round(coverage, 1),
        "score_confidence": confidence,
        "score_components": component_text,
    })


def sector_yield_trigger(sector: Any, global_minimum: float) -> float:
    sector_text = str(sector or "")
    return max(global_minimum, YIELD_TRIGGER_BY_SECTOR.get(sector_text, YIELD_TRIGGER_BY_SECTOR["_default"]))


def compute_value_trigger_and_score(
    row: pd.Series,
    drawdown_trigger: float,
    payout_max: float,
    score_min: float,
    yield_min: float,
) -> pd.Series:
    """Berechnet transparenten Value-Score und strengen Watchlist-Trigger."""
    last_price = safe_float(row.get("last_price"))
    high_52w = safe_float(row.get("high_52w"))
    dividend_yield = safe_float(row.get("dividend_yield"))
    payout_ratio = safe_float(row.get("payout_ratio"))
    total_score = safe_float(row.get("total_score"))
    score_coverage = safe_float(row.get("score_coverage"))
    pe = safe_float(row.get("pe_ratio"))
    pb = safe_float(row.get("pb_ratio"))
    debt_to_equity = safe_float(row.get("debt_to_equity"))

    drawdown = None
    if last_price is not None and high_52w not in (None, 0):
        drawdown = (last_price / high_52w - 1.0) * 100

    value_points = 0.0
    value_weight = 0.0

    if drawdown is not None:
        value_weight += 35
        value_points += min(max(-drawdown, 0), 50) / 50 * 35

    if dividend_yield is not None:
        value_weight += 20
        # Bis 8 % steigt der Nutzen. Extremrenditen erhalten keinen Zusatzbonus.
        value_points += min(dividend_yield, 8) / 8 * 20

    if pe is not None:
        value_weight += 20
        if pe <= 0:
            pe_points = 0
        elif pe <= 10:
            pe_points = 20
        elif pe <= 16:
            pe_points = 15
        elif pe <= 25:
            pe_points = 8
        else:
            pe_points = 2
        value_points += pe_points
    elif pb is not None and pb > 0:
        value_weight += 20
        value_points += 18 if pb <= 1 else 13 if pb <= 2 else 7 if pb <= 4 else 2

    if payout_ratio is not None:
        value_weight += 15
        value_points += 15 if 0 <= payout_ratio <= 70 else 10 if payout_ratio <= 90 else 3 if payout_ratio <= 110 else 0

    if debt_to_equity is not None and not is_financial_sector(row.get("sector")):
        value_weight += 10
        value_points += 10 if debt_to_equity < 1 else 7 if debt_to_equity < 2 else 3 if debt_to_equity < 3 else 0

    if value_weight == 0:
        value_score = None
        value_coverage = 0.0
    else:
        raw_value = value_points / value_weight * 100
        value_coverage = min(value_weight / 100 * 100, 100.0)
        value_score = round(raw_value * (0.70 + 0.30 * value_coverage / 100), 1)

    minimum_yield = sector_yield_trigger(row.get("sector"), yield_min)
    checks = {
        "Score": total_score is not None and total_score >= score_min,
        "Datenabdeckung": score_coverage is not None and score_coverage >= MIN_SCORE_COVERAGE_FOR_TRIGGER,
        "Payout": payout_ratio is not None and 0 <= payout_ratio <= payout_max,
        "Rendite": dividend_yield is not None and dividend_yield >= minimum_yield,
        "Drawdown": drawdown is not None and drawdown <= -drawdown_trigger,
    }
    triggered = all(checks.values())

    reasons = [
        f"Score {'✓' if checks['Score'] else '✗'} ({format_number(total_score, 1)} / min. {score_min:.0f})",
        f"Daten {'✓' if checks['Datenabdeckung'] else '✗'} ({format_number(score_coverage, 0)} %)",
        f"Payout {'✓' if checks['Payout'] else '✗'} ({format_percent(payout_ratio, 1)} / max. {payout_max:.0f} %)",
        f"Rendite {'✓' if checks['Rendite'] else '✗'} ({format_percent(dividend_yield, 1)} / min. {minimum_yield:.1f} %)",
        f"Drawdown {'✓' if checks['Drawdown'] else '✗'} ({format_percent(drawdown, 1, signed=True)} / ≤ -{drawdown_trigger:.0f} %)",
    ]

    status = "ausgelöst" if triggered else "Daten unvollständig" if payout_ratio is None else "nicht ausgelöst"
    return pd.Series({
        "drawdown_1y_high_pct": drawdown,
        "value_score": value_score,
        "value_coverage": round(value_coverage, 1),
        "value_trigger": triggered,
        "value_status": status,
        "value_reason": " | ".join(reasons),
    })


def enrich_with_scores(
    metrics: pd.DataFrame,
    drawdown_trigger: float,
    payout_max: float,
    score_min: float,
    yield_min: float,
) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return empty_metrics_frame()
    result = metrics.copy()
    total_score = result.apply(compute_total_score, axis=1)
    result = pd.concat([result, total_score], axis=1)
    value = result.apply(
        lambda row: compute_value_trigger_and_score(
            row,
            drawdown_trigger=drawdown_trigger,
            payout_max=payout_max,
            score_min=score_min,
            yield_min=yield_min,
        ),
        axis=1,
    )
    return pd.concat([result, value], axis=1)


# -----------------------------------------------------------------------------
# Währungen und Portfolio
# -----------------------------------------------------------------------------


def currency_code(currency: Any) -> str:
    value = str(currency or "EUR").strip()
    aliases = {"GBp": "GBP", "GBX": "GBP", "p": "GBP", "USD": "USD", "EUR": "EUR"}
    return aliases.get(value, value.upper())


def currency_unit_multiplier(currency: Any) -> float:
    return 0.01 if str(currency or "").strip() in {"GBp", "GBX", "p"} else 1.0


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fx_to_eur(currency: str) -> Optional[float]:
    """Umrechnung von Kurswährung in EUR via Yahoo-FX-Paaren."""
    raw_currency = str(currency or "EUR").strip()
    code = currency_code(raw_currency)
    multiplier = currency_unit_multiplier(raw_currency)
    if code == "EUR":
        return multiplier

    pair_map = {
        "USD": "EURUSD=X",
        "GBP": "EURGBP=X",
        "CHF": "EURCHF=X",
        "JPY": "EURJPY=X",
        "CAD": "EURCAD=X",
        "AUD": "EURAUD=X",
        "SEK": "EURSEK=X",
        "NOK": "EURNOK=X",
        "DKK": "EURDKK=X",
    }
    pair = pair_map.get(code)
    if not pair:
        return None

    try:
        history = yf.Ticker(pair).history(period="5d", auto_adjust=False)
        close = pd.to_numeric(history.get("Close"), errors="coerce").dropna()
        rate = safe_float(close.iloc[-1]) if not close.empty else None
    except Exception:
        rate = None
    if rate in (None, 0):
        return None
    # EURUSD=X bedeutet USD je EUR. Für USD -> EUR wird invertiert.
    return multiplier / rate


def read_portfolio() -> pd.DataFrame:
    if not PORTFOLIO_PATH.exists():
        return pd.DataFrame(columns=["ticker_yahoo", "shares", "cost_basis", "currency", "purchase_date"])
    try:
        portfolio = pd.read_csv(PORTFOLIO_PATH)
    except Exception as error:
        raise RuntimeError(f"portfolio.csv konnte nicht gelesen werden: {error}") from error

    required = {"ticker_yahoo", "shares", "cost_basis"}
    missing = required - set(portfolio.columns)
    if missing:
        raise ValueError(f"portfolio.csv fehlt: {', '.join(sorted(missing))}")

    portfolio = portfolio.copy()
    portfolio["ticker_yahoo"] = portfolio["ticker_yahoo"].map(clean_ticker)
    portfolio["shares"] = pd.to_numeric(portfolio["shares"], errors="coerce")
    portfolio["cost_basis"] = pd.to_numeric(portfolio["cost_basis"], errors="coerce")
    if "currency" not in portfolio.columns:
        portfolio["currency"] = None
    if "purchase_date" not in portfolio.columns:
        portfolio["purchase_date"] = None
    return portfolio.dropna(subset=["ticker_yahoo", "shares", "cost_basis"])


def build_portfolio_view(metrics: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or portfolio.empty:
        return pd.DataFrame()

    metric_columns = [
        "ticker_yahoo", "name", "sector", "currency", "last_price", "dividend_yield",
        "dividend_per_share", "total_score", "value_score", "value_trigger",
    ]
    available_columns = [column for column in metric_columns if column in metrics.columns]
    result = portfolio.merge(metrics[available_columns], on="ticker_yahoo", how="left", suffixes=("", "_asset"))

    asset_currency = result.get("currency_asset", result.get("currency", "EUR"))
    result["asset_currency"] = asset_currency.fillna("EUR") if isinstance(asset_currency, pd.Series) else "EUR"
    result["cost_currency"] = (
        result["currency"].replace(r"^\s*$", pd.NA, regex=True).fillna(result["asset_currency"])
        if "currency" in result.columns else result["asset_currency"]
    )
    result["fx_asset_to_eur"] = result["asset_currency"].map(fx_to_eur)
    result["fx_cost_to_eur"] = result["cost_currency"].map(fx_to_eur)

    result["market_value_local"] = result["shares"] * pd.to_numeric(result["last_price"], errors="coerce")
    result["market_value_eur"] = result["market_value_local"] * result["fx_asset_to_eur"]
    result["cost_value_eur"] = result["shares"] * result["cost_basis"] * result["fx_cost_to_eur"]
    result["pnl_abs_eur"] = result["market_value_eur"] - result["cost_value_eur"]
    result["pnl_pct"] = result["pnl_abs_eur"] / result["cost_value_eur"] * 100

    result["dividend_income_local"] = result["shares"] * pd.to_numeric(result["dividend_per_share"], errors="coerce")
    result["dividend_income_eur"] = result["dividend_income_local"] * result["fx_asset_to_eur"]
    result["yield_on_cost"] = pd.to_numeric(result["dividend_per_share"], errors="coerce") / result["cost_basis"] * 100

    total_value = result["market_value_eur"].sum(skipna=True)
    result["weight_pct"] = result["market_value_eur"] / total_value * 100 if total_value > 0 else None
    return result


# -----------------------------------------------------------------------------
# Watchlist
# -----------------------------------------------------------------------------


def read_watchlist() -> pd.DataFrame:
    columns = ["ticker_yahoo", "name", "sector", "added_at", "note"]
    if not WATCHLIST_PATH.exists():
        return pd.DataFrame(columns=columns)
    try:
        watchlist = pd.read_csv(WATCHLIST_PATH)
    except Exception:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in watchlist.columns:
            watchlist[column] = ""
    return watchlist[columns]


def add_to_watchlist(ticker: str, name: str, sector: str) -> bool:
    watchlist = read_watchlist()
    ticker = clean_ticker(ticker)
    if ticker in watchlist["ticker_yahoo"].astype(str).tolist():
        return False
    new_row = pd.DataFrame([{
        "ticker_yahoo": ticker,
        "name": name,
        "sector": sector,
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note": "",
    }])
    pd.concat([watchlist, new_row], ignore_index=True).to_csv(WATCHLIST_PATH, index=False)
    return True


def remove_from_watchlist(ticker: str) -> None:
    watchlist = read_watchlist()
    watchlist = watchlist[watchlist["ticker_yahoo"].astype(str) != clean_ticker(ticker)]
    watchlist.to_csv(WATCHLIST_PATH, index=False)


# -----------------------------------------------------------------------------
# News und Sentiment
# -----------------------------------------------------------------------------


def company_aliases(company_name: str, ticker: str) -> list[str]:
    """Wenige, möglichst präzise Suchbegriffe statt riskanter Einzelwortsuche."""
    normalized = str(company_name or "").lower().strip()
    normalized = re.sub(r"\b(ag|se|sa|s\.a\.|plc|n\.v\.|gmbh|kgaa|inc\.?|corp\.?|ltd\.?)$", "", normalized).strip()
    aliases = {normalized, clean_ticker(ticker).split(".")[0].lower()}
    # Nur lange Bestandteile aufnehmen; kurze Wortteile führen zu vielen False Positives.
    aliases.update(part for part in normalized.split() if len(part) >= 5)
    return sorted(alias for alias in aliases if len(alias) >= 3)


def contains_alias(text: str, aliases: list[str]) -> bool:
    lower_text = text.lower()
    return any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lower_text) for alias in aliases)


def simple_sentiment(text: str) -> tuple[int, str]:
    lower_text = text.lower()
    positive = sum(1 for word in POSITIVE_WORDS if word in lower_text)
    negative = sum(1 for word in NEGATIVE_WORDS if word in lower_text)
    score = positive - negative
    label = "positiv" if score > 0 else "negativ" if score < 0 else "neutral"
    return score, label


def entry_datetime(entry: Any) -> Optional[datetime]:
    raw = entry.get("published") or entry.get("updated")
    if raw:
        try:
            parsed = dateparser.parse(raw)
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except Exception:
            pass
    for field in ("published_parsed", "updated_parsed"):
        parsed_struct = entry.get(field)
        if parsed_struct:
            try:
                return datetime(
                    parsed_struct.tm_year, parsed_struct.tm_mon, parsed_struct.tm_mday,
                    parsed_struct.tm_hour, parsed_struct.tm_min, parsed_struct.tm_sec,
                    tzinfo=timezone.utc,
                )
            except Exception:
                continue
    return None


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_news_for_ticker(ticker: str, company_name: str, days_back: int, max_items: int = 50) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(days_back))
    aliases = company_aliases(company_name, ticker)
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            source_name = getattr(feed, "feed", {}).get("title", url)
        except Exception:
            continue

        for entry in getattr(feed, "entries", []):
            title = str(entry.get("title", "") or "")
            summary = str(entry.get("summary", "") or "")
            link = str(entry.get("link", "") or "")
            published = entry_datetime(entry)
            if not title or published is None or published < cutoff:
                continue

            text = f"{title} {summary}"
            if not contains_alias(text, aliases):
                continue
            dedupe_key = (re.sub(r"\W+", "", title.lower()), link.split("?")[0])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            score, label = simple_sentiment(text)
            entries.append({
                "published": published.replace(tzinfo=None),
                "title": title,
                "link": link,
                "source": source_name,
                "sentiment_score": score,
                "sentiment_label": label,
            })

    if not entries:
        return pd.DataFrame(columns=["published", "title", "link", "source", "sentiment_score", "sentiment_label"])
    return pd.DataFrame(entries).sort_values("published", ascending=False).head(max_items).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Charts und Tabellendarstellung
# -----------------------------------------------------------------------------


def colorize_change(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return "color: #16a34a; font-weight: 600" if number > 0 else "color: #dc2626; font-weight: 600" if number < 0 else ""


def colorize_score(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return "color: #16a34a; font-weight: 700" if number >= 70 else "color: #d97706; font-weight: 700" if number >= 45 else "color: #dc2626; font-weight: 700"


def colorize_value_trigger(value: Any) -> str:
    if value is True:
        return "background-color: #dcfce7; color: #166534; font-weight: 700"
    if value is False:
        return "color: #6b7280"
    return ""


def make_price_chart(
    history: pd.DataFrame,
    period_label: str,
    show_smas: bool,
    news: Optional[pd.DataFrame] = None,
) -> Optional[alt.Chart]:
    if history is None or history.empty or "Close" not in history.columns:
        return None

    frame = history.copy()
    frame = ensure_datetime_index(frame)
    frame["Kurs"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Kurs"])
    if frame.empty:
        return None

    # WICHTIG: SMA zuerst auf vollständiger Historie berechnen, dann Zeitraum filtern.
    frame["SMA 20"] = frame["Kurs"].rolling(20, min_periods=20).mean()
    frame["SMA 50"] = frame["Kurs"].rolling(50, min_periods=50).mean()
    frame["SMA 200"] = frame["Kurs"].rolling(200, min_periods=200).mean()

    days_by_period = {"2 Monate": 62, "6 Monate": 135, "1 Jahr": 260, "5 Jahre": 2000}
    visible = frame.tail(days_by_period.get(period_label, 260)).copy().reset_index()
    visible = visible.rename(columns={visible.columns[0]: "Datum"})

    base = alt.Chart(visible).mark_line(strokeWidth=2).encode(
        x=alt.X("Datum:T", title="Datum"),
        y=alt.Y("Kurs:Q", title="Kurs", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Datum:T", format="%d.%m.%Y"), alt.Tooltip("Kurs:Q", format=".2f")],
    )
    layers: list[alt.Chart] = [base]

    if show_smas:
        for sma_name in SMA_COLORS:
            sma_frame = visible[["Datum", sma_name]].dropna().copy()
            if sma_frame.empty:
                continue
            sma_frame["Linie"] = sma_name
            layers.append(
                alt.Chart(sma_frame).mark_line(strokeWidth=1.7).encode(
                    x="Datum:T",
                    y=alt.Y(f"{sma_name}:Q", title="Kurs", scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "Linie:N",
                        scale=alt.Scale(domain=list(SMA_COLORS), range=list(SMA_COLORS.values())),
                        legend=alt.Legend(title="Durchschnitte", orient="bottom"),
                    ),
                    tooltip=[alt.Tooltip("Datum:T", format="%d.%m.%Y"), alt.Tooltip(f"{sma_name}:Q", format=".2f")],
                )
            )

    if news is not None and not news.empty and "published" in news.columns:
        event_frame = news.copy()
        event_frame["Datum"] = pd.to_datetime(event_frame["published"], errors="coerce").dt.normalize()
        event_frame = event_frame.dropna(subset=["Datum"])
        event_frame = event_frame[event_frame["Datum"].between(visible["Datum"].min(), visible["Datum"].max())]
        if not event_frame.empty:
            price_lookup = visible[["Datum", "Kurs"]].sort_values("Datum")
            event_frame = pd.merge_asof(
                event_frame.sort_values("Datum"), price_lookup, on="Datum", direction="nearest"
            )
            layers.append(
                alt.Chart(event_frame).mark_point(filled=True, size=65).encode(
                    x="Datum:T",
                    y=alt.Y("Kurs:Q", scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "sentiment_label:N",
                        scale=alt.Scale(domain=["positiv", "neutral", "negativ"], range=["#16a34a", "#64748b", "#dc2626"]),
                        legend=alt.Legend(title="News", orient="bottom"),
                    ),
                    tooltip=[
                        alt.Tooltip("Datum:T", format="%d.%m.%Y"),
                        "title:N", "source:N", "sentiment_label:N", "link:N",
                    ],
                )
            )

    return alt.layer(*layers).resolve_scale(y="shared").properties(height=360)


def overview_styler(df: pd.DataFrame) -> Any:
    formats = {
        "last_price": lambda value: format_number(value, 2),
        "change_1d": lambda value: format_percent(value, 2, signed=True),
        "change_5d": lambda value: format_percent(value, 2, signed=True),
        "change_1y": lambda value: format_percent(value, 2, signed=True),
        "vol_1y": lambda value: format_percent(value, 1),
        "total_score": lambda value: format_number(value, 1),
        "score_coverage": lambda value: format_percent(value, 0),
        "value_score": lambda value: format_number(value, 1),
        "drawdown_1y_high_pct": lambda value: format_percent(value, 1, signed=True),
        "dividend_yield": lambda value: format_percent(value, 2),
    }
    return (
        df.style.format({key: value for key, value in formats.items() if key in df.columns}, na_rep="–")
        .map(colorize_change, subset=[column for column in ["change_1d", "change_5d", "change_1y"] if column in df.columns])
        .map(colorize_score, subset=[column for column in ["total_score"] if column in df.columns])
        .map(colorize_value_trigger, subset=[column for column in ["value_trigger"] if column in df.columns])
    )


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------


def show_header() -> None:
    st.title("Aktien Explorer")
    st.caption(
        f"Version {APP_VERSION} · Fundamentaldaten, Value-Scanner, Portfolio und News. "
        "Keine Anlageberatung – Daten vor Entscheidungen immer selbst prüfen."
    )


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Überblick")
    if df.empty:
        st.info("Keine Daten für die aktuelle Filtereinstellung.")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Unternehmen", f"{len(df):,}".replace(",", "."))
    metric_columns[1].metric("Trigger", int(df["value_trigger"].fillna(False).sum()))
    metric_columns[2].metric("Ø Qualitäts-Score", format_number(df["total_score"].mean(), 1))
    metric_columns[3].metric("Ø 1J-Performance", format_percent(df["change_1y"].mean(), 1, signed=True))

    columns = [
        "name", "ticker_yahoo", "sector", "currency", "last_price", "change_1d", "change_5d",
        "change_1y", "vol_1y", "dividend_yield", "total_score", "score_coverage", "value_score",
        "drawdown_1y_high_pct", "value_trigger",
    ]
    visible_columns = [column for column in columns if column in df.columns]
    st.dataframe(overview_styler(df[visible_columns]), use_container_width=True, hide_index=True)


def render_fundamentals(df: pd.DataFrame) -> None:
    st.subheader("Fundamentaldaten")
    columns = [
        "name", "ticker_yahoo", "currency", "market_cap", "pe_ratio", "forward_pe", "pb_ratio",
        "ps_ratio", "ev_ebitda", "net_margin", "operating_margin", "roe", "roa", "dividend_yield",
        "dividend_per_share", "payout_ratio", "dividend_growth_5y", "dividend_frequency",
        "debt_to_equity", "net_debt_ebitda", "score_raw", "score_coverage", "total_score",
    ]
    visible_columns = [column for column in columns if column in df.columns]

    # Nur für die Anzeige: interne Spaltennamen bleiben im restlichen Code unverändert.
    display_names = {
        "name": "Unternehmen",
        "ticker_yahoo": "Ticker",
        "currency": "Währung",
        "market_cap": "Marktkapitalisierung",
        "pe_ratio": "KGV",
        "forward_pe": "Forward KGV",
        "pb_ratio": "KBV",
        "ps_ratio": "KUV",
        "ev_ebitda": "EV/EBITDA",
        "net_margin": "Nettomarge",
        "operating_margin": "Operative Marge",
        "roe": "Eigenkapitalrendite",
        "roa": "Gesamtkapitalrendite",
        "dividend_yield": "Dividendenrendite",
        "dividend_per_share": "Dividende je Aktie",
        "payout_ratio": "Ausschüttungsquote",
        "dividend_growth_5y": "Dividendenwachstum 5J",
        "dividend_frequency": "Ausschüttungsrhythmus",
        "debt_to_equity": "Debt/Equity",
        "net_debt_ebitda": "Netto-Schulden/EBITDA",
        "score_raw": "Rohscore",
        "score_coverage": "Datenabdeckung",
        "total_score": "Qualitäts-Score",
    }

    display_df = df[visible_columns].rename(columns=display_names)
    formats = {
        "Marktkapitalisierung": human_market_cap,
        "KGV": lambda value: format_number(value, 2),
        "Forward KGV": lambda value: format_number(value, 2),
        "KBV": lambda value: format_number(value, 2),
        "KUV": lambda value: format_number(value, 2),
        "EV/EBITDA": lambda value: format_number(value, 2),
        "Nettomarge": lambda value: format_percent(value, 2),
        "Operative Marge": lambda value: format_percent(value, 2),
        "Eigenkapitalrendite": lambda value: format_percent(value, 2),
        "Gesamtkapitalrendite": lambda value: format_percent(value, 2),
        "Dividendenrendite": lambda value: format_percent(value, 2),
        "Dividende je Aktie": lambda value: format_number(value, 2),
        "Ausschüttungsquote": lambda value: format_percent(value, 2),
        "Dividendenwachstum 5J": lambda value: format_percent(value, 2),
        "Debt/Equity": lambda value: f"{format_number(value, 2)}x",
        "Netto-Schulden/EBITDA": lambda value: f"{format_number(value, 2)}x",
        "Rohscore": lambda value: format_number(value, 1),
        "Datenabdeckung": lambda value: format_percent(value, 0),
        "Qualitäts-Score": lambda value: format_number(value, 1),
    }
    styled = (
        display_df.style.format(formats, na_rep="–")
        .map(colorize_score, subset=["Qualitäts-Score"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "Die Marktkapitalisierung wird kompakt in Mio., Mrd. oder Bio. angezeigt und "
        "bezieht sich auf die jeweilige Originalwährung in der Spalte „Währung“. "
        "Score und Value-Score sind heuristische Recherchehilfen."
    )


def render_value_watchlist(df: pd.DataFrame) -> None:
    st.subheader("Dividenden-Value-Scanner")
    st.caption("Ein Trigger verlangt ausreichend Score-Daten, kontrollierte Ausschüttungsquote, Mindest-Rendite und klaren Drawdown.")
    columns = [
        "name", "ticker_yahoo", "sector", "total_score", "score_coverage", "value_score", "value_coverage",
        "value_status", "value_trigger", "dividend_yield", "payout_ratio", "pe_ratio", "pb_ratio",
        "drawdown_1y_high_pct", "value_reason",
    ]
    visible_columns = [column for column in columns if column in df.columns]
    result = df.sort_values(["value_trigger", "value_score"], ascending=[False, False], na_position="last")
    formats = {
        "total_score": lambda value: format_number(value, 1),
        "score_coverage": lambda value: format_percent(value, 0),
        "value_score": lambda value: format_number(value, 1),
        "value_coverage": lambda value: format_percent(value, 0),
        "dividend_yield": lambda value: format_percent(value, 2),
        "payout_ratio": lambda value: format_percent(value, 2),
        "pe_ratio": lambda value: format_number(value, 2),
        "pb_ratio": lambda value: format_number(value, 2),
        "drawdown_1y_high_pct": lambda value: format_percent(value, 1, signed=True),
    }
    styled = (
        result[visible_columns].style.format({key: value for key, value in formats.items() if key in visible_columns}, na_rep="–")
        .map(colorize_score, subset=["total_score"])
        .map(colorize_value_trigger, subset=["value_trigger"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_risk_and_chart(df: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> None:
    st.subheader("Einzelanalyse & Chart")
    ticker = st.selectbox("Aktie auswählen", df["ticker_yahoo"].tolist(), key="analysis_ticker")
    row = df.loc[df["ticker_yahoo"] == ticker].iloc[0]

    columns = st.columns(5)
    columns[0].metric("Letzter Kurs", f"{format_number(row.get('last_price'), 2)} {row.get('currency', '')}")
    columns[1].metric("Volatilität 1J", format_percent(row.get("vol_1y"), 1))
    columns[2].metric("Qualitäts-Score", format_number(row.get("total_score"), 1))
    columns[3].metric("Datenabdeckung", format_percent(row.get("score_coverage"), 0))
    columns[4].metric("Value-Score", format_number(row.get("value_score"), 1))

    left, right = st.columns([2, 1])
    with right:
        st.markdown("**Score-Komponenten**")
        st.caption(str(row.get("score_components", "–")))
        st.markdown("**Value-Check**")
        st.caption(str(row.get("value_reason", "–")))
        if st.button("Zur Watchlist hinzufügen", key=f"watch_{ticker}"):
            if add_to_watchlist(ticker, str(row.get("name", ticker)), str(row.get("sector", "Unbekannt"))):
                st.success(f"{ticker} wurde gespeichert.")
            else:
                st.info(f"{ticker} ist bereits auf der Watchlist.")

    with left:
        period = st.selectbox("Chart-Zeitraum", ["2 Monate", "6 Monate", "1 Jahr", "5 Jahre"], index=2)
        show_smas = st.checkbox("SMA 20 / 50 / 200 anzeigen", value=True)
        news_key = f"news::{ticker}"
        news = st.session_state.get(news_key)
        chart = make_price_chart(histories.get(ticker, pd.DataFrame()), period, show_smas, news)
        if chart is None:
            st.info("Für diese Aktie sind keine Kursdaten verfügbar.")
        else:
            st.altair_chart(chart, use_container_width=True)
            if news is not None and not news.empty:
                st.caption("Punkte im Chart stehen für die zuletzt geladenen News zu dieser Aktie.")


def render_sector_view(df: pd.DataFrame) -> None:
    st.subheader("Sektor-Übersicht")
    sector_stats = (
        df.groupby("sector", dropna=False)
        .agg(
            Unternehmen=("ticker_yahoo", "count"),
            Performance_1J=("change_1y", "mean"),
            Qualitäts_Score=("total_score", "mean"),
            Value_Score=("value_score", "mean"),
            Trigger=("value_trigger", lambda values: int(pd.Series(values).fillna(False).sum())),
        )
        .reset_index()
        .sort_values("Value_Score", ascending=False, na_position="last")
    )
    st.dataframe(
        sector_stats.style.format(
            {
                "Performance_1J": lambda value: format_percent(value, 2, signed=True),
                "Qualitäts_Score": lambda value: format_number(value, 1),
                "Value_Score": lambda value: format_number(value, 1),
            },
            na_rep="–",
        ),
        use_container_width=True,
        hide_index=True,
    )
    chart = alt.Chart(sector_stats.dropna(subset=["Performance_1J"])).mark_bar().encode(
        x=alt.X("Performance_1J:Q", title="Ø Performance 1 Jahr (%)"),
        y=alt.Y("sector:N", sort="-x", title="Sektor"),
        tooltip=["sector:N", "Unternehmen:Q", alt.Tooltip("Performance_1J:Q", format=".2f")],
    ).properties(height=max(220, 30 * len(sector_stats)))
    st.altair_chart(chart, use_container_width=True)


def render_news(df: pd.DataFrame) -> None:
    st.subheader("News & Ereignisse")
    st.caption("RSS-News werden per Unternehmensalias zugeordnet. Das Sentiment ist eine einfache deutsch/englische Heuristik, kein Signal.")
    ticker = st.selectbox("Aktie für News", df["ticker_yahoo"].tolist(), key="news_ticker")
    days_back = st.slider("Zeitraum", min_value=3, max_value=90, value=30, step=1, key="news_days")
    row = df.loc[df["ticker_yahoo"] == ticker].iloc[0]
    key = f"news::{ticker}"

    if st.button("News aktualisieren", key=f"refresh_news_{ticker}"):
        with st.spinner("RSS-Feeds werden durchsucht …"):
            st.session_state[key] = fetch_news_for_ticker(
                ticker=ticker,
                company_name=str(row.get("name", ticker)),
                days_back=days_back,
            )

    news = st.session_state.get(key)
    if news is None:
        st.info("Noch keine News geladen.")
    elif news.empty:
        st.info("In den hinterlegten RSS-Feeds wurden keine passenden aktuellen Einträge gefunden.")
    else:
        show_columns = ["published", "title", "source", "sentiment_label", "sentiment_score", "link"]
        st.dataframe(news[show_columns], use_container_width=True, hide_index=True)
        st.download_button(
            "News als CSV herunterladen",
            data=news.to_csv(index=False).encode("utf-8"),
            file_name=f"news_{ticker.lower()}.csv",
            mime="text/csv",
            key=f"download_news_{ticker}",
        )


def render_portfolio(metrics: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> None:
    st.subheader("Portfolio")
    st.caption("portfolio.csv: ticker_yahoo, shares, cost_basis, currency (optional), purchase_date (optional)")
    try:
        portfolio = read_portfolio()
    except Exception as error:
        st.error(str(error))
        return

    if portfolio.empty:
        st.info("Keine portfolio.csv gefunden. Lege die Datei neben app.py an und nutze die Beispielstruktur aus dem Kopf des Codes.")
        return

    portfolio_tickers = set(portfolio["ticker_yahoo"].map(clean_ticker))
    known_tickers = set(metrics["ticker_yahoo"].map(clean_ticker))
    missing_tickers = sorted(portfolio_tickers - known_tickers)

    if missing_tickers:
        st.warning("Diese Portfolio-Ticker liegen außerhalb der aktuellen Indexanalyse: " + ", ".join(missing_tickers))
        if st.button("Fehlende Portfolio-Ticker laden", key="load_portfolio_extras"):
            companies = pd.DataFrame({
                "name": missing_tickers,
                "ticker_yahoo": missing_tickers,
                "sector": "Unbekannt",
            })
            with st.spinner("Zusätzliche Portfolio-Daten werden geladen …"):
                extra_metrics, extra_histories, extra_errors = collect_metrics(companies)
            st.session_state["portfolio_extra_metrics"] = extra_metrics
            st.session_state["portfolio_extra_histories"] = extra_histories
            if extra_errors:
                st.warning(" | ".join(extra_errors[:10]))
            st.rerun()

    extras = st.session_state.get("portfolio_extra_metrics", empty_metrics_frame())
    combined = pd.concat([metrics, extras], ignore_index=True).drop_duplicates(subset=["ticker_yahoo"], keep="first")
    portfolio_view = build_portfolio_view(combined, portfolio)

    total_value = portfolio_view["market_value_eur"].sum(skipna=True)
    total_cost = portfolio_view["cost_value_eur"].sum(skipna=True)
    total_pnl = portfolio_view["pnl_abs_eur"].sum(skipna=True)
    total_income = portfolio_view["dividend_income_eur"].sum(skipna=True)

    columns = st.columns(4)
    columns[0].metric("Marktwert", format_eur(total_value, 0))
    columns[1].metric("Einstandswert", format_eur(total_cost, 0))
    columns[2].metric(
        "Gewinn / Verlust",
        format_eur(total_pnl, 0, signed=True),
        delta=format_percent(total_pnl / total_cost * 100 if total_cost else None, 1, signed=True),
    )
    columns[3].metric("Dividende p.a. (Schätzung)", format_eur(total_income, 0))

    visible_columns = [
        "ticker_yahoo", "name", "shares", "cost_basis", "cost_currency", "asset_currency", "last_price",
        "market_value_eur", "cost_value_eur", "pnl_abs_eur", "pnl_pct", "weight_pct", "dividend_income_eur",
        "yield_on_cost", "total_score", "value_score", "value_trigger",
    ]
    visible_columns = [column for column in visible_columns if column in portfolio_view.columns]
    formats = {
        "shares": lambda value: format_number(value, 4),
        "cost_basis": lambda value: format_number(value, 2),
        "last_price": lambda value: format_number(value, 2),
        "market_value_eur": lambda value: format_eur(value, 2),
        "cost_value_eur": lambda value: format_eur(value, 2),
        "pnl_abs_eur": lambda value: format_eur(value, 2, signed=True),
        "pnl_pct": lambda value: format_percent(value, 2, signed=True),
        "weight_pct": lambda value: format_percent(value, 2),
        "dividend_income_eur": lambda value: format_eur(value, 2),
        "yield_on_cost": lambda value: format_percent(value, 2),
        "total_score": lambda value: format_number(value, 1),
        "value_score": lambda value: format_number(value, 1),
    }
    styled = (
        portfolio_view[visible_columns].style.format({key: value for key, value in formats.items() if key in visible_columns}, na_rep="–")
        .map(colorize_change, subset=[column for column in ["pnl_abs_eur", "pnl_pct"] if column in visible_columns])
        .map(colorize_score, subset=[column for column in ["total_score"] if column in visible_columns])
        .map(colorize_value_trigger, subset=[column for column in ["value_trigger"] if column in visible_columns])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("EUR-Werte verwenden aktuelle Yahoo-FX-Kurse. Fehlt eine Umrechnung, bleibt die Position teilweise leer.")


def render_watchlist(metrics: pd.DataFrame) -> None:
    st.subheader("Eigene Watchlist")
    watchlist = read_watchlist()
    if watchlist.empty:
        st.info("Noch keine Titel gespeichert. Nutze in der Einzelanalyse den Button „Zur Watchlist hinzufügen“.")
        return

    merged = watchlist.merge(metrics, on="ticker_yahoo", how="left", suffixes=("", "_market"))
    visible_columns = [
        "ticker_yahoo", "name", "sector", "added_at", "note", "last_price", "currency", "change_1y",
        "dividend_yield", "total_score", "value_score", "value_trigger",
    ]
    visible_columns = [column for column in visible_columns if column in merged.columns]
    st.dataframe(overview_styler(merged[visible_columns]), use_container_width=True, hide_index=True)

    ticker_to_remove = st.selectbox("Titel von Watchlist entfernen", merged["ticker_yahoo"].tolist(), key="remove_watchlist_ticker")
    if st.button("Aus Watchlist entfernen", key="remove_watchlist_button"):
        remove_from_watchlist(ticker_to_remove)
        st.success(f"{ticker_to_remove} wurde entfernt.")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    show_header()

    # ----- Sidebar: Daten, Filter und Scanner-Parameter -----
    with st.sidebar:
        st.header("Analyse-Einstellungen")
        index_name = st.selectbox("Index", ["DAX 40", "S&P 500"])
        st.caption("Tipp: Lege optional data/indices/dax40.csv oder sp500.csv an, um Wikipedia als Quelle zu ersetzen.")

        try:
            constituents = load_index_constituents(index_name)
        except Exception as error:
            st.error(f"Index konnte nicht geladen werden: {error}")
            st.stop()

        sector_options = ["Alle"] + sorted(constituents["sector"].dropna().astype(str).unique().tolist())
        selected_sector = st.selectbox("Sektor", sector_options)
        query = st.text_input("Name oder Ticker suchen").strip()

        filtered_constituents = constituents.copy()
        if selected_sector != "Alle":
            filtered_constituents = filtered_constituents[filtered_constituents["sector"] == selected_sector]
        if query:
            mask = (
                filtered_constituents["name"].astype(str).str.contains(query, case=False, na=False)
                | filtered_constituents["ticker_yahoo"].astype(str).str.contains(query, case=False, na=False)
            )
            filtered_constituents = filtered_constituents[mask]

        maximum = min(150 if index_name == "S&P 500" else len(filtered_constituents), len(filtered_constituents))
        if maximum <= 0:
            st.error("Der Filter enthält keine Unternehmen.")
            st.stop()
        default_count = min(40 if index_name == "S&P 500" else 40, maximum)
        max_stocks = st.slider("Max. Unternehmen laden", min_value=1, max_value=maximum, value=default_count, step=1)

        st.divider()
        st.header("Dividenden-Value-Scanner")
        drawdown_trigger = st.slider("Min. Drawdown vom 52W-Hoch", 10, 60, int(DEFAULT_DRAWDOWN_TRIGGER), 5)
        payout_max = st.slider("Max. Payout Ratio", 40, 120, int(DEFAULT_PAYOUT_MAX), 5)
        score_min = st.slider("Min. Qualitäts-Score", 0, 100, int(DEFAULT_SCORE_MIN), 5)
        yield_min = st.slider("Min. Dividendenrendite", 1.0, 10.0, float(DEFAULT_YIELD_MIN), 0.5)

        st.divider()
        reload_clicked = st.button("Daten laden / aktualisieren", type="primary", use_container_width=True)
        if st.button("Zwischenspeicher leeren", use_container_width=True):
            load_index_constituents.clear()
            download_price_histories.clear()
            fetch_ticker_info.clear()
            fetch_dividends.clear()
            fetch_news_for_ticker.clear()
            fx_to_eur.clear()
            st.session_state.pop("metrics_raw", None)
            st.session_state.pop("histories", None)
            st.success("Zwischenspeicher wurde geleert.")
            st.rerun()

    selected_constituents = filtered_constituents.head(max_stocks).reset_index(drop=True)
    selected_tickers = tuple(selected_constituents["ticker_yahoo"].map(clean_ticker))

    # Rohdaten sind unabhängig von den Sliderwerten. Die Score-Berechnung läuft
    # bei jeder UI-Änderung neu und braucht keinen erneuten Datenabruf.
    loaded_tickers = tuple(st.session_state.get("loaded_tickers", ()))
    if reload_clicked or loaded_tickers != selected_tickers:
        with st.spinner(f"Lade Daten für {len(selected_constituents)} Unternehmen …"):
            raw_metrics, histories, errors = collect_metrics(selected_constituents)
        st.session_state["metrics_raw"] = raw_metrics
        st.session_state["histories"] = histories
        st.session_state["loaded_tickers"] = selected_tickers
        st.session_state["last_refresh"] = datetime.now().strftime("%d.%m.%Y %H:%M")
        if errors:
            st.warning("Einige Daten konnten nicht vollständig geladen werden: " + " | ".join(errors[:8]))

    raw_metrics = st.session_state.get("metrics_raw")
    histories = st.session_state.get("histories", {})

    if raw_metrics is None or raw_metrics.empty:
        st.info("Wähle links einen Index und klicke auf „Daten laden / aktualisieren“.")
        st.stop()

    data = enrich_with_scores(
        raw_metrics,
        drawdown_trigger=float(drawdown_trigger),
        payout_max=float(payout_max),
        score_min=float(score_min),
        yield_min=float(yield_min),
    )

    last_refresh = st.session_state.get("last_refresh", "–")
    st.caption(f"Aktualisiert: {last_refresh} · {len(data)} Unternehmen analysiert · Basiswährung Portfolio: {BASE_CURRENCY}")

    tabs = st.tabs([
        "Überblick", "Fundamentaldaten", "Einzelanalyse", "Sektoren", "News", "Portfolio", "Watchlist", "Value-Scanner",
    ])
    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_fundamentals(data)
    with tabs[2]:
        render_risk_and_chart(data, histories)
    with tabs[3]:
        render_sector_view(data)
    with tabs[4]:
        render_news(data)
    with tabs[5]:
        render_portfolio(data, histories)
    with tabs[6]:
        render_watchlist(data)
    with tabs[7]:
        render_value_watchlist(data)


if __name__ == "__main__":
    main()