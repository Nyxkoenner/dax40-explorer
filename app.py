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

import os
import logging
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote_plus

import altair as alt
import feedparser
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
from dateutil import parser as dateparser


# -----------------------------------------------------------------------------
# App-Konfiguration
# -----------------------------------------------------------------------------

APP_VERSION = "4.0"
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

INDEX_OPTIONS = ["DAX 40", "MDAX", "SDAX", "S&P 500"]

INDEX_LOCAL_FILES = {
    "DAX 40": INDEX_DIR / "dax40.csv",
    "MDAX": INDEX_DIR / "mdax.csv",
    "SDAX": INDEX_DIR / "sdax.csv",
    "S&P 500": INDEX_DIR / "sp500.csv",
}

# Robuster Fallback für DAX 40, falls die Webtabelle unvollständig oder
# uneinheitlich geparst wird. Stand: Wikipedia-Liste „as of 22 September 2025“.
# Für maximale Kontrolle kann die Liste jederzeit über data/indices/dax40.csv überschrieben werden.
DAX40_STATIC_CONSTITUENTS = [
    {"name": "Adidas", "ticker_yahoo": "ADS.DE", "sector": "Apparel"},
    {"name": "Airbus", "ticker_yahoo": "AIR.PA", "sector": "Aerospace & Defence"},
    {"name": "Allianz", "ticker_yahoo": "ALV.DE", "sector": "Financial Services"},
    {"name": "BASF", "ticker_yahoo": "BAS.DE", "sector": "Chemicals"},
    {"name": "Bayer", "ticker_yahoo": "BAYN.DE", "sector": "Pharmaceuticals"},
    {"name": "Beiersdorf", "ticker_yahoo": "BEI.DE", "sector": "Consumer Goods"},
    {"name": "BMW", "ticker_yahoo": "BMW.DE", "sector": "Automotive"},
    {"name": "Brenntag", "ticker_yahoo": "BNR.DE", "sector": "Distribution"},
    {"name": "Commerzbank", "ticker_yahoo": "CBK.DE", "sector": "Financial Services"},
    {"name": "Continental", "ticker_yahoo": "CON.DE", "sector": "Automotive"},
    {"name": "Daimler Truck", "ticker_yahoo": "DTG.DE", "sector": "Automotive"},
    {"name": "Deutsche Bank", "ticker_yahoo": "DBK.DE", "sector": "Financial Services"},
    {"name": "Deutsche Börse", "ticker_yahoo": "DB1.DE", "sector": "Financial Services"},
    {"name": "Deutsche Post", "ticker_yahoo": "DHL.DE", "sector": "Logistics"},
    {"name": "Deutsche Telekom", "ticker_yahoo": "DTE.DE", "sector": "Telecommunication"},
    {"name": "E.ON", "ticker_yahoo": "EOAN.DE", "sector": "Utilities"},
    {"name": "Fresenius", "ticker_yahoo": "FRE.DE", "sector": "Healthcare"},
    {"name": "Fresenius Medical Care", "ticker_yahoo": "FME.DE", "sector": "Healthcare"},
    {"name": "GEA Group", "ticker_yahoo": "G1A.DE", "sector": "Mechanical Engineering"},
    {"name": "Hannover Re", "ticker_yahoo": "HNR1.DE", "sector": "Insurance"},
    {"name": "Heidelberg Materials", "ticker_yahoo": "HEI.DE", "sector": "Construction Materials"},
    {"name": "Henkel", "ticker_yahoo": "HEN3.DE", "sector": "Consumer Goods"},
    {"name": "Infineon Technologies", "ticker_yahoo": "IFX.DE", "sector": "Technology"},
    {"name": "Mercedes-Benz Group", "ticker_yahoo": "MBG.DE", "sector": "Automotive"},
    {"name": "Merck", "ticker_yahoo": "MRK.DE", "sector": "Pharmaceuticals"},
    {"name": "MTU Aero Engines", "ticker_yahoo": "MTX.DE", "sector": "Aerospace & Defence"},
    {"name": "Munich Re", "ticker_yahoo": "MUV2.DE", "sector": "Financial Services"},
    {"name": "Porsche SE", "ticker_yahoo": "PAH3.DE", "sector": "Automotive"},
    {"name": "Qiagen", "ticker_yahoo": "QIA.DE", "sector": "Biotech"},
    {"name": "Rheinmetall", "ticker_yahoo": "RHM.DE", "sector": "Aerospace & Defence"},
    {"name": "RWE", "ticker_yahoo": "RWE.DE", "sector": "Utilities"},
    {"name": "SAP", "ticker_yahoo": "SAP.DE", "sector": "Technology"},
    {"name": "Scout24", "ticker_yahoo": "G24.DE", "sector": "E-Commerce"},
    {"name": "Siemens", "ticker_yahoo": "SIE.DE", "sector": "Industrials"},
    {"name": "Siemens Energy", "ticker_yahoo": "ENR.DE", "sector": "Energy technology"},
    {"name": "Siemens Healthineers", "ticker_yahoo": "SHL.DE", "sector": "Medical Equipment"},
    {"name": "Symrise", "ticker_yahoo": "SY1.DE", "sector": "Chemicals"},
    {"name": "Volkswagen Group", "ticker_yahoo": "VOW3.DE", "sector": "Automotive"},
    {"name": "Vonovia", "ticker_yahoo": "VNA.DE", "sector": "Real Estate"},
    {"name": "Zalando", "ticker_yahoo": "ZAL.DE", "sector": "E-Commerce"},
]

GERMAN_INDEX_SOURCES = {
    "DAX 40": {
        "urls": ["https://en.wikipedia.org/wiki/DAX", "https://de.wikipedia.org/wiki/DAX"],
        "min_rows": 30,
    },
    "MDAX": {
        "urls": ["https://en.wikipedia.org/wiki/MDAX", "https://de.wikipedia.org/wiki/MDAX"],
        "min_rows": 40,
    },
    "SDAX": {
        "urls": ["https://en.wikipedia.org/wiki/SDAX", "https://de.wikipedia.org/wiki/SDAX"],
        "min_rows": 50,
    },
}

BENCHMARK_BY_INDEX = {
    "DAX 40": "^GDAXI",
    "MDAX": "^MDAXI",
    "SDAX": "^SDAXI",
    "S&P 500": "^GSPC",
}

# Deutsche Indizes enthalten nicht nur Xetra-/Frankfurt-Ticker.
# Beispiel: Airbus ist im DAX, wird bei Yahoo aber zuverlässig als AIR.PA geführt.
# V3.8 hat solche Werte versehentlich herausgefiltert und dadurch beim DAX nur 39 Titel geladen.
GERMAN_INDEX_ALLOWED_SUFFIXES = (".DE", ".F", ".PA")


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
    """Normalisiert Yahoo-Ticker und entfernt Artefakte aus Webtabellen.

    Wikipedia/Finanzseiten enthalten teils Fußnoten, Dollarzeichen oder
    unsichtbare Leerzeichen. Diese Artefakte führten beim SDAX-Fallback zu
    Symbolen wie $SRV.DE.
    """
    value = str(ticker or "").strip().upper()
    value = re.sub(r"\[.*?\]", "", value)
    value = value.replace("\xa0", " ").replace("–", "-").replace("—", "-")
    value = value.strip().lstrip("$")
    value = re.sub(r"\s+", "", value)
    # Für Yahoo sind Buchstaben, Zahlen, Punkt und Bindestrich relevant.
    value = re.sub(r"[^A-Z0-9.\-]", "", value)
    return value


def is_probably_german_yahoo_symbol(symbol: str, exchange: str = "") -> bool:
    """Prüft, ob ein Yahoo-Suchtreffer plausibel zu einem deutschen Index passt.

    Wichtig: Der Yahoo-Search-Fallback darf US-Symbole wie TSLX oder OTC-Symbole
    wie DRRKF nicht einfach mit .DE ergänzen. Genau dadurch entstanden beim SDAX
    ungültige Ticker wie TSLX.DE oder DRRKF.DE.
    """
    symbol = clean_ticker(symbol)
    exchange = str(exchange or "").upper().strip()
    if not symbol:
        return False
    if symbol.endswith(GERMAN_INDEX_ALLOWED_SUFFIXES):
        return True
    german_exchanges = {"GER", "ETR", "EUX", "FRA", "STU", "MUN", "HAM", "HAN", "DUS", "BER"}
    if exchange in german_exchanges and "." not in symbol:
        return True
    return False


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
        "change_5d", "change_1y", "total_return_1y", "vol_30d", "vol_1y",
        "max_drawdown_1y", "drawdown_3y_high_pct", "drawdown_5y_high_pct",
        "high_52w", "low_52w", "market_cap", "pe_ratio",
        "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda", "net_margin",
        "operating_margin", "roe", "roa", "dividend_yield", "dividend_per_share",
        "payout_ratio", "dividend_growth_5y", "dividend_frequency",
        "dividend_yield_5y_avg", "dividend_yield_vs_5y_avg_pct",
        "operating_cashflow", "free_cashflow", "shares_outstanding",
        "cashflow_dividend_coverage", "debt_to_equity", "net_debt_ebitda", "beta", "data_updated_at",
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


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def resolve_german_yahoo_ticker(company_name: str) -> str:
    """Versucht bei Wikipedia-Tabellen ohne Symbol den passenden Yahoo-Ticker zu finden.

    Das ist ein Fallback für Seiten wie den SDAX, auf denen teils nur Firmenname,
    Branche und Sitz stehen. Für eine produktive Version bleiben lokale CSV-Dateien
    in data/indices/*.csv zuverlässiger.
    """
    if not company_name or pd.isna(company_name):
        return ""

    name = str(company_name)
    clean_name = re.sub(r"\[.*?\]", "", name)
    clean_name = re.sub(r"\b(AG|SE|KGaA|KGAA|GmbH|Holding|Group|S\.A\.|S\.A|N\.V\.|PLC)\b", " ", clean_name, flags=re.I)
    clean_name = re.sub(r"\s+", " ", clean_name).strip()

    queries = []
    for candidate in (name, clean_name, clean_name.split(" ")[0] if clean_name else ""):
        candidate = str(candidate).strip()
        if candidate and candidate not in queries:
            queries.append(candidate)

    best_symbol = ""
    best_score = -1
    for query in queries[:3]:
        try:
            response = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": query, "quotesCount": 8, "newsCount": 0, "lang": "de-DE", "region": "DE"},
                headers=DEFAULT_HEADERS,
                timeout=12,
            )
            response.raise_for_status()
            quotes = response.json().get("quotes", [])
        except Exception:
            continue

        for quote in quotes:
            symbol = clean_ticker(quote.get("symbol", ""))
            short_name = str(quote.get("shortname") or quote.get("longname") or "")
            exchange = str(quote.get("exchange", "")).upper()
            quote_type = str(quote.get("quoteType", "")).upper()

            if not symbol or quote_type not in ("EQUITY", ""):
                continue

            # Kein blindes ".DE"-Anhängen mehr: US-/OTC-Treffer werden verworfen.
            if not is_probably_german_yahoo_symbol(symbol, exchange):
                continue

            score = 0
            if symbol.endswith(".DE"):
                score += 120
            elif symbol.endswith(".F"):
                score += 70
            elif symbol.endswith(".PA"):
                # Airbus ist DAX-Mitglied, der zuverlässige Yahoo-Ticker ist AIR.PA.
                score += 65
            elif "." not in symbol and exchange in {"GER", "FRA", "EUX", "ETR"}:
                score += 80
            if exchange in {"GER", "FRA", "EUX", "ETR"}:
                score += 40

            first_token = clean_name.lower().split(" ")[0] if clean_name else ""
            if first_token and first_token in short_name.lower():
                score += 25
            if clean_name and clean_name.lower() in short_name.lower():
                score += 35

            if score > best_score:
                best_score = score
                best_symbol = symbol

    if best_symbol and "." not in best_symbol:
        best_symbol = f"{best_symbol}.DE"
    return clean_ticker(best_symbol)


def parse_german_index_tables(tables: list[pd.DataFrame], min_rows: int, index_label: str) -> pd.DataFrame:
    """Extrahiert deutsche Indexmitglieder aus Wikipedia-Tabellen.

    DAX/MDAX enthalten häufig eine Symbol-Spalte. SDAX-Tabellen enthalten je nach
    Seite manchmal nur Namen; dann wird als Fallback über Yahoo Finance gesucht.
    Lokale CSV-Dateien bleiben stabiler und sind für saubere Ergebnisse empfohlen.
    """
    partial_results: list[pd.DataFrame] = []

    for table in tables:
        table = normalize_columns(table)
        columns = list(table.columns)
        name_col = find_col(columns, [
            "Company", "Unternehmen", "Name", "Constituent", "Security", "Issuer", "Emittent"
        ])
        ticker_col = find_col(columns, [
            "Ticker symbol", "Ticker", "Symbol", "Yahoo", "Yahoo symbol"
        ])
        sector_col = find_col(columns, [
            "Industry", "Sector", "Industrie", "Branche", "Prime Standard sector", "Supersector"
        ])

        if not name_col:
            continue
        if ticker_col and any(word in ticker_col.lower() for word in ("isin", "wkn")):
            ticker_col = None

        if ticker_col:
            result = table[[name_col, ticker_col]].copy()
            result.columns = ["name", "ticker_yahoo"]
            result["ticker_yahoo"] = (
                result["ticker_yahoo"].astype(str)
                .str.replace(r"\[.*?\]", "", regex=True)
                .str.replace("\xa0", " ", regex=False)
                .str.strip()
            )
            result["ticker_yahoo"] = result["ticker_yahoo"].apply(
                lambda ticker: ticker if "." in clean_ticker(ticker) else f"{clean_ticker(ticker)}.DE"
            )
            # Offensichtliche Nicht-DE-Artefakte aus Webtabellen entfernen.
            result["ticker_yahoo"] = result["ticker_yahoo"].map(clean_ticker)
            result = result[result["ticker_yahoo"].str.endswith(GERMAN_INDEX_ALLOWED_SUFFIXES, na=False)]
        else:
            # Fallback für Tabellen ohne Symbol-Spalte, z. B. manche SDAX-Seiten.
            if len(table) < max(10, int(min_rows * 0.5)):
                continue
            result = table[[name_col]].copy()
            result.columns = ["name"]
            result["ticker_yahoo"] = result["name"].astype(str).map(resolve_german_yahoo_ticker)

        result["sector"] = table[sector_col] if sector_col else "Unbekannt"
        try:
            result = validate_constituents(result)
        except Exception:
            continue

        if len(result) >= min_rows:
            return result
        if not result.empty:
            partial_results.append(result)

    if partial_results:
        combined = pd.concat(partial_results, ignore_index=True)
        combined = validate_constituents(combined)
        # Lieber einen teilweise geladenen SDAX anzeigen als komplett scheitern;
        # im UI kann später eine Warnung ergänzt werden. Für exakte Listen: CSV nutzen.
        if len(combined) >= max(10, int(min_rows * 0.4)):
            return combined

    raise RuntimeError(f"Keine passende {index_label}-Tabelle mit Yahoo-Tickern gefunden.")


def parse_dax_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    return parse_german_index_tables(tables, min_rows=30, index_label="DAX")


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
    local_path = INDEX_LOCAL_FILES.get(index_name)
    if local_path and local_path.exists():
        return validate_constituents(pd.read_csv(local_path))

    if index_name in GERMAN_INDEX_SOURCES:
        config = GERMAN_INDEX_SOURCES[index_name]
        errors: list[str] = []
        best_result = pd.DataFrame()
        for url in config["urls"]:
            try:
                parsed = parse_german_index_tables(
                    read_html_tables(fetch_html(url)),
                    min_rows=int(config["min_rows"]),
                    index_label=index_name,
                )
                if len(parsed) > len(best_result):
                    best_result = parsed
                # Beim DAX erwarten wir 40 Werte. V3.8 hat Airbus/AIR.PA herausgefiltert
                # und dadurch schon beim Index-Universum nur 39 Werte geladen.
                if index_name != "DAX 40" or len(parsed) >= 40:
                    return parsed.reset_index(drop=True)
            except Exception as error:
                errors.append(f"{url}: {error}")

        if index_name == "DAX 40":
            static_dax = validate_constituents(pd.DataFrame(DAX40_STATIC_CONSTITUENTS))
            if not best_result.empty:
                combined = pd.concat([best_result, static_dax], ignore_index=True)
                combined = validate_constituents(combined)
                if len(combined) >= 40:
                    return combined.head(40).reset_index(drop=True)
            return static_dax

        if not best_result.empty:
            return best_result.reset_index(drop=True)
        raise RuntimeError(f"{index_name} konnte nicht geladen werden. " + " | ".join(errors))

    if index_name == "S&P 500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        return parse_sp500_tables(read_html_tables(fetch_html(url)))

    raise ValueError(f"Unbekannter Index: {index_name}")


def benchmark_for_index(index_name: str) -> str:
    return BENCHMARK_BY_INDEX.get(index_name, "^GSPC")


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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def download_price_histories(tickers: tuple[str, ...], period: str = "5y") -> dict[str, pd.DataFrame]:
    """Lädt historische Kurse gebündelt und behält Close sowie Adj Close.

    Close = Preisrendite, Adj Close = von Yahoo bereinigte Reihe (als
    Näherung für Total Return). Die Datenquelle bleibt Yahoo Finance.
    """
    clean_tickers = tuple(
        dict.fromkeys(clean_ticker(ticker) for ticker in tickers if clean_ticker(ticker))
    )
    # Schutz gegen Artefakte wie "$SRV.DE" oder versehentlich erzeugte leere Ticker.
    clean_tickers = tuple(ticker for ticker in clean_tickers if re.match(r"^[A-Z0-9][A-Z0-9.\-]*$", ticker))
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
            valid_columns = [
                column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                if column in history.columns
            ]
            histories[ticker] = history[valid_columns].copy()
        else:
            histories[ticker] = pd.DataFrame()
    return histories


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict[str, Any]:
    """Lädt Kennzahlen plus Basisprofil von Yahoo Finance.

    Yahoo liefert nicht für jeden Börsenplatz alle Profilfelder. Die Funktion
    gibt deshalb nur die Felder zurück, die das Tool tatsächlich verwendet;
    fehlende Werte werden im UI als „–“ dargestellt.
    """
    wanted_keys = [
        # Bewertung, Profitabilität, Bilanz und Währung
        "marketCap", "trailingPE", "forwardPE", "priceToBook",
        "priceToSalesTrailing12Months", "enterpriseToEbitda",
        "profitMargins", "operatingMargins", "returnOnEquity",
        "returnOnAssets", "dividendYield", "dividendRate", "payoutRatio",
        "debtToEquity", "totalDebt", "totalCash", "ebitda",
        "operatingCashflow", "freeCashflow", "sharesOutstanding",
        "currency", "financialCurrency", "beta",

        # Unternehmensprofil und Management
        "shortName", "longName", "sector", "industry", "country", "city",
        "fullTimeEmployees", "website", "longBusinessSummary",
        "companyOfficers",

        # Ownership und Analystenschätzungen
        "heldPercentInstitutions", "heldPercentInsiders",
        "recommendationKey", "recommendationMean",
        "targetMeanPrice", "targetHighPrice", "targetLowPrice",
        "numberOfAnalystOpinions",
    ]
    try:
        raw_info = yf.Ticker(ticker).get_info() or {}
    except Exception:
        raw_info = {}

    return {key: raw_info.get(key) for key in wanted_keys}


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


def dividend_yield_history_metrics(
    dividend_frame: pd.DataFrame,
    history: pd.DataFrame,
    current_yield: Any,
) -> tuple[Optional[float], Optional[float]]:
    """Vergleicht die aktuelle Dividendenrendite grob mit der eigenen 5J-Historie.

    Für jedes abgeschlossene Kalenderjahr wird die Jahressumme der Dividenden
    durch den durchschnittlichen Schlusskurs desselben Jahres geteilt. Das ist
    keine perfekte Total-Return-Rechnung, gibt aber eine nützliche Einordnung:
    Ist die heutige Rendite ungewöhnlich hoch im Vergleich zur eigenen Historie?
    """
    if dividend_frame is None or dividend_frame.empty or history is None or history.empty or "Close" not in history.columns:
        return None, None

    frame = dividend_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.dropna(subset=["date", "amount"])
    if frame.empty:
        return None, None

    prices = ensure_datetime_index(history.copy())
    close = pd.to_numeric(prices["Close"], errors="coerce").dropna()
    if close.empty:
        return None, None

    current_year = datetime.now().year

    # Wichtig: Erst filtern, dann mit einer gleich langen Gruppierungsachse gruppieren.
    # Sonst erzeugt Pandas bei aktuellen Versionen den Fehler
    # "Grouper and axis must be same length".
    div_past = frame[frame["date"].dt.year < current_year].copy()
    price_past = close[close.index.year < current_year].copy()

    if div_past.empty or price_past.empty:
        return None, None

    annual_div = div_past.groupby(div_past["date"].dt.year)["amount"].sum().sort_index()
    annual_price = price_past.groupby(price_past.index.year).mean().sort_index()
    common_years = sorted(set(annual_div.index).intersection(set(annual_price.index)))
    if not common_years:
        return None, None

    common_years = common_years[-5:]
    yields = []
    for year in common_years:
        price = safe_float(annual_price.loc[year])
        div = safe_float(annual_div.loc[year])
        if price not in (None, 0) and div is not None and div > 0:
            yields.append(div / price * 100)

    if not yields:
        return None, None

    avg_yield = sum(yields) / len(yields)
    current = safe_float(current_yield)
    vs_avg = None if current is None or avg_yield == 0 else (current / avg_yield - 1) * 100
    return avg_yield, vs_avg


def metrics_from_ticker(
    ticker: str,
    name: str,
    sector: str,
    history: pd.DataFrame,
    info: dict[str, Any],
    dividends: pd.Series,
) -> dict[str, Any]:
    """Erstellt Kennzahlen inklusive Preis- und bereinigter Gesamtrendite."""
    record: dict[str, Any] = {
        "name": name,
        "ticker_yahoo": ticker,
        "sector": sector or "Unbekannt",
        "currency": info.get("currency") or info.get("financialCurrency") or "–",
        "last_price": None,
        "change_1d": None,
        "change_5d": None,
        "change_1y": None,
        "total_return_1y": None,
        "vol_30d": None,
        "vol_1y": None,
        "max_drawdown_1y": None,
        "drawdown_3y_high_pct": None,
        "drawdown_5y_high_pct": None,
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
        "dividend_yield_5y_avg": None,
        "dividend_yield_vs_5y_avg_pct": None,
        "operating_cashflow": safe_float(info.get("operatingCashflow")),
        "free_cashflow": safe_float(info.get("freeCashflow")),
        "shares_outstanding": safe_float(info.get("sharesOutstanding")),
        "cashflow_dividend_coverage": None,
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
                cumulative = (1 + returns).cumprod()
                record["max_drawdown_1y"] = float((cumulative / cumulative.cummax() - 1).min() * 100)

            if last_price := safe_float(close.iloc[-1]):
                for years, column in ((3, "drawdown_3y_high_pct"), (5, "drawdown_5y_high_pct")):
                    window = close.tail(int(252 * years) + 5)
                    high_value = safe_float(window.max()) if not window.empty else None
                    if high_value not in (None, 0):
                        record[column] = (last_price / high_value - 1) * 100

        avg_yield, yield_vs_avg = dividend_yield_history_metrics(dividends, history, record.get("dividend_yield"))
        record["dividend_yield_5y_avg"] = avg_yield
        record["dividend_yield_vs_5y_avg_pct"] = yield_vs_avg

        if "Adj Close" in history.columns:
            adjusted = pd.to_numeric(history["Adj Close"], errors="coerce").dropna()
            record["total_return_1y"] = series_change(adjusted, 252) if len(adjusted) >= 253 else None

    trailing_dps = trailing_dividend_per_share(dividends)
    if record["dividend_per_share"] is None:
        record["dividend_per_share"] = trailing_dps

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
    """Lädt Kurse gebündelt und Fundamentals pro Ticker mit Cache."""
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
            if not info:
                errors.append(f"{ticker}: keine Fundamentaldaten")
        except Exception as error:
            errors.append(f"{ticker}: {type(error).__name__}: {error}")

    frame = pd.DataFrame(records)
    for column in empty_metrics_frame().columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[empty_metrics_frame().columns], histories, errors


# -----------------------------------------------------------------------------
# Datenstatus und Sektor-Zeitreihen
# -----------------------------------------------------------------------------

DATA_STATUS_FIELD_GROUPS = {
    "Kursdaten": ["last_price", "change_1d", "change_5d", "change_1y", "vol_1y", "max_drawdown_1y"],
    "Bewertung": ["market_cap", "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda"],
    "Qualität": ["net_margin", "operating_margin", "roe", "roa", "debt_to_equity", "net_debt_ebitda"],
    "Dividende": ["dividend_yield", "dividend_per_share", "payout_ratio", "dividend_growth_5y", "dividend_frequency"],
    "Cashflow": ["operating_cashflow", "free_cashflow", "cashflow_dividend_coverage"],
    "Deep Value": ["drawdown_3y_high_pct", "drawdown_5y_high_pct", "dividend_yield_5y_avg", "dividend_yield_vs_5y_avg_pct"],
}


def _has_value(value: Any) -> bool:
    """Robuste Prüfung für Datenstatus-Tab."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str) and value.strip() in ("", "–", "None", "nan"):
        return False
    return True


def row_coverage(row: pd.Series, fields: list[str]) -> tuple[int, int, float]:
    present = sum(1 for field in fields if field in row.index and _has_value(row.get(field)))
    total = len(fields)
    pct = present / total * 100 if total else 0.0
    return present, total, pct


def build_data_status(
    constituents: pd.DataFrame,
    metrics: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    errors: list[str],
    index_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Erzeugt eine verständliche Datenabdeckung je Ticker.

    Ziel: Wenn ein Index 40 Werte hat, aber nur 39/40 geladen werden oder einzelne
    Kennzahlen fehlen, sieht man im Tool sofort warum.
    """
    if constituents is None or constituents.empty:
        return {
            "index": index_name,
            "angefragt": 0,
            "analysiert": 0,
            "mit_kursdaten": 0,
            "abdeckung_prozent": 0.0,
            "fehler": len(errors or []),
        }, pd.DataFrame()

    wanted = constituents.copy()
    wanted["ticker_yahoo"] = wanted["ticker_yahoo"].map(clean_ticker)
    loaded = metrics.copy() if metrics is not None else pd.DataFrame()
    if not loaded.empty:
        loaded["ticker_yahoo"] = loaded["ticker_yahoo"].map(clean_ticker)

    merged = wanted[["name", "ticker_yahoo", "sector"]].merge(
        loaded, on="ticker_yahoo", how="left", suffixes=("_index", "")
    )
    if "name" not in merged.columns and "name_index" in merged.columns:
        merged["name"] = merged["name_index"]
    if "sector" not in merged.columns and "sector_index" in merged.columns:
        merged["sector"] = merged["sector_index"]

    error_map: dict[str, list[str]] = {}
    for err in errors or []:
        ticker = clean_ticker(str(err).split(":", 1)[0])
        error_map.setdefault(ticker, []).append(str(err))

    rows: list[dict[str, Any]] = []
    all_fields = sorted({field for fields in DATA_STATUS_FIELD_GROUPS.values() for field in fields})
    for _, row in merged.iterrows():
        ticker = clean_ticker(row.get("ticker_yahoo"))
        history = histories.get(ticker, pd.DataFrame()) if histories else pd.DataFrame()
        has_history = history is not None and not history.empty and "Close" in history.columns
        present, total, pct = row_coverage(row, all_fields)
        status = "OK"
        if not has_history:
            status = "keine Kursdaten"
        elif pct < 50:
            status = "viele Kennzahlen fehlen"
        elif pct < 75:
            status = "teilweise"

        detail = {
            "Status": status,
            "Name": row.get("name") or row.get("name_index"),
            "Ticker": ticker,
            "Sektor": row.get("sector") or row.get("sector_index"),
            "Kursdaten": "Ja" if has_history else "Nein",
            "Historie Tage": len(history) if history is not None else 0,
            "Datenabdeckung": pct,
            "Vorhandene Felder": present,
            "Mögliche Felder": total,
            "Fehler/Hinweis": " | ".join(error_map.get(ticker, [])),
        }
        for group_name, fields in DATA_STATUS_FIELD_GROUPS.items():
            g_present, g_total, g_pct = row_coverage(row, fields)
            detail[group_name] = g_pct
        rows.append(detail)

    detail_df = pd.DataFrame(rows)
    requested = len(wanted)
    analysed = len(loaded) if loaded is not None else 0
    with_prices = int((detail_df["Kursdaten"] == "Ja").sum()) if not detail_df.empty else 0
    summary = {
        "index": index_name,
        "angefragt": requested,
        "analysiert": analysed,
        "mit_kursdaten": with_prices,
        "abdeckung_prozent": analysed / requested * 100 if requested else 0.0,
        "kursdaten_prozent": with_prices / requested * 100 if requested else 0.0,
        "durchschnitt_datenabdeckung": safe_float(detail_df["Datenabdeckung"].mean()) if not detail_df.empty else None,
        "fehler": len(errors or []),
    }
    return summary, detail_df


def period_return_from_history(history: pd.DataFrame, trading_days: int, use_adjusted: bool = False) -> Optional[float]:
    if history is None or history.empty:
        return None
    column = "Adj Close" if use_adjusted and "Adj Close" in history.columns else "Close"
    if column not in history.columns:
        return None
    values = pd.to_numeric(history[column], errors="coerce").dropna()
    if len(values) < 2:
        return None
    if len(values) <= trading_days:
        base = values.iloc[0]
    else:
        base = values.iloc[-1 - trading_days]
    last = values.iloc[-1]
    base = safe_float(base)
    last = safe_float(last)
    if base in (None, 0) or last is None:
        return None
    return (last / base - 1) * 100.0


def sector_timeframe_stats(df: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    periods = {
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1J": 252,
        "3J": 756,
        "5J": 1260,
    }
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ticker = clean_ticker(row.get("ticker_yahoo"))
        history = histories.get(ticker, pd.DataFrame()) if histories else pd.DataFrame()
        base = {
            "ticker_yahoo": ticker,
            "name": row.get("name"),
            "sector": row.get("sector") or "Unbekannt",
            "value_trigger": bool(row.get("value_trigger")) if _has_value(row.get("value_trigger")) else False,
            "total_score": safe_float(row.get("total_score")),
            "value_score": safe_float(row.get("value_score")),
            "dividend_yield": safe_float(row.get("dividend_yield")),
            "vol_1y": safe_float(row.get("vol_1y")),
            "max_drawdown_1y": safe_float(row.get("max_drawdown_1y")),
        }
        for label, days in periods.items():
            base[f"Kursrendite_{label}"] = period_return_from_history(history, days, use_adjusted=False)
            base[f"Gesamtrendite_{label}"] = period_return_from_history(history, days, use_adjusted=True)
        rows.append(base)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame()
    agg_map: dict[str, Any] = {
        "Unternehmen": ("ticker_yahoo", "count"),
        "Value_Trigger": ("value_trigger", lambda values: int(pd.Series(values).fillna(False).sum())),
        "Qualitäts_Score": ("total_score", "mean"),
        "Value_Score": ("value_score", "mean"),
        "Dividendenrendite": ("dividend_yield", "mean"),
        "Volatilität_1J": ("vol_1y", "mean"),
        "Max_Drawdown_1J": ("max_drawdown_1y", "mean"),
    }
    for label in periods:
        agg_map[f"Kurs_{label}"] = (f"Kursrendite_{label}", "mean")
        agg_map[f"Gesamt_{label}"] = (f"Gesamtrendite_{label}", "mean")
    return detail.groupby("sector", dropna=False).agg(**agg_map).reset_index()


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


def load_recent_sentiment_map(days_back: int = 120) -> dict[str, dict[str, Any]]:
    """Liest gespeicherte News-/Event-Sentiments als Zusatzsignal für den Deep-Value-Scanner.

    Falls noch keine News für einen Ticker geladen wurden, bleibt das Signal leer.
    Der Scanner funktioniert dann weiterhin, erhält aber keine Punkte für das News-Sentiment.
    """
    try:
        if not EVENTS_PATH.exists():
            return {}
        events = pd.read_csv(EVENTS_PATH)
    except Exception:
        return {}
    if events.empty or "ticker_yahoo" not in events.columns:
        return {}

    events["ticker_yahoo"] = events["ticker_yahoo"].map(clean_ticker)
    events["date"] = pd.to_datetime(events.get("date"), errors="coerce")
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=days_back))
    events = events[events["date"].notna() & (events["date"] >= cutoff)]
    if events.empty:
        return {}

    if "sentiment_score" not in events.columns:
        events["sentiment_score"] = 0
    events["sentiment_score"] = pd.to_numeric(events["sentiment_score"], errors="coerce").fillna(0)

    result: dict[str, dict[str, Any]] = {}
    for ticker, group in events.groupby("ticker_yahoo"):
        scores = group["sentiment_score"]
        result[ticker] = {
            "count": int(len(group)),
            "negative_count": int((scores < 0).sum()),
            "positive_count": int((scores > 0).sum()),
            "avg_sentiment": float(scores.mean()) if len(scores) else 0.0,
        }
    return result


def compute_special_situation_score(row: pd.Series, sentiment_info: Optional[dict[str, Any]] = None) -> pd.Series:
    """BAT-/Deep-Value-Pattern: hohe Rendite + großer Drawdown + Cashflow-Stabilität.

    Das ist bewusst kein Kaufsignal. Es ist ein Warn-/Research-Trigger für Fälle,
    bei denen Standardkennzahlen wie KGV oder Payout Ratio durch Sondereffekte
    verzerrt sein können.
    """
    sentiment_info = sentiment_info or {}

    dividend_yield = safe_float(row.get("dividend_yield"))
    dd_3y = safe_float(row.get("drawdown_3y_high_pct"))
    dd_5y = safe_float(row.get("drawdown_5y_high_pct"))
    dd_1y = safe_float(row.get("drawdown_1y_high_pct"))
    drawdown_candidates = [value for value in [dd_5y, dd_3y, dd_1y] if value is not None]
    deepest_drawdown = min(drawdown_candidates) if drawdown_candidates else None

    operating_cashflow = safe_float(row.get("operating_cashflow"))
    free_cashflow = safe_float(row.get("free_cashflow"))
    cashflow_positive = (free_cashflow is not None and free_cashflow > 0) or (operating_cashflow is not None and operating_cashflow > 0)
    coverage = safe_float(row.get("cashflow_dividend_coverage"))
    net_debt_ebitda = safe_float(row.get("net_debt_ebitda"))
    pe = safe_float(row.get("pe_ratio"))
    payout = safe_float(row.get("payout_ratio"))
    yield_vs_avg = safe_float(row.get("dividend_yield_vs_5y_avg_pct"))

    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    # 1) Dividendenrendite
    if dividend_yield is not None:
        if dividend_yield >= 7:
            points = 20
        elif dividend_yield >= 5:
            points = 15
        elif dividend_yield >= 3.5:
            points = 9
        else:
            points = 0
        score += points
        reasons.append(f"Rendite {points:.0f}/20 ({format_percent(dividend_yield, 1)})")
        if yield_vs_avg is not None and yield_vs_avg >= 25:
            checks.append(f"Rendite liegt {format_percent(yield_vs_avg, 0, signed=True)} über dem eigenen 5J-Schnitt")
    else:
        reasons.append("Rendite 0/20 (keine Daten)")

    # 2) Drawdown vom 3J-/5J-Hoch
    if deepest_drawdown is not None:
        drawdown_abs = abs(min(deepest_drawdown, 0))
        if drawdown_abs >= 35:
            points = 20
        elif drawdown_abs >= 25:
            points = 15
        elif drawdown_abs >= 15:
            points = 8
        else:
            points = 0
        score += points
        reasons.append(f"Drawdown {points:.0f}/20 ({format_percent(deepest_drawdown, 1, signed=True)})")
    else:
        reasons.append("Drawdown 0/20 (keine Daten)")

    # 3) Cashflow positiv
    if cashflow_positive:
        score += 15
        source = "Free Cashflow" if free_cashflow is not None and free_cashflow > 0 else "operativer Cashflow"
        reasons.append(f"Cashflow 15/15 ({source} positiv)")
    else:
        reasons.append("Cashflow 0/15 (nicht positiv oder keine Daten)")

    # 4) Dividende durch Cashflow gedeckt
    if coverage is not None:
        if coverage >= 1.2:
            points = 15
        elif coverage >= 1.0:
            points = 11
        elif coverage >= 0.8:
            points = 5
        else:
            points = 0
        score += points
        reasons.append(f"FCF/Dividende {points:.0f}/15 ({format_number(coverage, 2)}x)")
    elif payout is not None and 0 <= payout <= 90:
        score += 6
        reasons.append(f"FCF/Dividende 6/15 (Payout ersatzweise {format_percent(payout, 0)})")
    else:
        reasons.append("FCF/Dividende 0/15 (keine belastbare Deckung)")

    # 5) Leverage kontrollierbar
    if net_debt_ebitda is not None:
        if net_debt_ebitda < 3.5:
            points = 10
        elif net_debt_ebitda < 5:
            points = 5
        else:
            points = 0
        score += points
        reasons.append(f"Leverage {points:.0f}/10 ({format_number(net_debt_ebitda, 2)}x Net Debt/EBITDA)")
    else:
        reasons.append("Leverage 0/10 (keine Daten)")

    # 6) KGV niedrig oder durch Sondereffekt möglicherweise verzerrt
    eps_distortion_possible = pe is not None and pe <= 0 and cashflow_positive
    if pe is not None:
        if 0 < pe <= 12:
            points = 10
            detail = f"KGV {format_number(pe, 1)}"
        elif 0 < pe <= 18:
            points = 6
            detail = f"KGV {format_number(pe, 1)}"
        elif eps_distortion_possible:
            points = 8
            detail = "negatives KGV, aber Cashflow positiv"
            checks.append("Reported EPS könnte durch Sondereffekt verzerrt sein")
        else:
            points = 0
            detail = f"KGV {format_number(pe, 1)}"
        score += points
        reasons.append(f"Bewertung/Sondereffekt {points:.0f}/10 ({detail})")
    else:
        reasons.append("Bewertung/Sondereffekt 0/10 (keine KGV-Daten)")

    # 7) Negatives News-Sentiment ohne Cashflow-Kollaps
    negative_count = int(sentiment_info.get("negative_count", 0) or 0)
    avg_sentiment = safe_float(sentiment_info.get("avg_sentiment"))
    if cashflow_positive and (negative_count > 0 or (avg_sentiment is not None and avg_sentiment < 0)):
        score += 10
        reasons.append(f"News-Kontrast 10/10 ({negative_count} negative Treffer, Cashflow positiv)")
    elif cashflow_positive and sentiment_info.get("count"):
        score += 3
        reasons.append("News-Kontrast 3/10 (News geladen, aber nicht klar negativ)")
    else:
        reasons.append("News-Kontrast 0/10 (keine negativen News oder News nicht geladen)")

    trigger = bool(
        score >= 70
        and dividend_yield is not None and dividend_yield >= 5
        and deepest_drawdown is not None and deepest_drawdown <= -25
        and cashflow_positive
    )

    status = "Sondersituation prüfen" if trigger else "Beobachten" if score >= 55 else "kein BAT-Muster"
    if cashflow_positive:
        checks.append("Cashflow weiterhin positiv")
    if coverage is not None and coverage >= 1:
        checks.append("Dividende grob durch Free Cashflow gedeckt")
    if net_debt_ebitda is not None and net_debt_ebitda < 3.5:
        checks.append("Leverage unter 3,5x")

    return pd.Series({
        "special_situation_score": round(min(max(score, 0), 100), 1),
        "special_situation_trigger": trigger,
        "special_situation_status": status,
        "special_situation_reason": " | ".join(reasons),
        "special_situation_checks": " | ".join(checks) if checks else "Keine besonderen Zusatzhinweise",
        "deepest_drawdown_3_5y_pct": deepest_drawdown,
    })


def enrich_with_special_situations(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    sentiment_map = load_recent_sentiment_map(days_back=120)
    scored = df.apply(
        lambda row: compute_special_situation_score(
            row, sentiment_map.get(clean_ticker(row.get("ticker_yahoo")), {})
        ),
        axis=1,
    )
    return pd.concat([df.copy(), scored], axis=1)


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
        "dividend_per_share", "total_score", "value_score", "value_trigger", "vol_1y",
        "total_return_1y", "max_drawdown_1y",
    ]
    available_columns = [column for column in metric_columns if column in metrics.columns]
    result = portfolio.merge(metrics[available_columns], on="ticker_yahoo", how="left", suffixes=("", "_asset"))

    # currency aus Portfolio/Transaktionsbuch = Einstandswährung; currency_asset
    # aus Market Data = Handelswährung des Wertpapiers.
    asset_currency = result.get("currency_asset", result.get("currency", "EUR"))
    result["asset_currency"] = asset_currency.fillna("EUR") if isinstance(asset_currency, pd.Series) else "EUR"
    if "currency" in result.columns:
        result["cost_currency"] = result["currency"].replace(r"^\s*$", pd.NA, regex=True).fillna(result["asset_currency"])
    else:
        result["cost_currency"] = result["asset_currency"]

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
    result["fx_status"] = result.apply(
        lambda row: "OK" if pd.notna(row.get("fx_asset_to_eur")) and pd.notna(row.get("fx_cost_to_eur")) else "FX fehlt",
        axis=1,
    )
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
# Version 3.0 – Datenqualität, News-/Event-Pipeline, Portfolio-Risiko & Research
# -----------------------------------------------------------------------------

# Die App legt diese Dateien bei Bedarf automatisch an. Sie sind bewusst CSV-
# basiert, damit du sie auch ohne Datenbank in Excel bearbeiten kannst.
TRANSACTIONS_PATH = DATA_DIR / "transactions.csv"
ALIAS_PATH = DATA_DIR / "company_aliases.csv"
EVENTS_PATH = DATA_DIR / "events.csv"
NEWS_SNAPSHOT_DIR = DATA_DIR / "news_snapshots"
NEWS_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_RSS_SOURCES: list[dict[str, str]] = [
    {
        "name": "finanzen.net News",
        "url": "https://www.finanzen.net/rss/news",
        "kind": "global",
    },
    {
        "name": "Tagesschau Wirtschaft",
        "url": "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
        "kind": "global",
    },
    {
        "name": "Tagesschau Unternehmen",
        "url": "https://www.tagesschau.de/wirtschaft/unternehmen/index~rss2.xml",
        "kind": "global",
    },
]

# Diese Profile setzen nur sinnvolle Scanner-Startwerte. Die Regeln bleiben
# anschließend transparent über die Sidebar-Slider anpassbar.
STRATEGY_PROFILES: dict[str, dict[str, Any]] = {
    "Ausgewogen": {
        "drawdown": 25,
        "payout": 90,
        "score": 65,
        "yield": 5.0,
        "description": "Qualität, Ausschüttungsquote, Rendite und Drawdown ausgewogen gewichtet.",
    },
    "Defensive Dividende": {
        "drawdown": 15,
        "payout": 75,
        "score": 70,
        "yield": 3.5,
        "description": "Strengere Qualität und Ausschüttungsquote, moderate Mindest-Rendite.",
    },
    "Value / Turnaround": {
        "drawdown": 35,
        "payout": 100,
        "score": 55,
        "yield": 2.5,
        "description": "Stärkerer Drawdown und etwas mehr Raum für zyklische Titel; kein Kaufsignal.",
    },
    "Ertragsorientiert": {
        "drawdown": 20,
        "payout": 85,
        "score": 60,
        "yield": 6.0,
        "description": "Fokus auf hohe Ausschüttung, aber mit Mindestqualität und Payout-Limit.",
    },
}

EVENT_META: dict[str, dict[str, str]] = {
    "news": {"label": "News", "shape": "circle"},
    "earnings": {"label": "Quartalszahlen", "shape": "triangle-up"},
    "dividend": {"label": "Dividende", "shape": "diamond"},
    "annual_meeting": {"label": "Hauptversammlung", "shape": "square"},
    "report": {"label": "Bericht", "shape": "cross"},
    "analyst": {"label": "Analysten", "shape": "triangle-down"},
}

KNOWN_ALIASES: dict[str, list[str]] = {
    "DHL.DE": ["DHL", "DHL Group", "Deutsche Post"],
    "VOW3.DE": ["Volkswagen", "Volkswagen Group", "VW"],
    "MBG.DE": ["Mercedes-Benz", "Mercedes Benz", "Mercedes"],
    "DTE.DE": ["Deutsche Telekom", "Telekom", "T-Mobile"],
    "MUV2.DE": ["Munich Re", "Münchener Rück", "Muenchener Rueck"],
    "DBK.DE": ["Deutsche Bank"],
    "BAS.DE": ["BASF"],
    "SIE.DE": ["Siemens"],
    "SAP.DE": ["SAP"],
    "ALV.DE": ["Allianz"],
}

# Mehrdeutige Ticker brauchen belastbare Firmenbegriffe. Ein bloßer Ticker wie
# MMM, T oder F ist als News-Suche zu unscharf und wird deshalb nicht als
# primärer Google-News-Suchbegriff verwendet.
KNOWN_ALIASES.update({
    "MMM": ["3M Company", "3M Co.", "3M Aktie", "3M"],
    "T": ["AT&T", "AT&T Inc.", "AT&T Aktie"],
    "F": ["Ford Motor", "Ford Motor Company", "Ford Aktie"],
    "A": ["Agilent Technologies", "Agilent"],
    "CAT": ["Caterpillar", "Caterpillar Inc."],
    "IT": ["Gartner", "Gartner Inc."],
})

# Begriffe, die auf den Firmenkontext hindeuten. Sie dienen nur als
# Relevanzhilfe und sind ausdrücklich kein Kauf-/Verkaufssignal.
NEWS_FINANCE_CONTEXT_TERMS = {
    "aktie", "aktien", "boerse", "börse", "kurs", "kursziel", "stock", "stocks",
    "share", "shares", "shareholder", "earnings", "results", "quarterly",
    "quartalszahlen", "quartalsbericht", "geschaeftszahlen", "geschäftszahlen",
    "jahreszahlen", "guidance", "prognose", "umsatz", "revenue", "gewinn",
    "profit", "dividende", "dividend", "investor", "investor relations",
    "analyst", "upgrade", "downgrade", "rating", "outperform", "underperform",
}

NEWS_COMPANY_ACTION_TERMS = {
    "announces", "announced", "reports", "reported", "appoints", "acquires",
    "acquisition", "settlement", "restructuring", "restructure", "spinoff",
    "spin off", "merger", "lawsuit", "verkauft", "uebernimmt", "übernimmt",
    "ernennung", "erwirbt", "vergleich", "umstrukturierung",
}


def normalize_for_search(value: Any) -> str:
    """Normalisiert Text für robuste, aber nicht zu breite Alias-Suchen."""
    text_value = str(value or "").lower()
    text_value = text_value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text_value = re.sub(r"[^a-z0-9]+", " ", text_value)
    return re.sub(r"\s+", " ", text_value).strip()


def safe_write_csv(
    df: pd.DataFrame,
    path: Path,
    retries: int = 6,
    retry_delay: float = 0.15,
) -> tuple[bool, str]:
    """Schreibt CSV robust unter Windows, ohne Streamlit bei Dateisperren zu stoppen.

    Ein eindeutiger Temp-Dateiname verhindert Kollisionen zwischen zwei schnellen
    Streamlit-Reruns. Falls Excel, Explorer-Vorschau oder ein Virenscanner die
    Ziel-Datei kurz sperren, wird mehrmals versucht und anschließend eine
    zeitgestempelte Sicherungsdatei geschrieben.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None

    try:
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}_",
            suffix=".tmp",
            dir=str(path.parent),
        )
        os.close(file_descriptor)
        temp_path = Path(temp_name)
        df.to_csv(temp_path, index=False, encoding="utf-8")

        last_error: PermissionError | None = None
        for attempt in range(max(1, int(retries))):
            try:
                os.replace(temp_path, path)
                return True, ""
            except PermissionError as error:
                last_error = error
                time.sleep(float(retry_delay) * (attempt + 1))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fallback_path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        os.replace(temp_path, fallback_path)
        return (
            False,
            f"{path.name} war gesperrt und wurde nicht überschrieben. "
            f"Die neuen Daten liegen stattdessen in {fallback_path.name}. "
            "Schließe die Datei in Excel oder einer Vorschau und aktualisiere danach erneut.",
        )
    except Exception as error:
        return False, f"CSV-Datei konnte nicht gespeichert werden: {type(error).__name__}: {error}"
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "published", "ticker_yahoo", "title", "link", "source", "source_kind",
            "matched_alias", "sentiment_score", "sentiment_label", "event_type",
            "relevance_score", "relevance_label", "relevance_reason", "is_relevant",
        ]
    )


def empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date", "ticker_yahoo", "event_type", "title", "source", "link",
            "sentiment_score", "sentiment_label", "importance", "is_future_event",
        ]
    )


def ensure_alias_file() -> None:
    if ALIAS_PATH.exists():
        return
    seed_rows = []
    for ticker, aliases in KNOWN_ALIASES.items():
        for alias in aliases:
            seed_rows.append({"ticker_yahoo": ticker, "alias": alias, "source": "Voreinstellung"})
    safe_write_csv(pd.DataFrame(seed_rows, columns=["ticker_yahoo", "alias", "source"]), ALIAS_PATH)


def read_aliases() -> pd.DataFrame:
    ensure_alias_file()
    try:
        aliases = pd.read_csv(ALIAS_PATH, dtype=str)
    except Exception:
        aliases = pd.DataFrame(columns=["ticker_yahoo", "alias", "source"])
    for column in ["ticker_yahoo", "alias", "source"]:
        if column not in aliases.columns:
            aliases[column] = ""
    aliases["ticker_yahoo"] = aliases["ticker_yahoo"].map(clean_ticker)
    aliases["alias"] = aliases["alias"].fillna("").astype(str).str.strip()
    aliases = aliases[aliases["alias"].str.len() >= 2].drop_duplicates(["ticker_yahoo", "alias"])
    return aliases.reset_index(drop=True)


def default_company_aliases(company_name: str, ticker: str) -> list[str]:
    cleaned_name = str(company_name or "").strip()
    simplified = re.sub(
        r"\b(ag|se|sa|s\.a\.|plc|n\.v\.|gmbh|kgaa|inc\.?|corp\.?|ltd\.?)\b",
        "",
        cleaned_name,
        flags=re.IGNORECASE,
    ).strip(" ,-")
    candidates = {
        cleaned_name,
        simplified,
        clean_ticker(ticker).split(".")[0],
        *KNOWN_ALIASES.get(clean_ticker(ticker), []),
    }
    # Einzelwörter werden nur übernommen, wenn sie nicht sehr kurz sind.
    candidates.update(part for part in simplified.split() if len(part) >= 5)
    return sorted({candidate.strip() for candidate in candidates if len(candidate.strip()) >= 3})


def aliases_for_ticker(company_name: str, ticker: str) -> list[str]:
    aliases = set(default_company_aliases(company_name, ticker))
    alias_frame = read_aliases()
    manual = alias_frame.loc[
        alias_frame["ticker_yahoo"] == clean_ticker(ticker), "alias"
    ].dropna().astype(str).tolist()
    aliases.update(manual)
    # Sehr kurze Ticker ohne Firmenname erzeugen viele False Positives.
    return sorted({alias for alias in aliases if len(normalize_for_search(alias)) >= 3})


def news_identity_aliases(company_name: str, ticker: str) -> list[str]:
    """Liefert Firmenbegriffe für News, ohne einen bloßen Ticker zu bevorzugen."""
    ticker = clean_ticker(ticker)
    ticker_base = ticker.split(".")[0]
    company = str(company_name or "").strip()
    simplified = re.sub(
        r"\b(ag|se|sa|s\.a\.|plc|n\.v\.|gmbh|kgaa|inc\.?|corp\.?|ltd\.?)\b",
        "",
        company,
        flags=re.IGNORECASE,
    ).strip(" ,-")

    candidates: list[str] = [company, simplified, *KNOWN_ALIASES.get(ticker, [])]
    candidates.extend(aliases_for_ticker(company_name, ticker))

    result: list[str] = []
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        norm = normalize_for_search(candidate)
        if not norm:
            continue
        # Der automatisch aus dem Ticker entstandene Begriff ist nur dann
        # sinnvoll, wenn er wirklich auch der Firmenname ist (z. B. SAP).
        if norm == normalize_for_search(ticker_base) and norm != normalize_for_search(company):
            continue
        # Sehr kurze Marken (z. B. 3M) sind nur erlaubt, wenn sie bewusst in
        # den gepflegten Standardaliasen stehen.
        is_known_short_brand = candidate in KNOWN_ALIASES.get(ticker, []) and len(norm) >= 2
        if len(norm) < 3 and not is_known_short_brand:
            continue
        result.append(candidate)

    return sorted(set(result), key=lambda item: len(normalize_for_search(item)), reverse=True)


def primary_news_query_alias(company_name: str, ticker: str) -> str:
    """Wählt für Google News einen möglichst eindeutigen Unternehmensbegriff."""
    candidates = news_identity_aliases(company_name, ticker)
    if not candidates:
        return str(company_name or clean_ticker(ticker)).strip()

    # Mehrwort-Aliase sind bei mehrdeutigen Kürzeln wesentlich präziser.
    multiword = [value for value in candidates if len(normalize_for_search(value).split()) >= 2]
    pool = multiword or candidates
    return max(pool, key=lambda item: len(normalize_for_search(item)))


def contains_alias_precise(text: str, aliases: Iterable[str]) -> Optional[str]:
    normalized = f" {normalize_for_search(text)} "
    for alias in sorted(set(aliases), key=lambda item: len(normalize_for_search(item)), reverse=True):
        candidate = normalize_for_search(alias)
        if not candidate:
            continue
        if f" {candidate} " in normalized:
            return alias
    return None


def news_context_score(text: str) -> tuple[int, list[str]]:
    normalized = normalize_for_search(text)
    terms: list[str] = []
    for term in NEWS_FINANCE_CONTEXT_TERMS:
        if normalize_for_search(term) in normalized:
            terms.append(term)
    for term in NEWS_COMPANY_ACTION_TERMS:
        if normalize_for_search(term) in normalized:
            terms.append(term)
    # Begrenze, damit eine lange Schlagwortliste nicht den Firmenbezug ersetzt.
    return min(20, len(set(terms)) * 7), sorted(set(terms))[:4]


def evaluate_news_relevance(
    title: str,
    summary: str,
    company_name: str,
    ticker: str,
    source_kind: str,
) -> dict[str, Any]:
    """Bewertet Firmenbezug und trennt verlässliche von unsicheren Treffern."""
    aliases = news_identity_aliases(company_name, ticker)
    title_alias = contains_alias_precise(title, aliases)
    summary_alias = contains_alias_precise(summary, aliases)
    matched_alias = title_alias or summary_alias
    score = 0
    reasons: list[str] = []

    if matched_alias:
        alias_length = len(normalize_for_search(matched_alias))
        if title_alias:
            score = 82 if alias_length >= 5 else 52
            reasons.append("Firmenalias im Titel")
        else:
            score = 64 if alias_length >= 5 else 38
            reasons.append("Firmenalias in der Beschreibung")

    context_points, context_terms = news_context_score(f"{title} {summary}")
    if context_points:
        score += context_points
        # Kurze, aber bewusst gepflegte Marken wie SAP, 3M oder AT&T werden
        # erst zusammen mit einem Finanz-/Unternehmenskontext als belastbar
        # eingestuft. Ohne Kontext bleiben sie absichtlich unsicher.
        if title_alias and len(normalize_for_search(title_alias)) < 5:
            score += 12
        reasons.append("Finanz-/Unternehmenskontext: " + ", ".join(context_terms))

    # Google-Suchergebnisse ohne sichtbaren Firmenbezug werden bewusst nicht
    # als relevante Unternehmensnews durchgewunken.
    if source_kind == "search_fallback" and not matched_alias:
        score = min(score, 20)
        reasons.append("nur Suchabfrage, kein sichtbarer Firmenalias")

    score = max(0, min(100, score))
    if score >= 85:
        label = "hoch"
    elif score >= 70:
        label = "mittel"
    else:
        label = "niedrig"

    return {
        "matched_alias": matched_alias or "–",
        "relevance_score": score,
        "relevance_label": label,
        "relevance_reason": " · ".join(reasons) if reasons else "kein ausreichend eindeutiger Firmenbezug",
        "is_relevant": bool(score >= 70),
    }


def save_aliases_for_ticker(ticker: str, aliases_text: str) -> None:
    alias_frame = read_aliases()
    ticker = clean_ticker(ticker)
    values = [part.strip() for part in re.split(r"[,\n;|]+", aliases_text) if part.strip()]
    alias_frame = alias_frame[alias_frame["ticker_yahoo"] != ticker].copy()
    if values:
        additions = pd.DataFrame(
            [{"ticker_yahoo": ticker, "alias": value, "source": "manuell"} for value in values]
        )
        alias_frame = pd.concat([alias_frame, additions], ignore_index=True)
    alias_frame = alias_frame.drop_duplicates(["ticker_yahoo", "alias"])
    safe_write_csv(alias_frame, ALIAS_PATH)


def classify_event_from_text(text: str) -> str:
    """Ordnet nur konkrete Finanzereignisse ein; allgemeine News bleiben News."""
    normalized = normalize_for_search(text)
    if any(phrase in normalized for phrase in [
        "quarterly results", "quarterly earnings", "earnings results", "quartalszahlen",
        "quartalsbericht", "geschaeftszahlen", "geschäftszahlen", "jahreszahlen",
    ]):
        return "earnings"
    if any(phrase in normalized for phrase in ["ex divid", "dividend", "dividende"]):
        return "dividend"
    if any(phrase in normalized for phrase in ["hauptversammlung", "annual general meeting", "annual meeting", "agm"]):
        return "annual_meeting"
    if any(phrase in normalized for phrase in ["geschaeftsbericht", "geschäftsbericht", "annual report", "quarterly report"]):
        return "report"
    if any(phrase in normalized for phrase in ["kursziel", "analyst", "upgrade", "downgrade", "outperform", "underperform"]):
        return "analyst"
    return "news"


def request_feed(url: str) -> tuple[Optional[bytes], dict[str, Any]]:
    """HTTP-Abruf mit Diagnose statt stiller Fehlerunterdrückung."""
    started = time.perf_counter()
    diagnostic: dict[str, Any] = {
        "url": url,
        "status": "Fehler",
        "http_status": None,
        "content_type": None,
        "entries": 0,
        "matches": 0,
        "message": "",
        "duration_ms": None,
    }
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
        diagnostic["http_status"] = response.status_code
        diagnostic["content_type"] = response.headers.get("Content-Type", "")
        diagnostic["duration_ms"] = round((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        payload = response.content
        if not payload:
            diagnostic["status"] = "Leer"
            diagnostic["message"] = "Antwort ohne Inhalt"
            return None, diagnostic
        diagnostic["status"] = "OK"
        return payload, diagnostic
    except requests.RequestException as error:
        diagnostic["duration_ms"] = round((time.perf_counter() - started) * 1000)
        diagnostic["message"] = f"{type(error).__name__}: {error}"
        return None, diagnostic


def parse_feed_entries(payload: bytes, source_name: str, source_kind: str) -> tuple[list[dict[str, Any]], Optional[str]]:
    parsed = feedparser.parse(payload)
    if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", []):
        error = getattr(parsed, "bozo_exception", None)
        return [], f"Parser: {type(error).__name__ if error else 'unbekannt'}"
    entries: list[dict[str, Any]] = []
    for entry in getattr(parsed, "entries", []):
        published = entry_datetime(entry)
        if published is None:
            continue
        entries.append(
            {
                "published": published.replace(tzinfo=None),
                "title": str(entry.get("title", "") or "").strip(),
                "summary": str(entry.get("summary", "") or "").strip(),
                "link": str(entry.get("link", "") or "").strip(),
                "source": source_name,
                "source_kind": source_kind,
            }
        )
    return entries, None


def google_news_rss_url(query: str, locale: str = "de") -> str:
    # Google News RSS ist nur ein optionaler Such-Fallback. Der Link enthält
    # einen klaren Firmennamen statt einer breiten allgemeinen Abfrage.
    locale = "en" if locale == "en" else "de"
    country = "US" if locale == "en" else "DE"
    return (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl={locale}&gl={country}&ceid={country}:{locale}"
    )


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_news_bundle(
    ticker: str,
    company_name: str,
    days_back: int,
    include_google_news: bool = True,
    locale: str = "de",
    max_items: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt News mit Quellenstatus und einem transparenten Firmenbezug-Score."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(days_back))).replace(tzinfo=None)
    ticker = clean_ticker(ticker)
    source_specs = [dict(source) for source in GLOBAL_RSS_SOURCES]

    if include_google_news:
        preferred_alias = primary_news_query_alias(company_name, ticker)
        query = f'"{preferred_alias}"'
        source_specs.append(
            {
                "name": f"Google News Suche: {preferred_alias}",
                "url": google_news_rss_url(query, locale),
                "kind": "search_fallback",
            }
        )

    news_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for source in source_specs:
        payload, diagnostic = request_feed(source["url"])
        diagnostic["source"] = source["name"]
        diagnostic["kind"] = source["kind"]
        diagnostic["uncertain_matches"] = 0

        if payload is None:
            diagnostics.append(diagnostic)
            continue

        parsed_entries, parser_error = parse_feed_entries(payload, source["name"], source["kind"])
        diagnostic["entries"] = len(parsed_entries)
        if parser_error:
            diagnostic["status"] = "Parser-Fehler"
            diagnostic["message"] = parser_error
            diagnostics.append(diagnostic)
            continue

        for entry in parsed_entries:
            if not entry["title"] or entry["published"] < cutoff:
                continue

            relevance = evaluate_news_relevance(
                title=entry["title"],
                summary=entry["summary"],
                company_name=company_name,
                ticker=ticker,
                source_kind=source["kind"],
            )
            # Treffer ohne sichtbaren Firmenbezug werden nicht einmal als
            # unsicher gespeichert; so bleibt die Oberfläche sauber.
            if int(relevance["relevance_score"]) < 35:
                continue

            if bool(relevance["is_relevant"]):
                diagnostic["matches"] += 1
            else:
                diagnostic["uncertain_matches"] += 1

            combined_text = f"{entry['title']} {entry['summary']}"
            sentiment_score, sentiment_label = simple_sentiment(combined_text)
            event_type = classify_event_from_text(combined_text) if bool(relevance["is_relevant"]) else "news"
            news_rows.append(
                {
                    "published": entry["published"],
                    "ticker_yahoo": ticker,
                    "title": entry["title"],
                    "link": entry["link"],
                    "source": entry["source"],
                    "source_kind": entry["source_kind"],
                    "matched_alias": relevance["matched_alias"],
                    "sentiment_score": sentiment_score,
                    "sentiment_label": sentiment_label,
                    "event_type": event_type,
                    "relevance_score": relevance["relevance_score"],
                    "relevance_label": relevance["relevance_label"],
                    "relevance_reason": relevance["relevance_reason"],
                    "is_relevant": relevance["is_relevant"],
                }
            )

        if diagnostic["status"] == "OK" and diagnostic["entries"] == 0:
            diagnostic["status"] = "Keine Einträge"
        elif diagnostic["status"] == "OK" and diagnostic["matches"] == 0 and diagnostic["uncertain_matches"] > 0:
            diagnostic["status"] = "Nur unsichere Treffer"
            diagnostic["message"] = "Treffer mit schwachem Firmenbezug werden standardmäßig ausgeblendet."
        elif diagnostic["status"] == "OK" and diagnostic["matches"] == 0:
            diagnostic["status"] = "Keine Firmen-Treffer"
        diagnostics.append(diagnostic)

    news = pd.DataFrame(news_rows) if news_rows else empty_news_frame()
    if not news.empty:
        # Gleiche Überschriften aus mehreren Quellen werden einmal behalten –
        # bevorzugt mit höherer Relevanz und neuerem Datum.
        news["_dedupe_key"] = news["title"].map(normalize_for_search)
        news = (
            news.sort_values(["relevance_score", "published"], ascending=[False, False])
            .drop_duplicates("_dedupe_key", keep="first")
            .drop(columns="_dedupe_key")
            .sort_values("published", ascending=False)
            .head(max_items)
            .reset_index(drop=True)
        )

    diagnostic_frame = pd.DataFrame(diagnostics)
    return news, diagnostic_frame


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_yahoo_calendar_events(ticker: str, days_back: int = 365, days_forward: int = 365) -> pd.DataFrame:
    """Liest verfügbare Dividenden- und Kalendereinträge defensiv aus yfinance."""
    ticker = clean_ticker(ticker)
    now = datetime.now().replace(tzinfo=None)
    earliest = now - timedelta(days=int(days_back))
    latest = now + timedelta(days=int(days_forward))
    rows: list[dict[str, Any]] = []

    try:
        dividends = fetch_dividends(ticker)
    except Exception:
        dividends = pd.DataFrame(columns=["date", "amount"])

    if dividends is not None and not dividends.empty:
        if isinstance(dividends, pd.DataFrame) and {"date", "amount"}.issubset(dividends.columns):
            dividend_rows = dividends[["date", "amount"]].copy()
        else:
            series = pd.to_numeric(dividends, errors="coerce").dropna()
            dividend_rows = pd.DataFrame({"date": series.index, "amount": series.values})

        dividend_rows["date"] = pd.to_datetime(dividend_rows["date"], errors="coerce")
        dividend_rows["amount"] = pd.to_numeric(dividend_rows["amount"], errors="coerce")
        for _, dividend_row in dividend_rows.dropna(subset=["date", "amount"]).iterrows():
            date_value = pd.Timestamp(dividend_row["date"]).tz_localize(None).to_pydatetime()
            value = float(dividend_row["amount"])
            if earliest <= date_value <= latest:
                rows.append(
                    {
                        "date": date_value,
                        "ticker_yahoo": ticker,
                        "event_type": "dividend",
                        "title": f"Dividende: {format_number(value, 4)} je Aktie",
                        "source": "Yahoo Finance",
                        "link": "",
                        "sentiment_score": 0.0,
                        "sentiment_label": "neutral",
                        "importance": "mittel",
                        "is_future_event": bool(date_value > now),
                    }
                )

    # Yahoo liefert Kalenderinformationen nicht für alle Titel und in unter-
    # schiedlichen Formaten. Deshalb sind diese Zeilen best effort.
    try:
        calendar = yf.Ticker(ticker).calendar
    except Exception:
        calendar = None

    def add_calendar_dates(value: Any, event_type: str, title: str) -> None:
        if value is None:
            return
        values: list[Any]
        if isinstance(value, (pd.Series, pd.Index, list, tuple, set)):
            values = list(value)
        else:
            values = [value]
        for raw_date in values:
            parsed_date = pd.to_datetime(raw_date, errors="coerce")
            if pd.isna(parsed_date):
                continue
            date_value = pd.Timestamp(parsed_date).tz_localize(None).to_pydatetime()
            if earliest <= date_value <= latest:
                rows.append(
                    {
                        "date": date_value,
                        "ticker_yahoo": ticker,
                        "event_type": event_type,
                        "title": title,
                        "source": "Yahoo Finance Calendar",
                        "link": "",
                        "sentiment_score": 0.0,
                        "sentiment_label": "neutral",
                        "importance": "hoch" if event_type == "earnings" else "mittel",
                        "is_future_event": bool(date_value > now),
                    }
                )

    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
        for index_value, row in calendar.iterrows():
            label = str(index_value)
            values = row.tolist()
            lowered = label.lower()
            if "earning" in lowered:
                add_calendar_dates(values, "earnings", "Quartalszahlen / Earnings")
            elif "ex-dividend" in lowered or "ex dividend" in lowered:
                add_calendar_dates(values, "dividend", "Ex-Dividende")
    elif isinstance(calendar, dict):
        for key, value in calendar.items():
            lowered = str(key).lower()
            if "earning" in lowered:
                add_calendar_dates(value, "earnings", "Quartalszahlen / Earnings")
            elif "ex-dividend" in lowered or "ex dividend" in lowered:
                add_calendar_dates(value, "dividend", "Ex-Dividende")

    events = pd.DataFrame(rows) if rows else empty_events_frame()
    if not events.empty:
        events["date"] = pd.to_datetime(events["date"], errors="coerce")
        events = (
            events.dropna(subset=["date"])
            .drop_duplicates(["ticker_yahoo", "date", "event_type", "title"])
            .sort_values("date", ascending=False)
            .reset_index(drop=True)
        )
    return events


def news_to_events(news: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Übernimmt nur relevante, konkret klassifizierte Nachrichten in den Kalender."""
    if news is None or news.empty:
        return empty_events_frame()

    eligible = news.copy()
    if "is_relevant" in eligible.columns:
        eligible = eligible[eligible["is_relevant"].fillna(False)]
    if "event_type" in eligible.columns:
        eligible = eligible[eligible["event_type"].astype(str) != "news"]
    if eligible.empty:
        return empty_events_frame()

    events = pd.DataFrame(
        {
            "date": pd.to_datetime(eligible["published"], errors="coerce"),
            "ticker_yahoo": ticker,
            "event_type": eligible.get("event_type", "news"),
            "title": eligible.get("title", ""),
            "source": eligible.get("source", ""),
            "link": eligible.get("link", ""),
            "sentiment_score": eligible.get("sentiment_score", 0.0),
            "sentiment_label": eligible.get("sentiment_label", "neutral"),
            "importance": "mittel",
            "is_future_event": False,
        }
    )
    return events.dropna(subset=["date"]).reset_index(drop=True)


def persist_events(events: pd.DataFrame, replace_ticker: Optional[str] = None) -> tuple[bool, str]:
    """Speichert Kalenderdaten; bei Aktualisierung kann ein Ticker ersetzt werden.

    Das verhindert, dass alte, vor einer Alias-Korrektur gespeicherte
    Google-News-Ereignisse dauerhaft im Chart verbleiben.
    """
    if events is None:
        events = empty_events_frame()

    existing = empty_events_frame()
    if EVENTS_PATH.exists():
        try:
            existing = pd.read_csv(EVENTS_PATH)
        except Exception:
            existing = empty_events_frame()

    if replace_ticker:
        ticker = clean_ticker(replace_ticker)
        if not existing.empty and "ticker_yahoo" in existing.columns:
            existing["ticker_yahoo"] = existing["ticker_yahoo"].map(clean_ticker)
            existing = existing[existing["ticker_yahoo"] != ticker].copy()

    combined = pd.concat([existing, events], ignore_index=True)
    if combined.empty:
        return safe_write_csv(empty_events_frame(), EVENTS_PATH)

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined = (
        combined.drop_duplicates(["ticker_yahoo", "date", "event_type", "title"], keep="last")
        .sort_values(["ticker_yahoo", "date"], ascending=[True, False])
    )
    return safe_write_csv(combined, EVENTS_PATH)


def load_persisted_events(ticker: str, days_back: int = 1825) -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        return empty_events_frame()
    try:
        events = pd.read_csv(EVENTS_PATH)
    except Exception:
        return empty_events_frame()
    if events.empty or "ticker_yahoo" not in events.columns:
        return empty_events_frame()
    events["ticker_yahoo"] = events["ticker_yahoo"].map(clean_ticker)
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days_back)
    events = events[
        (events["ticker_yahoo"] == clean_ticker(ticker)) & (events["date"] >= cutoff)
    ].copy()
    for column, default in {
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "importance": "mittel",
        "is_future_event": False,
        "link": "",
        "source": "",
        "title": "",
        "event_type": "news",
    }.items():
        if column not in events.columns:
            events[column] = default
    return events.sort_values("date", ascending=False).reset_index(drop=True)


def combine_events(news: pd.DataFrame, calendar_events: pd.DataFrame, ticker: str) -> pd.DataFrame:
    combined = pd.concat([news_to_events(news, ticker), calendar_events], ignore_index=True)
    if combined.empty:
        return empty_events_frame()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined = (
        combined.drop_duplicates(["ticker_yahoo", "date", "event_type", "title"])
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    return combined


def empty_transactions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "ticker_yahoo", "type", "shares", "price", "currency", "fees", "comment"]
    )


def ensure_transaction_template() -> None:
    if TRANSACTIONS_PATH.exists():
        return
    safe_write_csv(empty_transactions_frame(), TRANSACTIONS_PATH)


def read_transactions() -> pd.DataFrame:
    ensure_transaction_template()
    try:
        transactions = pd.read_csv(TRANSACTIONS_PATH)
    except Exception as error:
        raise RuntimeError(f"data/transactions.csv konnte nicht gelesen werden: {error}") from error
    if transactions.empty:
        return empty_transactions_frame()

    required = {"date", "ticker_yahoo", "type", "shares", "price"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions.csv fehlt: {', '.join(sorted(missing))}")

    frame = transactions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker_yahoo"] = frame["ticker_yahoo"].map(clean_ticker)
    frame["type"] = frame["type"].astype(str).str.lower().str.strip()
    translations = {"kauf": "buy", "buy": "buy", "verkauf": "sell", "sell": "sell"}
    frame["type"] = frame["type"].map(translations)
    for column in ["shares", "price"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "fees" not in frame.columns:
        frame["fees"] = 0.0
    frame["fees"] = pd.to_numeric(frame["fees"], errors="coerce").fillna(0.0)
    if "currency" not in frame.columns:
        frame["currency"] = ""
    if "comment" not in frame.columns:
        frame["comment"] = ""
    return (
        frame.dropna(subset=["date", "ticker_yahoo", "type", "shares", "price"])
        .query("shares > 0")
        .sort_values(["ticker_yahoo", "date"])
        .reset_index(drop=True)
    )


def holdings_from_transactions(transactions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ermittelt Bestände nach gleitendem Durchschnittseinstand.

    Das ist bewusst eine Depotübersicht, keine steuerliche Gewinnermittlung.
    Mehrere Handelswährungen pro Ticker werden als Warnung ausgegeben.
    """
    if transactions is None or transactions.empty:
        return pd.DataFrame(columns=["ticker_yahoo", "shares", "cost_basis", "currency", "realized_pnl_local"]), []

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for ticker, group in transactions.groupby("ticker_yahoo", dropna=False):
        shares = 0.0
        cost_total = 0.0
        realized = 0.0
        currencies = {str(value).upper().strip() for value in group["currency"].dropna() if str(value).strip()}
        if len(currencies) > 1:
            warnings.append(f"{ticker}: mehrere Transaktionswährungen ({', '.join(sorted(currencies))})")
        last_currency = next(iter(currencies), "")

        for _, tx in group.sort_values("date").iterrows():
            quantity = safe_float(tx["shares"]) or 0.0
            price = safe_float(tx["price"]) or 0.0
            fees = safe_float(tx["fees"]) or 0.0
            if tx["type"] == "buy":
                shares += quantity
                cost_total += quantity * price + fees
            elif tx["type"] == "sell":
                if shares <= 0:
                    warnings.append(f"{ticker}: Verkauf ohne verfügbaren Bestand am {tx['date'].date()}")
                    continue
                sold = min(quantity, shares)
                if quantity > shares + 1e-9:
                    warnings.append(f"{ticker}: Verkaufsmenge übersteigt Bestand am {tx['date'].date()}")
                average_cost = cost_total / shares if shares else 0.0
                proceeds = sold * price - fees
                realized += proceeds - sold * average_cost
                cost_total -= sold * average_cost
                shares -= sold

        if shares > 1e-9:
            rows.append(
                {
                    "ticker_yahoo": ticker,
                    "shares": shares,
                    "cost_basis": cost_total / shares if shares else None,
                    "currency": last_currency or None,
                    "realized_pnl_local": realized,
                }
            )

    return pd.DataFrame(rows), warnings


def portfolio_input() -> tuple[pd.DataFrame, str, list[str]]:
    """Nutzt transactions.csv bevorzugt, sonst die bisherige portfolio.csv."""
    try:
        transactions = read_transactions()
    except Exception as error:
        return pd.DataFrame(), "Transaktionsbuch", [str(error)]

    if not transactions.empty:
        holdings, warnings = holdings_from_transactions(transactions)
        return holdings, "Transaktionsbuch", warnings

    try:
        holdings = read_portfolio()
    except Exception as error:
        return pd.DataFrame(), "portfolio.csv", [str(error)]
    return holdings, "portfolio.csv", []


def compute_return_metrics(series: pd.Series) -> dict[str, Optional[float]]:
    prices = pd.to_numeric(series, errors="coerce").dropna()
    if len(prices) < 3:
        return {"return_pct": None, "volatility_pct": None, "max_drawdown_pct": None}
    returns = prices.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    return {
        "return_pct": (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
        "volatility_pct": returns.std() * (252 ** 0.5) * 100 if len(returns) > 1 else None,
        "max_drawdown_pct": drawdown.min() * 100 if not drawdown.empty else None,
    }


def portfolio_risk_analysis(
    portfolio_view: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    risk_free_rate_pct: float = 2.0,
) -> tuple[dict[str, Optional[float]], pd.DataFrame, pd.DataFrame]:
    """Berechnet Risiko-Kennzahlen aus verfügbaren, währungslokalen Returns.

    Für Multiwährungsportfolios ist das eine Näherung, weil die täglichen FX-
    Bewegungen nicht in jeder Kursserie enthalten sind. Die aktuelle EUR-
    Gewichtung wird dennoch korrekt aus dem Portfolio verwendet.
    """
    if portfolio_view is None or portfolio_view.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    weight_map = (
        portfolio_view.dropna(subset=["ticker_yahoo", "weight_pct"])
        .set_index("ticker_yahoo")["weight_pct"]
        .astype(float)
        .div(100)
        .to_dict()
    )
    return_series: list[pd.Series] = []
    for ticker, weight in weight_map.items():
        history = histories.get(ticker, pd.DataFrame())
        if history is None or history.empty:
            continue
        price_column = "Adj Close" if "Adj Close" in history.columns else "Close"
        prices = pd.to_numeric(history[price_column], errors="coerce").dropna()
        if len(prices) < 30:
            continue
        returns = prices.pct_change().rename(ticker)
        return_series.append(returns)

    if not return_series:
        return {}, pd.DataFrame(), pd.DataFrame()

    matrix = pd.concat(return_series, axis=1).sort_index().dropna(how="all")
    available_tickers = [ticker for ticker in matrix.columns if ticker in weight_map]
    if not available_tickers:
        return {}, pd.DataFrame(), pd.DataFrame()

    weights = pd.Series({ticker: weight_map[ticker] for ticker in available_tickers}, dtype=float)
    weights = weights / weights.sum()
    matrix = matrix[available_tickers].fillna(0.0)
    portfolio_returns = matrix.mul(weights, axis=1).sum(axis=1)
    portfolio_index = (1 + portfolio_returns).cumprod() * 100
    drawdown = portfolio_index / portfolio_index.cummax() - 1

    periods = len(portfolio_returns)
    annual_return = ((1 + portfolio_returns).prod() ** (252 / periods) - 1) * 100 if periods else None
    annual_volatility = portfolio_returns.std() * (252 ** 0.5) * 100 if periods > 1 else None
    sharpe = None
    if annual_return is not None and annual_volatility not in (None, 0):
        sharpe = (annual_return - risk_free_rate_pct) / annual_volatility

    metrics = {
        "annual_return_pct": annual_return,
        "annual_volatility_pct": annual_volatility,
        "max_drawdown_pct": drawdown.min() * 100 if not drawdown.empty else None,
        "sharpe": sharpe,
        "observations": float(periods),
        "covered_weight_pct": weights.sum() * 100,
    }
    curve = pd.DataFrame({"date": portfolio_index.index, "portfolio_index": portfolio_index.values, "drawdown_pct": drawdown.values * 100})
    correlation = matrix.corr()
    return metrics, curve, correlation


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_benchmark_history(benchmark: str, period: str = "5y") -> pd.DataFrame:
    try:
        history = yf.Ticker(benchmark).history(period=period, auto_adjust=False)
    except Exception:
        history = pd.DataFrame()
    return ensure_datetime_index(history) if not history.empty else pd.DataFrame()



def render_company_profile(ticker: str) -> None:
    """Zeigt verfügbare Basisdaten aus Yahoo, ohne Vollständigkeit zu behaupten."""
    try:
        info = fetch_ticker_info(ticker)
    except Exception:
        info = {}
    if not info:
        st.info("Für dieses Unternehmen sind aktuell keine Profilinformationen verfügbar.")
        return

    st.markdown("#### Unternehmensprofil")
    left, right, third = st.columns(3)
    left.metric("Branche", str(info.get("industry") or "–"))
    right.metric("Land", str(info.get("country") or "–"))
    third.metric("Mitarbeitende", format_number(info.get("fullTimeEmployees"), 0))
    if info.get("website"):
        st.link_button("Unternehmenswebsite", str(info["website"]))

    summary = str(info.get("longBusinessSummary") or "").strip()
    if summary:
        st.caption(summary[:1800] + (" …" if len(summary) > 1800 else ""))

    ownership = pd.DataFrame(
        [
            {"Kennzahl": "Institutioneller Anteil", "Wert": format_percent(to_percent(info.get("heldPercentInstitutions")), 2)},
            {"Kennzahl": "Insider-Anteil", "Wert": format_percent(to_percent(info.get("heldPercentInsiders")), 2)},
            {"Kennzahl": "Analysten-Einstufung", "Wert": str(info.get("recommendationKey") or "–")},
            {"Kennzahl": "Analystenziel", "Wert": format_number(info.get("targetMeanPrice"), 2)},
        ]
    )
    st.dataframe(ownership, hide_index=True, use_container_width=True)

    officers = info.get("companyOfficers") or []
    if officers:
        officer_rows = []
        for officer in officers[:12]:
            if not isinstance(officer, dict):
                continue
            officer_rows.append(
                {
                    "Name": officer.get("name", ""),
                    "Funktion": officer.get("title", ""),
                    "Alter": officer.get("age", ""),
                }
            )
        if officer_rows:
            st.markdown("**Management (soweit von Yahoo geliefert)**")
            st.dataframe(pd.DataFrame(officer_rows), hide_index=True, use_container_width=True)

    st.caption(
        "Regionale Umsätze, Segmentumsätze und vollständige Ownership-Strukturen liefert Yahoo nicht zuverlässig. "
        "Dafür wäre später eine ergänzende, lizenzierte Quelle oder eine manuell gepflegte Profildatei nötig."
    )


def render_event_calendar(events: pd.DataFrame) -> None:
    st.markdown("#### Ereigniskalender")
    if events is None or events.empty:
        st.info("Noch keine Ereignisse gespeichert. Aktualisiere im News-Tab eine Aktie.")
        return

    display = events.copy()
    display["date"] = pd.to_datetime(display["date"], errors="coerce")
    display["Typ"] = display["event_type"].map(lambda value: EVENT_META.get(str(value), {}).get("label", str(value)))
    future_flags = display["is_future_event"].astype(str).str.lower().isin({"true", "1", "yes", "ja"})
    display["Zukunft"] = future_flags.map(lambda value: "Ja" if value else "Nein")
    visible = ["date", "ticker_yahoo", "Typ", "title", "source", "sentiment_label", "Zukunft", "link"]
    visible = [column for column in visible if column in display.columns]
    st.dataframe(
        display[visible].sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )








def render_portfolio_risk(portfolio_view: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> None:
    st.markdown("#### Portfolio-Risiko & Konzentration")
    risk_free_rate = st.slider("Risikofreier Zinssatz für Sharpe", 0.0, 8.0, 2.0, 0.25, key="risk_free_rate")
    metrics, curve, correlation = portfolio_risk_analysis(portfolio_view, histories, risk_free_rate)

    if not metrics:
        st.info("Für die Risikoanalyse fehlen ausreichend historische Kursdaten zu den Portfolio-Positionen.")
        return

    columns = st.columns(5)
    columns[0].metric("Ann. Rendite*", format_percent(metrics.get("annual_return_pct"), 1, signed=True))
    columns[1].metric("Ann. Volatilität*", format_percent(metrics.get("annual_volatility_pct"), 1))
    columns[2].metric("Max. Drawdown*", format_percent(metrics.get("max_drawdown_pct"), 1, signed=True))
    columns[3].metric("Sharpe*", format_number(metrics.get("sharpe"), 2))
    columns[4].metric("Datengewicht", format_percent(metrics.get("covered_weight_pct"), 0))

    if not curve.empty:
        curve_chart = alt.Chart(curve).mark_line().encode(
            x=alt.X("date:T", title="Datum"),
            y=alt.Y("portfolio_index:Q", title="Index (Start = 100)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T", format="%d.%m.%Y"), alt.Tooltip("portfolio_index:Q", format=".2f")],
        ).properties(height=240, title="Portfolioentwicklung aus historischen Tagesrenditen")
        st.altair_chart(curve_chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        sector = (
            portfolio_view.groupby("sector", dropna=False)["weight_pct"]
            .sum()
            .reset_index()
            .dropna(subset=["weight_pct"])
        )
        if not sector.empty:
            sector_chart = alt.Chart(sector).mark_bar().encode(
                x=alt.X("weight_pct:Q", title="Gewichtung (%)"),
                y=alt.Y("sector:N", sort="-x", title="Sektor"),
                tooltip=["sector:N", alt.Tooltip("weight_pct:Q", format=".2f")],
            ).properties(height=max(180, 32 * len(sector)), title="Sektor-Konzentration")
            st.altair_chart(sector_chart, use_container_width=True)

    with right:
        currencies = (
            portfolio_view.groupby("asset_currency", dropna=False)["weight_pct"]
            .sum()
            .reset_index()
            .dropna(subset=["weight_pct"])
        )
        if not currencies.empty:
            currency_chart = alt.Chart(currencies).mark_bar().encode(
                x=alt.X("weight_pct:Q", title="Gewichtung (%)"),
                y=alt.Y("asset_currency:N", sort="-x", title="Handelswährung"),
                tooltip=["asset_currency:N", alt.Tooltip("weight_pct:Q", format=".2f")],
            ).properties(height=max(180, 32 * len(currencies)), title="Währungs-Exponierung")
            st.altair_chart(currency_chart, use_container_width=True)

    if not correlation.empty and len(correlation.columns) >= 2:
        correlation_long = (
            correlation.reset_index()
            .melt(id_vars="index", var_name="Ticker 2", value_name="Korrelation")
            .rename(columns={"index": "Ticker 1"})
        )
        heatmap = alt.Chart(correlation_long).mark_rect().encode(
            x=alt.X("Ticker 1:N", title=None),
            y=alt.Y("Ticker 2:N", title=None),
            color=alt.Color("Korrelation:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue"), title="Korrelation"),
            tooltip=["Ticker 1:N", "Ticker 2:N", alt.Tooltip("Korrelation:Q", format=".2f")],
        ).properties(height=max(260, 25 * len(correlation.columns)), title="Korrelation der Tagesrenditen")
        st.altair_chart(heatmap, use_container_width=True)

    st.caption(
        "*Die Kennzahlen basieren auf verfügbaren Yahoo-Preisreihen (Adj Close, wenn vorhanden) und heutigen Gewichten. "
        "Bei mehreren Währungen ist dies eine Näherung ohne tägliche FX-Absicherung."
    )




def render_research(df: pd.DataFrame, histories: dict[str, pd.DataFrame], index_name: str) -> None:
    st.subheader("Research & historische Einordnung")
    st.caption(
        "Dieser Bereich vergleicht historische Kurs-/Adj-Close-Entwicklung mit einem Benchmark. "
        "Er ist bewusst kein fundamentaler Backtest: Historische Fundamentals, damalige Indexzusammensetzungen und Transaktionskosten fehlen dafür."
    )
    ticker = st.selectbox("Aktie für Vergleich", df["ticker_yahoo"].tolist(), key="research_ticker")
    use_adjusted = st.checkbox("Bereinigte Kurse verwenden (Adj Close)", value=True, key="research_adjusted")
    history = histories.get(ticker, pd.DataFrame())
    benchmark_ticker = benchmark_for_index(index_name)
    benchmark = fetch_benchmark_history(benchmark_ticker, period="5y")
    column = "Adj Close" if use_adjusted and "Adj Close" in history.columns else "Close"
    benchmark_column = "Adj Close" if use_adjusted and "Adj Close" in benchmark.columns else "Close"

    if history.empty or benchmark.empty or column not in history.columns or benchmark_column not in benchmark.columns:
        st.info("Für Aktie oder Benchmark fehlen ausreichende historische Daten.")
        return

    company_series = pd.to_numeric(history[column], errors="coerce").dropna().rename("Aktie")
    benchmark_series = pd.to_numeric(benchmark[benchmark_column], errors="coerce").dropna().rename("Benchmark")
    comparison = pd.concat([company_series, benchmark_series], axis=1, join="inner").dropna()
    if len(comparison) < 30:
        st.info("Zu wenige gemeinsame Handelstage für den Vergleich.")
        return

    indexed = comparison / comparison.iloc[0] * 100
    long = indexed.reset_index().melt(id_vars=indexed.index.name or "Date", var_name="Serie", value_name="Index")
    long = long.rename(columns={long.columns[0]: "Datum"})
    chart = alt.Chart(long).mark_line().encode(
        x=alt.X("Datum:T", title="Datum"),
        y=alt.Y("Index:Q", title="Index (Start = 100)", scale=alt.Scale(zero=False)),
        color=alt.Color("Serie:N", title=None),
        tooltip=[alt.Tooltip("Datum:T", format="%d.%m.%Y"), "Serie:N", alt.Tooltip("Index:Q", format=".2f")],
    ).properties(height=350)
    st.altair_chart(chart, use_container_width=True)

    company_metrics = compute_return_metrics(comparison["Aktie"])
    benchmark_metrics = compute_return_metrics(comparison["Benchmark"])
    table = pd.DataFrame(
        [
            {"Serie": ticker, "Rendite": company_metrics["return_pct"], "Volatilität": company_metrics["volatility_pct"], "Max. Drawdown": company_metrics["max_drawdown_pct"]},
            {"Serie": benchmark_ticker, "Rendite": benchmark_metrics["return_pct"], "Volatilität": benchmark_metrics["volatility_pct"], "Max. Drawdown": benchmark_metrics["max_drawdown_pct"]},
        ]
    )
    st.dataframe(
        table.style.format(
            {
                "Rendite": lambda value: format_percent(value, 2, signed=True),
                "Volatilität": lambda value: format_percent(value, 2),
                "Max. Drawdown": lambda value: format_percent(value, 2, signed=True),
            },
            na_rep="–",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Fallstudien-Checkliste")
    st.markdown(
        "Für ein Reverse Engineering wie „BAT bei 28 €“ solltest du zum jeweiligen Einstiegszeitpunkt "
        "manuell erfassen: Bewertung, Verschuldung, Ausschüttungsquote, Dividendenhistorie, Geschäftslage, "
        "Makroumfeld und den damals verfügbaren Nachrichtenstand. Erst mit historischen Fundamentals wäre daraus ein belastbarer Backtest möglich."
    )


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


def format_value_trigger(value: Any) -> str:
    """Macht den strengen Ja/Nein-Trigger in Tabellen eindeutig lesbar."""
    if value is None or pd.isna(value):
        return "–"
    return "✓ erfüllt" if bool(value) else "✗ offen"


def format_special_trigger(value: Any) -> str:
    if value is None or pd.isna(value):
        return "–"
    return "⚠ prüfen" if bool(value) else "–"


def colorize_value_trigger(value: Any) -> str:
    """Hebt nur wirklich ausgelöste Trigger grün hervor."""
    if value is None or pd.isna(value):
        return "color: #6b7280"
    if bool(value):
        return "background-color: #dcfce7; color: #166534; font-weight: 700"
    return "color: #9ca3af"


def colorize_special_trigger(value: Any) -> str:
    if value is None or pd.isna(value):
        return "color: #6b7280"
    if bool(value):
        return "background-color: #fef3c7; color: #92400e; font-weight: 700"
    return "color: #9ca3af"


def make_price_chart(
    history: pd.DataFrame,
    period_label: str,
    show_smas: bool,
    events: Optional[pd.DataFrame] = None,
    use_total_return: bool = False,
) -> Optional[alt.Chart]:
    if history is None or history.empty or "Close" not in history.columns:
        return None

    frame = ensure_datetime_index(history.copy())
    price_column = "Adj Close" if use_total_return and "Adj Close" in frame.columns else "Close"
    frame["Kurs"] = pd.to_numeric(frame[price_column], errors="coerce")
    frame = frame.dropna(subset=["Kurs"])
    if frame.empty:
        return None

    # SMAs stets auf vollständiger Historie berechnen, erst danach sichtbar filtern.
    frame["SMA 20"] = frame["Kurs"].rolling(20, min_periods=20).mean()
    frame["SMA 50"] = frame["Kurs"].rolling(50, min_periods=50).mean()
    frame["SMA 200"] = frame["Kurs"].rolling(200, min_periods=200).mean()

    days_by_period = {"2 Monate": 62, "6 Monate": 135, "1 Jahr": 260, "5 Jahre": 2000}
    visible = frame.tail(days_by_period.get(period_label, 260)).copy().reset_index()
    visible = visible.rename(columns={visible.columns[0]: "Datum"})

    title = "Bereinigter Kurs (Yahoo Adj Close)" if price_column == "Adj Close" else "Schlusskurs"
    base = alt.Chart(visible).mark_line(strokeWidth=2).encode(
        x=alt.X("Datum:T", title="Datum"),
        y=alt.Y("Kurs:Q", title=title, scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("Datum:T", format="%d.%m.%Y"),
            alt.Tooltip("Kurs:Q", title=title, format=".2f"),
        ],
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
                    y=alt.Y(f"{sma_name}:Q", title=title, scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "Linie:N",
                        scale=alt.Scale(domain=list(SMA_COLORS), range=list(SMA_COLORS.values())),
                        legend=alt.Legend(title="Durchschnitte", orient="bottom"),
                    ),
                    tooltip=[
                        alt.Tooltip("Datum:T", format="%d.%m.%Y"),
                        alt.Tooltip(f"{sma_name}:Q", format=".2f"),
                    ],
                )
            )

    if events is not None and not events.empty and "date" in events.columns:
        # `merge_asof` verlangt auf beiden Seiten exakt denselben Datetime-Datentyp.
        # Ereignisse aus CSV/RSS können unter Windows/Pandas als datetime64[us]
        # ankommen, während Yahoo-Kurse datetime64[ns] verwenden. Deshalb werden
        # beide Seiten explizit nach UTC konvertiert, zeitzonenfrei gemacht,
        # auf Tagesebene normalisiert und als datetime64[ns] gespeichert.
        def normalize_chart_dates(values: pd.Series) -> pd.Series:
            normalized = pd.to_datetime(values, errors="coerce", utc=True)
            normalized = normalized.dt.tz_convert(None).dt.normalize()
            return normalized.astype("datetime64[ns]")

        visible["Datum"] = normalize_chart_dates(visible["Datum"])
        visible = visible.dropna(subset=["Datum"]).sort_values("Datum").reset_index(drop=True)

        event_frame = events.copy()
        event_frame["Datum"] = normalize_chart_dates(event_frame["date"])
        event_frame = event_frame.dropna(subset=["Datum"])
        event_frame = event_frame[
            event_frame["Datum"].between(visible["Datum"].min(), visible["Datum"].max())
        ]

        if not event_frame.empty and not visible.empty:
            event_frame["event_type"] = event_frame["event_type"].fillna("news")
            event_frame["sentiment_label"] = event_frame["sentiment_label"].fillna("neutral")
            price_lookup = (
                visible[["Datum", "Kurs"]]
                .dropna(subset=["Datum", "Kurs"])
                .drop_duplicates(subset=["Datum"], keep="last")
                .sort_values("Datum")
                .astype({"Datum": "datetime64[ns]"})
            )
            event_frame = (
                event_frame.sort_values("Datum")
                .reset_index(drop=True)
                .astype({"Datum": "datetime64[ns]"})
            )

            try:
                event_frame = pd.merge_asof(
                    event_frame,
                    price_lookup,
                    on="Datum",
                    direction="nearest",
                )
            except (TypeError, ValueError, pd.errors.MergeError):
                # Fallback: Ein Chart bleibt auch dann funktionsfähig, wenn
                # eine externe Quelle ungewöhnliche Datumswerte liefert.
                price_series = price_lookup.set_index("Datum")["Kurs"]
                event_frame["Kurs"] = price_series.reindex(
                    event_frame["Datum"], method="nearest"
                ).to_numpy()

            event_frame = event_frame.dropna(subset=["Kurs"])
            if not event_frame.empty:
                layers.append(
                    alt.Chart(event_frame).mark_point(filled=True, size=85, opacity=0.9).encode(
                        x="Datum:T",
                        y=alt.Y("Kurs:Q", scale=alt.Scale(zero=False)),
                        shape=alt.Shape(
                            "event_type:N",
                            scale=alt.Scale(
                                domain=list(EVENT_META),
                                range=[EVENT_META[item]["shape"] for item in EVENT_META],
                            ),
                            legend=alt.Legend(title="Ereignistyp", orient="bottom"),
                        ),
                        color=alt.Color(
                            "sentiment_label:N",
                            scale=alt.Scale(
                                domain=["positiv", "neutral", "negativ"],
                                range=["#16a34a", "#64748b", "#dc2626"],
                            ),
                            legend=alt.Legend(title="News-Sentiment", orient="bottom"),
                        ),
                        tooltip=[
                            alt.Tooltip("Datum:T", format="%d.%m.%Y"),
                            "event_type:N", "title:N", "source:N", "sentiment_label:N", "link:N",
                        ],
                    )
                )

    return alt.layer(*layers).resolve_scale(y="shared").properties(height=380)


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
        f"Version {APP_VERSION} · Fundamentaldaten, Score-Profile, News & Events, Portfolio-Risiko und Research. "
        "Keine Anlageberatung – Daten, Quellen und Annahmen vor Entscheidungen immer selbst prüfen."
    )


def render_data_status(summary: dict[str, Any], detail: pd.DataFrame, metrics: pd.DataFrame) -> None:
    st.subheader("Datenstatus")
    st.caption(
        "Hier siehst du, ob der gewählte Index vollständig geladen wurde und welche Datenbereiche je Aktie fehlen."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Index", str(summary.get("index", "–")))
    col2.metric("Angefragt", int(summary.get("angefragt", 0)))
    col3.metric("Analysiert", int(summary.get("analysiert", 0)))
    col4.metric("Mit Kursdaten", int(summary.get("mit_kursdaten", 0)))
    avg_cov = safe_float(summary.get("durchschnitt_datenabdeckung"))
    col5.metric("Ø Datenabdeckung", format_percent(avg_cov, 1) if avg_cov is not None else "–")

    if detail is None or detail.empty:
        st.info("Noch keine Statusdetails vorhanden.")
        return

    problems = detail[detail["Status"] != "OK"].copy()
    if problems.empty:
        st.success("Alle aktuell angefragten Werte wurden ohne offensichtliche Datenprobleme verarbeitet.")
    else:
        st.warning(f"{len(problems)} Werte haben fehlende Kursdaten oder größere Datenlücken.")
        st.dataframe(
            problems[["Status", "Name", "Ticker", "Sektor", "Kursdaten", "Datenabdeckung", "Fehler/Hinweis"]].style.format(
                {"Datenabdeckung": lambda value: format_percent(value, 1)},
                na_rep="–",
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Abdeckung je Datenbereich")
    group_cols = list(DATA_STATUS_FIELD_GROUPS.keys())
    coverage_by_group = []
    for group in group_cols:
        coverage_by_group.append({
            "Datenbereich": group,
            "Durchschnittliche Abdeckung": safe_float(detail[group].mean()) if group in detail.columns else None,
        })
    group_df = pd.DataFrame(coverage_by_group)
    st.dataframe(
        group_df.style.format(
            {"Durchschnittliche Abdeckung": lambda value: format_percent(value, 1)},
            na_rep="–",
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart = alt.Chart(group_df.dropna()).mark_bar().encode(
        x=alt.X("Durchschnittliche Abdeckung:Q", title="Abdeckung in %", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Datenbereich:N", sort="-x", title="Datenbereich"),
        tooltip=["Datenbereich:N", alt.Tooltip("Durchschnittliche Abdeckung:Q", format=".1f")],
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Alle Werte im Detail"):
        display_cols = [
            "Status", "Name", "Ticker", "Sektor", "Kursdaten", "Historie Tage", "Datenabdeckung",
            "Kursdaten", "Bewertung", "Qualität", "Dividende", "Cashflow", "Deep Value", "Fehler/Hinweis",
        ]
        display_cols = [col for col in display_cols if col in detail.columns]
        st.dataframe(
            detail[display_cols].style.format(
                {
                    "Datenabdeckung": lambda value: format_percent(value, 1),
                    "Bewertung": lambda value: format_percent(value, 1),
                    "Qualität": lambda value: format_percent(value, 1),
                    "Dividende": lambda value: format_percent(value, 1),
                    "Cashflow": lambda value: format_percent(value, 1),
                    "Deep Value": lambda value: format_percent(value, 1),
                },
                na_rep="–",
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Warum ist dieser Tab wichtig?"):
        st.write(
            "Viele Fehler im Tool entstehen nicht im UI, sondern durch fehlende oder unvollständige externe Daten. "
            "Der Datenstatus trennt deshalb klar zwischen Indexbestandteil, Kursdaten, Fundamentaldaten, Dividenden, "
            "Cashflow und Deep-Value-Kennzahlen. Wenn ein Wert nicht im Scanner auftaucht, findest du hier meistens den Grund."
        )


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Überblick")
    if df.empty:
        st.info("Keine Daten für die aktuelle Filtereinstellung.")
        return

    metric_columns = st.columns(6)
    metric_columns[0].metric("Unternehmen", f"{len(df):,}".replace(",", "."))
    metric_columns[1].metric("Value-Trigger", int(df["value_trigger"].fillna(False).sum()))
    metric_columns[2].metric("Deep-Value prüfen", int(df.get("special_situation_trigger", pd.Series(dtype=bool)).fillna(False).sum()))
    metric_columns[3].metric("Ø Qualitäts-Score", format_number(df["total_score"].mean(), 1))
    metric_columns[4].metric("Ø Kursrendite 1J", format_percent(df["change_1y"].mean(), 1, signed=True))
    metric_columns[5].metric("Ø Gesamtrendite 1J*", format_percent(df["total_return_1y"].mean(), 1, signed=True))

    columns = [
        "name", "ticker_yahoo", "sector", "currency", "last_price", "change_1d", "change_5d",
        "change_1y", "total_return_1y", "vol_1y", "max_drawdown_1y", "dividend_yield",
        "total_score", "score_coverage", "value_score", "special_situation_score",
        "special_situation_trigger", "drawdown_1y_high_pct", "drawdown_3y_high_pct", "value_trigger",
    ]
    visible_columns = [column for column in columns if column in df.columns]
    display = df[visible_columns].rename(
        columns={
            "name": "Unternehmen",
            "ticker_yahoo": "Ticker",
            "sector": "Sektor",
            "currency": "Währung",
            "last_price": "Letzter Kurs",
            "change_1d": "Veränderung 1T",
            "change_5d": "Veränderung 5T",
            "change_1y": "Kursrendite 1J",
            "total_return_1y": "Gesamtrendite 1J*",
            "vol_1y": "Volatilität 1J",
            "max_drawdown_1y": "Max. Drawdown 1J",
            "dividend_yield": "Dividendenrendite",
            "total_score": "Qualitäts-Score",
            "score_coverage": "Datenabdeckung",
            "value_score": "Value-Score",
            "special_situation_score": "Deep-Value-Score",
            "special_situation_trigger": "Deep-Value",
            "drawdown_1y_high_pct": "Drawdown vom 52W-Hoch",
            "drawdown_3y_high_pct": "Drawdown vom 3J-Hoch",
            "value_trigger": "Value-Trigger",
        }
    )
    formats = {
        "Letzter Kurs": lambda value: format_number(value, 2),
        "Veränderung 1T": lambda value: format_percent(value, 2, signed=True),
        "Veränderung 5T": lambda value: format_percent(value, 2, signed=True),
        "Kursrendite 1J": lambda value: format_percent(value, 2, signed=True),
        "Gesamtrendite 1J*": lambda value: format_percent(value, 2, signed=True),
        "Volatilität 1J": lambda value: format_percent(value, 1),
        "Max. Drawdown 1J": lambda value: format_percent(value, 1, signed=True),
        "Dividendenrendite": lambda value: format_percent(value, 2),
        "Qualitäts-Score": lambda value: format_number(value, 1),
        "Datenabdeckung": lambda value: format_percent(value, 0),
        "Value-Score": lambda value: format_number(value, 1),
        "Deep-Value-Score": lambda value: format_number(value, 1),
        "Deep-Value": format_special_trigger,
        "Drawdown vom 52W-Hoch": lambda value: format_percent(value, 1, signed=True),
        "Drawdown vom 3J-Hoch": lambda value: format_percent(value, 1, signed=True),
        "Value-Trigger": format_value_trigger,
    }
    styled = (
        display.style.format({key: value for key, value in formats.items() if key in display.columns}, na_rep="–")
        .map(colorize_change, subset=[column for column in ["Veränderung 1T", "Veränderung 5T", "Kursrendite 1J", "Gesamtrendite 1J*"] if column in display.columns])
        .map(colorize_score, subset=[column for column in ["Qualitäts-Score", "Deep-Value-Score"] if column in display.columns])
        .map(colorize_special_trigger, subset=[column for column in ["Deep-Value"] if column in display.columns])
        .map(colorize_value_trigger, subset=[column for column in ["Value-Trigger"] if column in display.columns])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "Value-Score und Value-Trigger sind verschieden: Der Score ordnet Kandidaten ein; der Trigger ist ein strenger Filter, "
        "bei dem alle Regeln erfüllt sein müssen. Der Deep-Value-Score sucht zusätzlich nach BAT-ähnlichen Sondersituationen."
    )
    st.caption("*Gesamtrendite = Yahoo Adj Close als Näherung; sie kann Dividenden und Splits abbilden, ist aber keine garantierte steuer- oder brokeridentische Rendite.")


def render_fundamentals(df: pd.DataFrame) -> None:
    st.subheader("Fundamentaldaten")
    columns = [
        "name", "ticker_yahoo", "currency", "market_cap", "pe_ratio", "forward_pe", "pb_ratio",
        "ps_ratio", "ev_ebitda", "net_margin", "operating_margin", "roe", "roa", "dividend_yield",
        "dividend_per_share", "payout_ratio", "dividend_growth_5y", "dividend_frequency",
        "debt_to_equity", "net_debt_ebitda", "beta", "score_raw", "score_coverage", "score_confidence",
        "total_score",
    ]
    visible_columns = [column for column in columns if column in df.columns]
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
        "dividend_yield_5y_avg": "Ø Div.-Rendite 5J",
        "dividend_yield_vs_5y_avg_pct": "Rendite vs. 5J-Schnitt",
        "free_cashflow": "Free Cashflow",
        "operating_cashflow": "Operativer Cashflow",
        "cashflow_dividend_coverage": "FCF/Dividende",
        "debt_to_equity": "Debt/Equity",
        "net_debt_ebitda": "Netto-Schulden/EBITDA",
        "beta": "Beta",
        "score_raw": "Rohscore",
        "score_coverage": "Datenabdeckung",
        "score_confidence": "Datenvertrauen",
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
        "Ø Div.-Rendite 5J": lambda value: format_percent(value, 2),
        "Rendite vs. 5J-Schnitt": lambda value: format_percent(value, 0, signed=True),
        "Free Cashflow": human_market_cap,
        "Operativer Cashflow": human_market_cap,
        "FCF/Dividende": lambda value: f"{format_number(value, 2)}x",
        "Debt/Equity": lambda value: f"{format_number(value, 2)}x",
        "Netto-Schulden/EBITDA": lambda value: f"{format_number(value, 2)}x",
        "Beta": lambda value: format_number(value, 2),
        "Rohscore": lambda value: format_number(value, 1),
        "Datenabdeckung": lambda value: format_percent(value, 0),
        "Qualitäts-Score": lambda value: format_number(value, 1),
    }
    styled = display_df.style.format(formats, na_rep="–").map(colorize_score, subset=["Qualitäts-Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "Die Marktkapitalisierung bezieht sich auf die Originalwährung. "
        "Negative KGVs werden im Score nicht als günstig bewertet. Datenabdeckung und Datenvertrauen zeigen, "
        "wie belastbar eine heuristische Einstufung ist."
    )


def render_value_trigger_explanation(
    df: pd.DataFrame,
    drawdown_trigger: float,
    payout_max: float,
    score_min: float,
    yield_min: float,
    profile_name: str,
) -> None:
    """Erklärt die aktuelle strenge Trigger-Logik und prüft einen Titel einzeln."""
    with st.expander("So funktioniert der Value-Trigger", expanded=False):
        st.markdown(
            "Der **Value-Trigger** ist ein strenger Ja/Nein-Filter – kein Kaufsignal. Er wird nur ausgelöst, "
            "wenn eine Aktie **alle fünf** Bedingungen des aktuell gewählten Scanner-Profils erfüllt. "
            "Der **Value-Score** dagegen ist eine Rangfolge von 0 bis 100 und kann auch hoch sein, wenn der Trigger noch offen ist."
        )

        sector_default_note = (
            "Die Mindest-Dividendenrendite ist je nach Sektor unterschiedlich: Es gilt immer der höhere Wert aus "
            f"deinem Slider ({format_percent(yield_min, 1)}) und dem Sektor-Mindestwert."
        )
        rules = pd.DataFrame([
            {"Kriterium": "Qualitäts-Score", "Aktuelle Regel": f"mindestens {format_number(score_min, 0)} Punkte", "Warum": "Grundqualität und Stabilität müssen ausreichend sein."},
            {"Kriterium": "Datenabdeckung", "Aktuelle Regel": f"mindestens {format_percent(MIN_SCORE_COVERAGE_FOR_TRIGGER, 0)}", "Warum": "Ein guter Score soll auf genügend Kennzahlen beruhen."},
            {"Kriterium": "Ausschüttungsquote", "Aktuelle Regel": f"0 % bis {format_percent(payout_max, 0)}", "Warum": "Eine Dividende soll aus heutiger Sicht nicht zu stark überzogen wirken."},
            {"Kriterium": "Dividendenrendite", "Aktuelle Regel": "Sektor- bzw. Slider-Mindestwert", "Warum": "Der Scanner sucht bewusst ertragsorientierte Value-Kandidaten."},
            {"Kriterium": "Drawdown", "Aktuelle Regel": f"mindestens {format_percent(-drawdown_trigger, 0, signed=True)} unter dem 52W-Hoch", "Warum": "Es muss ein klarer Rücksetzer gegenüber dem jüngsten Hoch vorliegen."},
        ])
        st.dataframe(rules, use_container_width=True, hide_index=True)
        st.caption(sector_default_note)

        st.markdown("**Wie entsteht der Value-Score?**")
        st.markdown(
            "Er bewertet Kandidaten unabhängig vom strengen Trigger: **Drawdown 35 %**, **Dividendenrendite 20 %**, "
            "**KGV oder KBV 20 %**, **Ausschüttungsquote 15 %** und – außerhalb von Finanzsektoren – **Verschuldung 10 %**. "
            "Fehlende Werte senken die Datenabdeckung und dämpfen den Score. Ein negatives KGV zählt dabei nicht als günstig."
        )

        if df.empty:
            return
        ticker = st.selectbox("Trigger für einen Titel prüfen", df["ticker_yahoo"].tolist(), key="value_trigger_explanation_ticker")
        row = df.loc[df["ticker_yahoo"] == ticker].iloc[0]
        minimum_yield = sector_yield_trigger(row.get("sector"), yield_min)
        drawdown = safe_float(row.get("drawdown_1y_high_pct"))
        quality_score = safe_float(row.get("total_score"))
        coverage = safe_float(row.get("score_coverage"))
        payout = safe_float(row.get("payout_ratio"))
        dividend_yield = safe_float(row.get("dividend_yield"))

        checks = [
            ("Qualitäts-Score", quality_score, f"≥ {format_number(score_min, 0)}", quality_score is not None and quality_score >= score_min),
            ("Datenabdeckung", coverage, f"≥ {format_percent(MIN_SCORE_COVERAGE_FOR_TRIGGER, 0)}", coverage is not None and coverage >= MIN_SCORE_COVERAGE_FOR_TRIGGER),
            ("Ausschüttungsquote", payout, f"0 % bis {format_percent(payout_max, 0)}", payout is not None and 0 <= payout <= payout_max),
            ("Dividendenrendite", dividend_yield, f"≥ {format_percent(minimum_yield, 1)}", dividend_yield is not None and dividend_yield >= minimum_yield),
            ("Drawdown vom 52W-Hoch", drawdown, f"≤ {format_percent(-drawdown_trigger, 0, signed=True)}", drawdown is not None and drawdown <= -drawdown_trigger),
        ]
        detail = pd.DataFrame([
            {
                "Kriterium": label,
                "Aktuell": format_percent(value, 1, signed=(label == "Drawdown vom 52W-Hoch")) if label in {"Ausschüttungsquote", "Dividendenrendite", "Datenabdeckung", "Drawdown vom 52W-Hoch"} else format_number(value, 1),
                "Regel": rule,
                "Status": "✓ erfüllt" if passed else "✗ offen",
            }
            for label, value, rule, passed in checks
        ])
        st.markdown(f"**Einzelprüfung: {row.get('name', ticker)} ({ticker})**")
        st.dataframe(detail, use_container_width=True, hide_index=True)
        if bool(row.get("value_trigger", False)):
            st.success("Der Value-Trigger ist aktuell erfüllt. Das ist eine Vorauswahl nach Regeln, keine Kaufempfehlung.")
        else:
            st.info("Der Value-Trigger ist noch nicht erfüllt. In der Tabelle siehst du, welches Kriterium fehlt oder welche Kennzahl nicht vorliegt.")
        st.caption(str(row.get("value_reason", "")))


def render_special_situation_scanner(df: pd.DataFrame) -> None:
    st.subheader("Sondersituation / Deep Value")
    st.caption(
        "Dieser Scanner sucht nach BAT-ähnlichen Situationen: hohe Dividendenrendite, großer Mehrjahres-Drawdown "
        "und trotzdem positive Cashflows. Das ist ein Research-Hinweis, keine Kaufempfehlung."
    )

    with st.expander("So funktioniert der BAT-/Deep-Value-Pattern-Score", expanded=True):
        st.markdown(
            "Der Score bewertet Fälle, bei denen normale Kennzahlen wie KGV oder Payout Ratio durch Sondereffekte verzerrt sein können. "
            "Das Muster lautet: **hohe Rendite + starker Drawdown + Cashflow bleibt positiv + Dividende wirkt gedeckt + Verschuldung kontrollierbar**."
        )
        rules = pd.DataFrame([
            {"Baustein": "Dividendenrendite", "Punkte": "bis 20", "Gedanke": "> 7 % ist ein starkes Ertragssignal, kann aber auch Dividendenfalle sein."},
            {"Baustein": "Drawdown vom 3J-/5J-Hoch", "Punkte": "bis 20", "Gedanke": "Der Markt hat die Aktie deutlich abgestraft."},
            {"Baustein": "Positiver Cashflow", "Punkte": "15", "Gedanke": "Wichtigster Unterschied zwischen Sondersituation und operativem Kollaps."},
            {"Baustein": "Dividende durch FCF gedeckt", "Punkte": "bis 15", "Gedanke": "Free Cashflow geteilt durch geschätzte jährliche Dividendenlast."},
            {"Baustein": "Net Debt / EBITDA", "Punkte": "bis 10", "Gedanke": "Unter 3,5x wird als kontrollierbarer gewertet."},
            {"Baustein": "KGV oder EPS-Verzerrung", "Punkte": "bis 10", "Gedanke": "Negatives KGV ist kein Ausschluss, wenn Cashflow positiv bleibt."},
            {"Baustein": "Negatives News-Sentiment", "Punkte": "bis 10", "Gedanke": "Punkte nur, wenn News geladen wurden und der Cashflow nicht kollabiert."},
        ])
        st.dataframe(rules, use_container_width=True, hide_index=True)
        st.info(
            "Ein Trigger bedeutet: **Bitte prüfen** – nicht kaufen. Besonders wichtig ist die Frage: "
            "Ist der Gewinnrückgang operativ, oder wurde das Ergebnis durch Abschreibungen/Sondereffekte verzerrt?"
        )

    if df.empty:
        st.info("Keine Daten für die aktuelle Filtereinstellung.")
        return

    required = ["special_situation_score", "special_situation_trigger"]
    if not all(column in df.columns for column in required):
        df = enrich_with_special_situations(df)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Prüfen", int(df["special_situation_trigger"].fillna(False).sum()))
    metric_cols[1].metric("Score ≥ 70", int((pd.to_numeric(df["special_situation_score"], errors="coerce") >= 70).sum()))
    metric_cols[2].metric("Ø Deep-Value-Score", format_number(pd.to_numeric(df["special_situation_score"], errors="coerce").mean(), 1))
    metric_cols[3].metric("Ø Drawdown 3/5J", format_percent(pd.to_numeric(df.get("deepest_drawdown_3_5y_pct"), errors="coerce").mean(), 1, signed=True))

    sorted_df = df.sort_values(["special_situation_trigger", "special_situation_score"], ascending=[False, False], na_position="last")
    columns = [
        "name", "ticker_yahoo", "sector", "special_situation_score", "special_situation_trigger",
        "special_situation_status", "dividend_yield", "dividend_yield_5y_avg", "dividend_yield_vs_5y_avg_pct",
        "deepest_drawdown_3_5y_pct", "free_cashflow", "cashflow_dividend_coverage",
        "net_debt_ebitda", "pe_ratio", "payout_ratio", "special_situation_reason",
    ]
    visible = [column for column in columns if column in sorted_df.columns]
    display = sorted_df[visible].rename(columns={
        "name": "Unternehmen",
        "ticker_yahoo": "Ticker",
        "sector": "Sektor",
        "special_situation_score": "BAT-/Deep-Value-Score",
        "special_situation_trigger": "Prüfen",
        "special_situation_status": "Status",
        "dividend_yield": "Dividendenrendite",
        "dividend_yield_5y_avg": "Ø Div.-Rendite 5J",
        "dividend_yield_vs_5y_avg_pct": "Rendite vs. 5J-Schnitt",
        "deepest_drawdown_3_5y_pct": "Drawdown 3/5J",
        "free_cashflow": "Free Cashflow",
        "cashflow_dividend_coverage": "FCF/Dividende",
        "net_debt_ebitda": "Net Debt/EBITDA",
        "pe_ratio": "KGV",
        "payout_ratio": "Payout",
        "special_situation_reason": "Begründung",
    })
    formats = {
        "BAT-/Deep-Value-Score": lambda value: format_number(value, 1),
        "Prüfen": format_special_trigger,
        "Dividendenrendite": lambda value: format_percent(value, 2),
        "Ø Div.-Rendite 5J": lambda value: format_percent(value, 2),
        "Rendite vs. 5J-Schnitt": lambda value: format_percent(value, 0, signed=True),
        "Drawdown 3/5J": lambda value: format_percent(value, 1, signed=True),
        "Free Cashflow": human_market_cap,
        "FCF/Dividende": lambda value: f"{format_number(value, 2)}x",
        "Net Debt/EBITDA": lambda value: f"{format_number(value, 2)}x",
        "KGV": lambda value: format_number(value, 2),
        "Payout": lambda value: format_percent(value, 1),
    }
    styled = (
        display.style.format({key: value for key, value in formats.items() if key in display.columns}, na_rep="–")
        .map(colorize_score, subset=[column for column in ["BAT-/Deep-Value-Score"] if column in display.columns])
        .map(colorize_special_trigger, subset=[column for column in ["Prüfen"] if column in display.columns])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("### Einzelprüfung")
    ticker = st.selectbox("Titel prüfen", sorted_df["ticker_yahoo"].tolist(), key="special_situation_detail_ticker")
    row = sorted_df.loc[sorted_df["ticker_yahoo"] == ticker].iloc[0]
    detail_cols = st.columns(4)
    detail_cols[0].metric("Score", format_number(row.get("special_situation_score"), 1))
    detail_cols[1].metric("Dividendenrendite", format_percent(row.get("dividend_yield"), 2))
    detail_cols[2].metric("Drawdown 3/5J", format_percent(row.get("deepest_drawdown_3_5y_pct"), 1, signed=True))
    detail_cols[3].metric("FCF/Dividende", f"{format_number(row.get('cashflow_dividend_coverage'), 2)}x")

    if bool(row.get("special_situation_trigger", False)):
        st.warning(
            "Sondersituation erkannt: hohe Dividende + starker Drawdown + positive Cashflow-Indizien. "
            "Bitte prüfen, ob der Gewinnrückgang operativ oder bilanziell/Sondereffekt-getrieben ist."
        )
    elif safe_float(row.get("special_situation_score")) is not None and safe_float(row.get("special_situation_score")) >= 55:
        st.info("Teilweise BAT-ähnliches Profil. Noch fehlen aber ein oder mehrere harte Kriterien für den Prüf-Trigger.")
    else:
        st.info("Aktuell kein klares BAT-/Deep-Value-Muster nach den hinterlegten Regeln.")

    st.markdown("**Score-Bausteine**")
    st.code(str(row.get("special_situation_reason", "")))
    st.markdown("**Zusatzhinweise**")
    st.code(str(row.get("special_situation_checks", "")))

    with st.expander("BAT-Fallstudie als Denkmodell", expanded=False):
        st.markdown(
            "BAT war ein Beispiel für eine Aktie, bei der der Markt stark auf strukturelle Risiken und eine große Abschreibung reagiert hat, "
            "während Cashflow und Dividendenfähigkeit auf adjustierter Basis nicht im selben Maß kollabierten. Genau solche Spannungen soll dieser Scanner sichtbar machen."
        )
        st.markdown(
            "Für eine belastbare Replikation fehlen weiterhin historische Fundamentaldaten zum damaligen Zeitpunkt. "
            "Der Scanner ist deshalb ein **Hinweisgeber**, kein Backtest und keine automatische Entscheidung."
        )


def render_value_watchlist(
    df: pd.DataFrame,
    drawdown_trigger: float,
    payout_max: float,
    score_min: float,
    yield_min: float,
    profile_name: str,
) -> None:
    st.subheader("Dividenden-Value-Scanner")
    st.caption("Der Trigger ist ein strenger Filter. Value-Score und Trigger sind keine Anlageberatung und keine Kaufempfehlung.")
    render_value_trigger_explanation(
        df=df,
        drawdown_trigger=drawdown_trigger,
        payout_max=payout_max,
        score_min=score_min,
        yield_min=yield_min,
        profile_name=profile_name,
    )

    columns = [
        "name", "ticker_yahoo", "sector", "total_score", "score_coverage", "value_score", "value_coverage",
        "value_status", "value_trigger", "dividend_yield", "payout_ratio", "pe_ratio", "pb_ratio",
        "drawdown_1y_high_pct", "value_reason",
    ]
    visible_columns = [column for column in columns if column in df.columns]
    result = df.sort_values(["value_trigger", "value_score"], ascending=[False, False], na_position="last")
    display = result[visible_columns].rename(columns={
        "name": "Unternehmen",
        "ticker_yahoo": "Ticker",
        "sector": "Sektor",
        "total_score": "Qualitäts-Score",
        "score_coverage": "Datenabdeckung",
        "value_score": "Value-Score",
        "value_coverage": "Value-Daten",
        "value_status": "Trigger-Status",
        "value_trigger": "Value-Trigger",
        "dividend_yield": "Dividendenrendite",
        "payout_ratio": "Ausschüttungsquote",
        "pe_ratio": "KGV",
        "pb_ratio": "KBV",
        "drawdown_1y_high_pct": "Drawdown vom 52W-Hoch",
        "value_reason": "Begründung",
    })
    formats = {
        "Qualitäts-Score": lambda value: format_number(value, 1),
        "Datenabdeckung": lambda value: format_percent(value, 0),
        "Value-Score": lambda value: format_number(value, 1),
        "Value-Daten": lambda value: format_percent(value, 0),
        "Value-Trigger": format_value_trigger,
        "Dividendenrendite": lambda value: format_percent(value, 2),
        "Ausschüttungsquote": lambda value: format_percent(value, 2),
        "KGV": lambda value: format_number(value, 2),
        "KBV": lambda value: format_number(value, 2),
        "Drawdown vom 52W-Hoch": lambda value: format_percent(value, 1, signed=True),
    }
    styled = (
        display.style.format({key: value for key, value in formats.items() if key in display.columns}, na_rep="–")
        .map(colorize_score, subset=["Qualitäts-Score"])
        .map(colorize_value_trigger, subset=["Value-Trigger"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_risk_and_chart(df: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> None:
    st.subheader("Einzelanalyse & Chart")
    ticker = st.selectbox("Aktie auswählen", df["ticker_yahoo"].tolist(), key="analysis_ticker")
    row = df.loc[df["ticker_yahoo"] == ticker].iloc[0]
    events = load_persisted_events(ticker, days_back=1825)

    columns = st.columns(6)
    columns[0].metric("Letzter Kurs", f"{format_number(row.get('last_price'), 2)} {row.get('currency', '')}")
    columns[1].metric("Kursrendite 1J", format_percent(row.get("change_1y"), 1, signed=True))
    columns[2].metric("Gesamtrendite 1J*", format_percent(row.get("total_return_1y"), 1, signed=True))
    columns[3].metric("Volatilität 1J", format_percent(row.get("vol_1y"), 1))
    columns[4].metric("Max. Drawdown 1J", format_percent(row.get("max_drawdown_1y"), 1, signed=True))
    columns[5].metric("Qualitäts-Score", format_number(row.get("total_score"), 1))

    left, right = st.columns([2, 1])
    with right:
        st.markdown("**Score-Komponenten**")
        st.caption(str(row.get("score_components", "–")))
        st.markdown("**Value-Check**")
        st.caption(str(row.get("value_reason", "–")))
        st.markdown("**Datenvertrauen**")
        st.caption(f"{row.get('score_confidence', '–')} · Abdeckung: {format_percent(row.get('score_coverage'), 0)}")
        if st.button("Zur Watchlist hinzufügen", key=f"watch_{ticker}"):
            if add_to_watchlist(ticker, str(row.get("name", ticker)), str(row.get("sector", "Unbekannt"))):
                st.success(f"{ticker} wurde gespeichert.")
            else:
                st.info(f"{ticker} ist bereits auf der Watchlist.")

        with st.expander("Unternehmensprofil & Management"):
            render_company_profile(ticker)

    with left:
        period = st.selectbox("Chart-Zeitraum", ["2 Monate", "6 Monate", "1 Jahr", "5 Jahre"], index=2)
        show_smas = st.checkbox("SMA 20 / 50 / 200 anzeigen", value=True)
        use_total_return = st.checkbox(
            "Bereinigten Kurs verwenden (Adj Close / Total-Return-Näherung)",
            value=False,
            help="Yahoo Adj Close kann Dividenden und Splits berücksichtigen.",
        )
        chart = make_price_chart(
            histories.get(ticker, pd.DataFrame()),
            period,
            show_smas,
            events,
            use_total_return=use_total_return,
        )
        if chart is None:
            st.info("Für diese Aktie sind keine Kursdaten verfügbar.")
        else:
            st.altair_chart(chart, use_container_width=True)
            if not events.empty:
                st.caption("Chartmarker stammen aus gespeicherten News sowie Yahoo-Dividenden-/Kalenderdaten. Verfügbarkeit variiert je Aktie.")

    st.caption("*Gesamtrendite und bereinigter Kurs: Yahoo Adj Close als Näherung, nicht als Broker- oder Steuerberechnung.")


def render_sector_view(df: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> None:
    st.subheader("Sektor-Übersicht 2.0")
    st.caption(
        "Vergleicht Sektoren über mehrere Zeiträume. Kursrendite = Close, Gesamtrendite = Yahoo Adj Close als Näherung."
    )
    if df is None or df.empty:
        st.info("Keine Daten für die Sektoransicht vorhanden.")
        return

    sector_stats = sector_timeframe_stats(df, histories)
    if sector_stats.empty:
        st.info("Für die geladenen Werte liegen keine ausreichenden Kursdaten vor.")
        return

    metric_mode = st.radio(
        "Kennzahl anzeigen",
        ["Kursrendite", "Gesamtrendite", "Bewertung & Risiko"],
        horizontal=True,
        key="sector_view_metric_mode_v2",
    )

    if metric_mode in ("Kursrendite", "Gesamtrendite"):
        prefix = "Kurs" if metric_mode == "Kursrendite" else "Gesamt"
        columns = ["sector", "Unternehmen"] + [f"{prefix}_{label}" for label in ["1M", "3M", "6M", "1J", "3J", "5J"]]
        visible = sector_stats[columns].copy().sort_values(f"{prefix}_1J", ascending=False, na_position="last")
        rename_map = {
            "sector": "Sektor",
            "Unternehmen": "Unternehmen",
            f"{prefix}_1M": "1M",
            f"{prefix}_3M": "3M",
            f"{prefix}_6M": "6M",
            f"{prefix}_1J": "1J",
            f"{prefix}_3J": "3J",
            f"{prefix}_5J": "5J",
        }
        display = visible.rename(columns=rename_map)
        st.dataframe(
            display.style.format(
                {col: (lambda value: format_percent(value, 1, signed=True)) for col in ["1M", "3M", "6M", "1J", "3J", "5J"]},
                na_rep="–",
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_period = st.selectbox(
            "Diagramm-Zeitraum",
            ["1M", "3M", "6M", "1J", "3J", "5J"],
            index=3,
            key=f"sector_chart_period_{prefix}",
        )
        chart_data = display.dropna(subset=[chart_period]).copy()
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X(f"{chart_period}:Q", title=f"{metric_mode} {chart_period} in %"),
            y=alt.Y("Sektor:N", sort="-x", title="Sektor"),
            tooltip=["Sektor:N", "Unternehmen:Q", alt.Tooltip(f"{chart_period}:Q", format=".2f")],
        ).properties(height=max(260, 34 * len(chart_data)))
        st.altair_chart(chart, use_container_width=True)

    else:
        columns = [
            "sector", "Unternehmen", "Qualitäts_Score", "Value_Score", "Value_Trigger",
            "Dividendenrendite", "Volatilität_1J", "Max_Drawdown_1J",
        ]
        visible = sector_stats[columns].copy().sort_values("Value_Score", ascending=False, na_position="last")
        display = visible.rename(columns={
            "sector": "Sektor",
            "Qualitäts_Score": "Qualitäts-Score",
            "Value_Score": "Value-Score",
            "Value_Trigger": "Value-Trigger",
            "Dividendenrendite": "Dividendenrendite",
            "Volatilität_1J": "Volatilität 1J",
            "Max_Drawdown_1J": "Max. Drawdown 1J",
        })
        st.dataframe(
            display.style.format(
                {
                    "Qualitäts-Score": lambda value: format_number(value, 1),
                    "Value-Score": lambda value: format_number(value, 1),
                    "Dividendenrendite": lambda value: format_percent(value, 2),
                    "Volatilität 1J": lambda value: format_percent(value, 1),
                    "Max. Drawdown 1J": lambda value: format_percent(value, 1, signed=True),
                },
                na_rep="–",
            ),
            use_container_width=True,
            hide_index=True,
        )

        scatter = alt.Chart(display.dropna(subset=["Value-Score", "Volatilität 1J"])).mark_circle(size=120).encode(
            x=alt.X("Volatilität 1J:Q", title="Volatilität 1J in %"),
            y=alt.Y("Value-Score:Q", title="Value-Score"),
            size=alt.Size("Unternehmen:Q", title="Unternehmen"),
            tooltip=["Sektor:N", "Unternehmen:Q", alt.Tooltip("Value-Score:Q", format=".1f"), alt.Tooltip("Volatilität 1J:Q", format=".1f")],
        ).properties(height=420)
        st.altair_chart(scatter, use_container_width=True)

    with st.expander("Was sagt diese Ansicht aus?"):
        st.write(
            "Die Sektoransicht zeigt, ob ein Scanner-Kandidat nur wegen eines einzelnen Unternehmens auffällt "
            "oder ob ein kompletter Sektor unter Druck steht. Besonders nützlich ist der Vergleich aus 1J/3J/5J-Rendite, "
            "Value-Score, Dividendenrendite und Volatilität. Ein hoher Value-Score bei gleichzeitig hohem Drawdown kann "
            "eine Chance oder eine Value Trap sein – deshalb immer mit News, Cashflow und Verschuldung gegenprüfen."
        )


def render_news_card(item: pd.Series, card_index: int) -> None:
    """Kompakte, lesbare News-Karte statt einer breiten Rohdaten-Tabelle."""
    published = pd.to_datetime(item.get("published"), errors="coerce")
    date_label = published.strftime("%d.%m.%Y, %H:%M") if pd.notna(published) else "Datum unbekannt"
    event_label = EVENT_META.get(str(item.get("event_type", "news")), {}).get("label", "News")
    sentiment = str(item.get("sentiment_label") or "neutral").capitalize()
    relevance = str(item.get("relevance_label") or "–").capitalize()

    with st.container(border=True):
        st.markdown(f"**{str(item.get('title') or 'Ohne Titel')}**")
        st.caption(
            f"{event_label} · {sentiment} · Relevanz: {relevance} · "
            f"{str(item.get('source') or 'Quelle unbekannt')} · {date_label}"
        )
        reason = str(item.get("relevance_reason") or "").strip()
        if reason:
            st.caption(f"Warum zugeordnet: {reason}")
        link = str(item.get("link") or "").strip()
        if link:
            st.link_button("Artikel öffnen", link, key=f"news_link_{card_index}_{normalize_for_search(str(item.get('title', '')))}")


def render_news_summary(
    ticker: str,
    company_name: str,
    relevant_news: pd.DataFrame,
    uncertain_news: pd.DataFrame,
    events: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    """Zeigt eine sachliche Einordnung ohne daraus ein Handelssignal abzuleiten."""
    future_events = pd.DataFrame()
    if events is not None and not events.empty:
        future_events = events.copy()
        future_events["date"] = pd.to_datetime(future_events["date"], errors="coerce")
        future_events["_future"] = future_events["is_future_event"].astype(str).str.lower().isin({"true", "1", "yes", "ja"})
        future_events = future_events[future_events["_future"] & future_events["date"].notna()].sort_values("date")

    if relevant_news.empty:
        headline = "Keine verlässlich passende Unternehmensmeldung erkannt."
    else:
        latest = relevant_news.iloc[0]
        headline = f"{len(relevant_news)} relevante Meldung(en) im gewählten Zeitraum; zuletzt: {str(latest.get('title') or '')}"

    st.markdown("#### Einordnung")
    st.info(headline)

    notes: list[str] = []
    if not future_events.empty:
        next_event = future_events.iloc[0]
        event_label = EVENT_META.get(str(next_event.get("event_type")), {}).get("label", str(next_event.get("event_type")))
        notes.append(f"Nächstes Ereignis: **{event_label}** am **{pd.Timestamp(next_event['date']).strftime('%d.%m.%Y')}**.")
    else:
        notes.append("Kein kommender Termin aus den aktuell verfügbaren Yahoo-Kalenderdaten erkannt.")

    if not diagnostics.empty:
        reachable = diagnostics[diagnostics.get("http_status", pd.Series(dtype=float)).fillna(0).astype(int).between(200, 299)]
        notes.append(f"Quellenabruf: **{len(reachable)} von {len(diagnostics)}** Quellen technisch erreichbar.")
    if not uncertain_news.empty:
        notes.append(f"{len(uncertain_news)} Treffer mit schwachem Firmenbezug wurden standardmäßig ausgeblendet.")
    notes.append("Die Einordnung ist eine Recherchehilfe und kein Kauf- oder Verkaufssignal.")

    for note in notes:
        st.markdown(f"- {note}")


def render_news(df: pd.DataFrame) -> None:
    st.subheader("News & Ereignisse")
    st.caption(
        "Relevante Unternehmensnews, bestätigte Kalendertermine und technische Quelleninformationen sind getrennt dargestellt. "
        "Mehrdeutige Ticker werden bewusst strenger gefiltert."
    )

    ticker = st.selectbox("Aktie für News", df["ticker_yahoo"].tolist(), key="news_ticker")
    row = df.loc[df["ticker_yahoo"] == ticker].iloc[0]
    company_name = str(row.get("name", ticker))

    controls_left, controls_right = st.columns([2, 1])
    with controls_left:
        days_back = st.slider("Zeitraum", min_value=3, max_value=180, value=30, step=1, key="news_days")
    with controls_right:
        locale = st.radio("News-Sprache", ["Deutsch", "Englisch"], horizontal=True, key="news_locale")
    include_google = st.checkbox("Google News als firmenbezogenen Such-Fallback verwenden", value=True)
    locale_code = "en" if locale == "Englisch" else "de"
    key = f"news_bundle::{ticker}::{days_back}::{include_google}::{locale_code}"

    aliases = news_identity_aliases(company_name, ticker)
    with st.expander("Firmen-Aliase verwalten", expanded=False):
        st.caption(
            "Für mehrdeutige Kürzel wie MMM, T oder F nutze bitte möglichst eindeutige Namen, "
            "beispielsweise „3M Company“ statt nur „MMM“."
        )
        alias_text = ", ".join(aliases)
        edited_aliases = st.text_area(
            "Suchbegriffe (Komma, Semikolon oder Zeilenumbruch trennen)",
            value=alias_text,
            key=f"aliases_editor_{ticker}",
        )
        if st.button("Aliase speichern", key=f"save_aliases_{ticker}"):
            save_aliases_for_ticker(ticker, edited_aliases)
            fetch_news_bundle.clear()
            st.success("Aliase wurden gespeichert. Aktualisiere danach die News.")
            st.rerun()

    if st.button("News & Ereignisse aktualisieren", key=f"refresh_news_{ticker}", type="primary"):
        with st.spinner("RSS-Feeds, Such-Fallback und Yahoo-Kalender werden geladen …"):
            news, diagnostics = fetch_news_bundle(
                ticker=ticker,
                company_name=company_name,
                days_back=days_back,
                include_google_news=include_google,
                locale=locale_code,
            )
            calendar_events = fetch_yahoo_calendar_events(ticker, days_back=max(365, days_back), days_forward=365)
            events = combine_events(news, calendar_events, ticker)
            events_ok, events_message = persist_events(events, replace_ticker=ticker)
            snapshot_path = NEWS_SNAPSHOT_DIR / f"{ticker.replace('.', '_')}.csv"
            snapshot_ok, snapshot_message = safe_write_csv(news, snapshot_path)
            st.session_state[key] = {
                "news": news,
                "diagnostics": diagnostics,
                "events": events,
                "loaded_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "snapshot_warning": "" if snapshot_ok else snapshot_message,
                "events_warning": "" if events_ok else events_message,
            }

    bundle = st.session_state.get(key)
    if bundle is None:
        stored_events = load_persisted_events(ticker, days_back=max(365, days_back))
        if not stored_events.empty:
            st.info("Gespeicherte Kalenderereignisse sind verfügbar. Für aktuelle News bitte aktualisieren.")
            render_event_calendar(stored_events)
        else:
            st.info("Noch keine News geladen. Klicke auf „News & Ereignisse aktualisieren“.")
        return

    news = bundle["news"].copy()
    diagnostics = bundle["diagnostics"].copy()
    events = bundle["events"].copy()
    warning = str(bundle.get("snapshot_warning") or "")
    events_warning = str(bundle.get("events_warning") or "")
    if warning:
        st.warning(warning)
    if events_warning:
        st.warning(events_warning)

    is_relevant = news.get("is_relevant", pd.Series(False, index=news.index)).fillna(False).astype(bool) if not news.empty else pd.Series(dtype=bool)
    relevant_news = news[is_relevant].copy() if not news.empty else empty_news_frame()
    uncertain_news = news[~is_relevant].copy() if not news.empty else empty_news_frame()

    future_events = pd.DataFrame()
    if not events.empty:
        future_events = events.copy()
        future_events["date"] = pd.to_datetime(future_events["date"], errors="coerce")
        future_events["_future"] = future_events["is_future_event"].astype(str).str.lower().isin({"true", "1", "yes", "ja"})
        future_events = future_events[future_events["_future"] & future_events["date"].notna()].sort_values("date")

    reachable_sources = 0
    total_sources = len(diagnostics)
    if not diagnostics.empty and "http_status" in diagnostics.columns:
        reachable_sources = int(diagnostics["http_status"].fillna(0).astype(int).between(200, 299).sum())

    status_columns = st.columns(4)
    status_columns[0].metric("Relevante News", int(len(relevant_news)))
    status_columns[1].metric("Nächstes Ereignis", pd.Timestamp(future_events.iloc[0]["date"]).strftime("%d.%m.%Y") if not future_events.empty else "–")
    status_columns[2].metric("Abrufbare Quellen", f"{reachable_sources} / {total_sources}" if total_sources else "–")
    status_columns[3].metric("Daten aktualisiert", bundle.get("loaded_at", "–"))

    render_news_summary(ticker, company_name, relevant_news, uncertain_news, events, diagnostics)

    tab_news, tab_calendar, tab_sources, tab_export = st.tabs([
        "Aktuelle News", "Kalender", "Quellen & Diagnose", "Export"
    ])

    with tab_news:
        if relevant_news.empty:
            st.warning(
                "Keine verlässlich passende Unternehmensmeldung gefunden. "
                "Prüfe bei Bedarf die Firmen-Aliase oder die Quellen-Diagnose."
            )
        else:
            for index, (_, item) in enumerate(relevant_news.iterrows()):
                render_news_card(item, index)

        if not uncertain_news.empty:
            with st.expander(f"Unsichere Treffer anzeigen ({len(uncertain_news)})", expanded=False):
                st.caption(
                    "Diese Artikel hatten einen zu schwachen Firmenbezug und werden nicht als relevante Unternehmensnews gezählt. "
                    "Sie erscheinen nicht im Kalender und nicht im Chart."
                )
                for index, (_, item) in enumerate(uncertain_news.iterrows(), start=1000):
                    render_news_card(item, index)

    with tab_calendar:
        st.caption("Der Kalender enthält Yahoo-Dividenden-/Kalenderdaten sowie nur konkret klassifizierte, relevante Nachrichten.")
        render_event_calendar(events)

    with tab_sources:
        st.caption("Dieser Bereich ist für technische Prüfung gedacht. Fehlerhafte oder eingeschränkte Quellen beeinflussen die News-Abdeckung.")
        if diagnostics.empty:
            st.info("Keine Quelle wurde abgefragt.")
        else:
            diagnostic_columns = [
                "source", "kind", "status", "http_status", "entries", "matches",
                "uncertain_matches", "duration_ms", "message", "url",
            ]
            visible_diagnostics = [column for column in diagnostic_columns if column in diagnostics.columns]
            st.dataframe(diagnostics[visible_diagnostics], use_container_width=True, hide_index=True)
            failed = diagnostics[~diagnostics.get("http_status", pd.Series(dtype=float)).fillna(0).astype(int).between(200, 299)]
            if not failed.empty:
                st.warning("Mindestens eine Quelle war nicht erreichbar oder lieferte keinen brauchbaren Feed. Das ist ein Quellenproblem, kein Handelssignal.")

    with tab_export:
        st.caption("CSV-Export enthält relevante und unsichere Treffer samt Relevanzbegründung.")
        st.download_button(
            "Alle News als CSV herunterladen",
            data=news.to_csv(index=False).encode("utf-8"),
            file_name=f"news_{ticker.lower()}.csv",
            mime="text/csv",
            key=f"download_news_{ticker}",
        )
        if not events.empty:
            st.download_button(
                "Kalender als CSV herunterladen",
                data=events.to_csv(index=False).encode("utf-8"),
                file_name=f"events_{ticker.lower()}.csv",
                mime="text/csv",
                key=f"download_events_{ticker}",
            )

    st.caption(
        "News-Relevanz wird aus Firmenalias und Finanz-/Unternehmenskontext abgeleitet. "
        "Sie ist eine Filterhilfe; Quellen, Inhalte und Termine können unvollständig sein."
    )


def render_portfolio(metrics: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> None:
    st.subheader("Portfolio")
    st.caption(
        "Nutze bevorzugt data/transactions.csv für mehrere Käufe/Verkäufe. "
        "Ohne Transaktionsbuch bleibt die bisherige portfolio.csv kompatibel."
    )
    portfolio, origin, warnings = portfolio_input()
    if portfolio.empty:
        st.info(
            "Keine Bestände gefunden. Lege entweder portfolio.csv neben app.py an oder fülle data/transactions.csv. "
            "Im Tab kannst du eine Vorlage herunterladen."
        )
        st.download_button(
            "Vorlage transactions.csv herunterladen",
            data=empty_transactions_frame().to_csv(index=False).encode("utf-8"),
            file_name="transactions.csv",
            mime="text/csv",
        )
        return

    if warnings:
        for warning in warnings:
            st.warning(warning)
    st.caption(f"Datenbasis: {origin}")

    portfolio_tickers = set(portfolio["ticker_yahoo"].map(clean_ticker))
    known_tickers = set(metrics["ticker_yahoo"].map(clean_ticker))
    missing_tickers = sorted(portfolio_tickers - known_tickers)
    if missing_tickers:
        st.warning("Portfolio-Ticker außerhalb der Indexanalyse: " + ", ".join(missing_tickers))
        if st.button("Fehlende Portfolio-Ticker laden", key="load_portfolio_extras"):
            companies = pd.DataFrame({"name": missing_tickers, "ticker_yahoo": missing_tickers, "sector": "Unbekannt"})
            with st.spinner("Zusätzliche Portfolio-Daten werden geladen …"):
                extra_metrics, extra_histories, extra_errors = collect_metrics(companies)
            st.session_state["portfolio_extra_metrics"] = extra_metrics
            st.session_state["portfolio_extra_histories"] = extra_histories
            if extra_errors:
                st.warning(" | ".join(extra_errors[:10]))
            st.rerun()

    extras = st.session_state.get("portfolio_extra_metrics", empty_metrics_frame())
    extra_histories = st.session_state.get("portfolio_extra_histories", {})
    all_histories = {**histories, **extra_histories}
    combined = pd.concat([metrics, extras], ignore_index=True).drop_duplicates(subset=["ticker_yahoo"], keep="first")
    portfolio_view = build_portfolio_view(combined, portfolio)

    total_value = portfolio_view["market_value_eur"].sum(skipna=True)
    total_cost = portfolio_view["cost_value_eur"].sum(skipna=True)
    total_pnl = portfolio_view["pnl_abs_eur"].sum(skipna=True)
    total_income = portfolio_view["dividend_income_eur"].sum(skipna=True)

    columns = st.columns(5)
    columns[0].metric("Marktwert", format_eur(total_value, 0))
    columns[1].metric("Einstandswert", format_eur(total_cost, 0))
    columns[2].metric(
        "Gewinn / Verlust",
        format_eur(total_pnl, 0, signed=True),
        delta=format_percent(total_pnl / total_cost * 100 if total_cost else None, 1, signed=True),
    )
    columns[3].metric("Dividende p.a. (Schätzung)", format_eur(total_income, 0))
    columns[4].metric("Yield on Cost", format_percent(total_income / total_cost * 100 if total_cost else None, 2))

    visible_columns = [
        "ticker_yahoo", "name", "sector", "shares", "cost_basis", "cost_currency", "asset_currency",
        "last_price", "market_value_eur", "cost_value_eur", "pnl_abs_eur", "pnl_pct", "weight_pct",
        "dividend_income_eur", "yield_on_cost", "total_return_1y", "vol_1y", "max_drawdown_1y",
        "total_score", "value_score", "value_trigger", "fx_status",
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
        "total_return_1y": lambda value: format_percent(value, 2, signed=True),
        "vol_1y": lambda value: format_percent(value, 1),
        "max_drawdown_1y": lambda value: format_percent(value, 1, signed=True),
        "total_score": lambda value: format_number(value, 1),
        "value_score": lambda value: format_number(value, 1),
    }
    styled = (
        portfolio_view[visible_columns].style.format({key: value for key, value in formats.items() if key in visible_columns}, na_rep="–")
        .map(colorize_change, subset=[column for column in ["pnl_abs_eur", "pnl_pct", "total_return_1y"] if column in visible_columns])
        .map(colorize_score, subset=[column for column in ["total_score"] if column in visible_columns])
        .map(colorize_value_trigger, subset=[column for column in ["value_trigger"] if column in visible_columns])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.download_button(
        "Portfolio-Übersicht als CSV herunterladen",
        data=portfolio_view.to_csv(index=False).encode("utf-8"),
        file_name="portfolio_uebersicht.csv",
        mime="text/csv",
    )
    st.caption("EUR-Werte verwenden aktuelle Yahoo-FX-Kurse. Für steuerliche Werte, Gebühren oder historische FX-Kurse ist diese Ansicht nicht geeignet.")

    render_portfolio_risk(portfolio_view, all_histories)


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
        index_name = st.selectbox("Index", INDEX_OPTIONS, key="index_name")
        st.caption("Optional stabiler über CSV: data/indices/dax40.csv, mdax.csv, sdax.csv oder sp500.csv. Wikipedia bleibt nur Fallback.")

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

        maximum = len(filtered_constituents)
        if maximum <= 0:
            st.error("Der Filter enthält keine Unternehmen.")
            st.stop()
        default_count = min(40, maximum)
        slider_step = 1 if maximum <= 150 else 10
        max_stocks = st.slider(
            "Max. Unternehmen laden",
            min_value=1,
            max_value=maximum,
            value=default_count,
            step=slider_step,
            help="Mehr Unternehmen bedeuten deutlich längere Ladezeiten, weil viele Fundamentaldaten einzeln abgefragt werden.",
        )

        st.divider()
        st.header("Scanner-Profil")
        profile_name = st.selectbox("Profil", list(STRATEGY_PROFILES), key="strategy_profile")
        profile = STRATEGY_PROFILES[profile_name]

        # Presets werden nur bei einem Profilwechsel gesetzt, danach bleiben die
        # Slider frei einstellbar.
        if st.session_state.get("_applied_strategy_profile") != profile_name:
            st.session_state["scanner_drawdown"] = int(profile["drawdown"])
            st.session_state["scanner_payout"] = int(profile["payout"])
            st.session_state["scanner_score"] = int(profile["score"])
            st.session_state["scanner_yield"] = float(profile["yield"])
            st.session_state["_applied_strategy_profile"] = profile_name

        st.caption(profile["description"])
        drawdown_trigger = st.slider("Min. Drawdown vom 52W-Hoch", 10, 60, int(profile["drawdown"]), 5, key="scanner_drawdown")
        payout_max = st.slider("Max. Payout Ratio", 40, 120, int(profile["payout"]), 5, key="scanner_payout")
        score_min = st.slider("Min. Qualitäts-Score", 0, 100, int(profile["score"]), 5, key="scanner_score")
        yield_min = st.slider("Min. Dividendenrendite", 1.0, 10.0, float(profile["yield"]), 0.5, key="scanner_yield")

        st.divider()
        reload_clicked = st.button("Daten laden / aktualisieren", type="primary", use_container_width=True)
        if st.button("Zwischenspeicher leeren", use_container_width=True):
            load_index_constituents.clear()
            download_price_histories.clear()
            fetch_ticker_info.clear()
            fetch_dividends.clear()
            fetch_news_for_ticker.clear()
            fetch_news_bundle.clear()
            fetch_yahoo_calendar_events.clear()
            fetch_benchmark_history.clear()
            fx_to_eur.clear()
            for state_key in [
                "metrics_raw", "histories", "loaded_tickers", "portfolio_extra_metrics",
                "portfolio_extra_histories",
            ]:
                st.session_state.pop(state_key, None)
            st.success("Zwischenspeicher wurde geleert.")
            st.rerun()

    selected_constituents = filtered_constituents.head(max_stocks).reset_index(drop=True)
    selected_tickers = tuple(selected_constituents["ticker_yahoo"].map(clean_ticker))

    # Rohdaten sind unabhängig vom Profil und den Sliderwerten. Der Score wird
    # bei jeder UI-Änderung erneut berechnet, ohne externe Daten erneut zu laden.
    loaded_tickers = tuple(st.session_state.get("loaded_tickers", ()))
    if reload_clicked or loaded_tickers != selected_tickers:
        with st.spinner(f"Lade Daten für {len(selected_constituents)} Unternehmen …"):
            raw_metrics, histories, errors = collect_metrics(selected_constituents)
        st.session_state["metrics_raw"] = raw_metrics
        st.session_state["histories"] = histories
        st.session_state["loaded_tickers"] = selected_tickers
        st.session_state["load_errors"] = errors
        st.session_state["selected_constituents_snapshot"] = selected_constituents.copy()
        st.session_state["last_refresh"] = datetime.now().strftime("%d.%m.%Y %H:%M")
        if errors:
            st.warning("Einige Daten konnten nicht vollständig geladen werden: " + " | ".join(errors[:8]))

    raw_metrics = st.session_state.get("metrics_raw")
    histories = st.session_state.get("histories", {})
    if raw_metrics is None or raw_metrics.empty:
        st.info("Wähle links einen Index und klicke auf „Daten laden / aktualisieren“. ")
        st.stop()

    data = enrich_with_scores(
        raw_metrics,
        drawdown_trigger=float(drawdown_trigger),
        payout_max=float(payout_max),
        score_min=float(score_min),
        yield_min=float(yield_min),
    )
    data = enrich_with_special_situations(data)

    load_errors = st.session_state.get("load_errors", [])
    selected_snapshot = st.session_state.get("selected_constituents_snapshot", selected_constituents)
    status_summary, status_detail = build_data_status(
        selected_snapshot, data, histories, load_errors, index_name=index_name
    )

    last_refresh = st.session_state.get("last_refresh", "–")
    requested_count = int(status_summary.get("angefragt", len(selected_snapshot)))
    loaded_count = int(status_summary.get("analysiert", len(data)))
    coverage = safe_float(status_summary.get("abdeckung_prozent"))
    coverage_label = format_percent(coverage, 1) if coverage is not None else "–"
    st.caption(
        f"Aktualisiert: {last_refresh} · {loaded_count}/{requested_count} Unternehmen analysiert "
        f"({coverage_label} Abdeckung) · Profil: {profile_name} · Portfolio-Basiswährung: {BASE_CURRENCY}"
    )

    tabs = st.tabs([
        "Überblick", "Datenstatus", "Fundamentaldaten", "Einzelanalyse", "Sektoren",
        "News & Events", "Portfolio", "Watchlist", "Value-Scanner", "Deep Value", "Research",
    ])
    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_data_status(status_summary, status_detail, data)
    with tabs[2]:
        render_fundamentals(data)
    with tabs[3]:
        render_risk_and_chart(data, histories)
    with tabs[4]:
        render_sector_view(data, histories)
    with tabs[5]:
        render_news(data)
    with tabs[6]:
        render_portfolio(data, histories)
    with tabs[7]:
        render_watchlist(data)
    with tabs[8]:
        render_value_watchlist(
            data,
            drawdown_trigger=float(drawdown_trigger),
            payout_max=float(payout_max),
            score_min=float(score_min),
            yield_min=float(yield_min),
            profile_name=profile_name,
        )
    with tabs[9]:
        render_special_situation_scanner(data)
    with tabs[10]:
        render_research(data, histories, index_name)



if __name__ == "__main__":
    main()