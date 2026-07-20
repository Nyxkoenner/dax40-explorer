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

APP_VERSION = "5.6.0"
APP_TITLE = "Aktien Explorer"
BASE_CURRENCY = "EUR"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "indices"
CACHE_DIR = ROOT_DIR / ".cache"
WATCHLIST_PATH = DATA_DIR / "watchlist.csv"
PORTFOLIO_PATH = ROOT_DIR / "portfolio.csv"
RESEARCH_CASES_PATH = DATA_DIR / "research_cases.csv"
BACKTEST_DIR = DATA_DIR / "backtests"
BACKTEST_RUNS_PATH = BACKTEST_DIR / "backtest_runs.csv"

for directory in (DATA_DIR, INDEX_DIR, CACHE_DIR, BACKTEST_DIR):
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

# Gewichtete Phrasen statt einzelner, mehrdeutiger Wörter. Dadurch wird z. B.
# "profit warning" nicht mehr durch das Wort "profit" fälschlich neutralisiert.
# Der Score ist eine transparente Heuristik auf Basis von Überschrift und RSS-
# Beschreibung; er ersetzt keine vollständige Artikelanalyse.
POSITIVE_SENTIMENT_PHRASES: dict[str, int] = {
    "beats estimates": 3,
    "beat estimates": 3,
    "better than expected": 3,
    "uebertrifft erwartungen": 3,
    "besser als erwartet": 3,
    "raises guidance": 3,
    "raised guidance": 3,
    "guidance raised": 3,
    "prognose angehoben": 3,
    "hebt prognose an": 3,
    "erhoeht prognose": 3,
    "record profit": 3,
    "record revenue": 3,
    "rekordgewinn": 3,
    "rekordumsatz": 3,
    "dividend increase": 3,
    "dividend raised": 3,
    "raises dividend": 3,
    "dividende erhoeht": 3,
    "erhoeht dividende": 3,
    "share buyback": 2,
    "stock buyback": 2,
    "aktienrueckkauf": 2,
    "analyst upgrade": 2,
    "upgraded to buy": 2,
    "upgrade to buy": 2,
    "outperform rating": 2,
    "kursziel angehoben": 2,
    "hebt kursziel": 2,
    "heben kursziel": 2,
    "wins contract": 2,
    "contract win": 2,
    "auftrag gewonnen": 2,
    "starkes wachstum": 2,
    "strong growth": 2,
    "profit rises": 2,
    "earnings rise": 2,
    "gewinn steigt": 2,
    "revenue rises": 1,
    "sales rise": 1,
    "umsatz steigt": 1,
    "margin improvement": 2,
    "margin expands": 2,
    "marge verbessert": 2,
    "debt reduced": 2,
    "reduces debt": 2,
    "verschuldung gesenkt": 2,
    "regulatory approval": 2,
    "zulassung erhalten": 2,
}

NEGATIVE_SENTIMENT_PHRASES: dict[str, int] = {
    "profit warning": 4,
    "gewinnwarnung": 4,
    "cuts guidance": 3,
    "cut guidance": 3,
    "guidance cut": 3,
    "lowers guidance": 3,
    "senkt prognose": 3,
    "prognose gesenkt": 3,
    "dividend cut": 4,
    "cuts dividend": 4,
    "dividende gekuerzt": 4,
    "kuerzt dividende": 4,
    "suspends dividend": 4,
    "dividende ausgesetzt": 4,
    "misses estimates": 3,
    "missed estimates": 3,
    "below expectations": 3,
    "verfehlt erwartungen": 3,
    "unter den erwartungen": 3,
    "analyst downgrade": 2,
    "downgraded to sell": 2,
    "underperform rating": 2,
    "kursziel gesenkt": 2,
    "senkt kursziel": 2,
    "lawsuit": 2,
    "class action": 2,
    "klage": 2,
    "investigation": 2,
    "untersuchung": 2,
    "fraud": 4,
    "betrug": 4,
    "recall": 2,
    "product recall": 2,
    "rueckruf": 2,
    "job cuts": 2,
    "layoffs": 2,
    "stellenabbau": 2,
    "loss widens": 3,
    "verlust weitet sich aus": 3,
    "revenue falls": 2,
    "sales fall": 2,
    "umsatz sinkt": 2,
    "profit falls": 2,
    "earnings fall": 2,
    "gewinn sinkt": 2,
    "bankruptcy": 5,
    "insolvency": 5,
    "insolvenz": 5,
    "regulatory setback": 2,
    "trial fails": 3,
    "clinical trial failure": 3,
    "studie gescheitert": 3,
    "plant closure": 2,
    "werksschliessung": 2,
    "ruecklaeufe": 1,
}

# Beibehalten für ältere Hilfsfunktionen/Kompatibilität.
POSITIVE_WORDS = set(POSITIVE_SENTIMENT_PHRASES)
NEGATIVE_WORDS = set(NEGATIVE_SENTIMENT_PHRASES)

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

# Integrierte Offline-Listen für deutsche Indizes. Stand: 22. Juni 2026.
# Lokale CSV-Dateien unter data/indices/ überschreiben diese Vorlagen.
# Dadurch ist der normale App-Start nicht von Wikipedia oder dessen Rate-Limits abhängig.
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
    {"name": "Hochtief", "ticker_yahoo": "HOT.DE", "sector": "Construction"},
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


MDAX_STATIC_CONSTITUENTS = [
    {"name": "AIXTRON", "ticker_yahoo": "AIXA.DE", "sector": "Technology"},
    {"name": "Aroundtown", "ticker_yahoo": "AT1.DE", "sector": "Real Estate"},
    {"name": "AUMOVIO", "ticker_yahoo": "AMV0.DE", "sector": "Automotive Technology"},
    {"name": "Aurubis", "ticker_yahoo": "NDA.DE", "sector": "Basic Materials"},
    {"name": "AUTO1 Group", "ticker_yahoo": "AG1.DE", "sector": "Consumer Cyclical"},
    {"name": "Bechtle", "ticker_yahoo": "BC8.DE", "sector": "Technology"},
    {"name": "Bilfinger", "ticker_yahoo": "GBF.DE", "sector": "Industrials"},
    {"name": "CTS Eventim", "ticker_yahoo": "EVD.DE", "sector": "Communication Services"},
    {"name": "Delivery Hero", "ticker_yahoo": "DHER.DE", "sector": "Consumer Cyclical"},
    {"name": "Deutz", "ticker_yahoo": "DEZ.DE", "sector": "Industrials"},
    {"name": "DWS Group", "ticker_yahoo": "DWS.DE", "sector": "Financial Services"},
    {"name": "Elmos Semiconductor", "ticker_yahoo": "ELG.DE", "sector": "Technology"},
    {"name": "Evonik Industries", "ticker_yahoo": "EVK.DE", "sector": "Chemicals"},
    {"name": "flatexDEGIRO", "ticker_yahoo": "FTK.DE", "sector": "Financial Services"},
    {"name": "Fraport", "ticker_yahoo": "FRA.DE", "sector": "Industrials"},
    {"name": "freenet", "ticker_yahoo": "FNTN.DE", "sector": "Communication Services"},
    {"name": "FUCHS", "ticker_yahoo": "FPE3.DE", "sector": "Chemicals"},
    {"name": "HELLA", "ticker_yahoo": "HLE.DE", "sector": "Automotive"},
    {"name": "HENSOLDT", "ticker_yahoo": "HAG.DE", "sector": "Aerospace & Defence"},
    {"name": "HUGO BOSS", "ticker_yahoo": "BOSS.DE", "sector": "Consumer Cyclical"},
    {"name": "IONOS Group", "ticker_yahoo": "IOS.DE", "sector": "Technology"},
    {"name": "Jenoptik", "ticker_yahoo": "JEN.DE", "sector": "Technology"},
    {"name": "K+S", "ticker_yahoo": "SDF.DE", "sector": "Basic Materials"},
    {"name": "KION Group", "ticker_yahoo": "KGX.DE", "sector": "Industrials"},
    {"name": "Knorr-Bremse", "ticker_yahoo": "KBX.DE", "sector": "Industrials"},
    {"name": "KRONES", "ticker_yahoo": "KRN.DE", "sector": "Industrials"},
    {"name": "LANXESS", "ticker_yahoo": "LXS.DE", "sector": "Chemicals"},
    {"name": "LEG Immobilien", "ticker_yahoo": "LEG.DE", "sector": "Real Estate"},
    {"name": "Lufthansa", "ticker_yahoo": "LHA.DE", "sector": "Industrials"},
    {"name": "Nemetschek", "ticker_yahoo": "NEM.DE", "sector": "Technology"},
    {"name": "Nordex", "ticker_yahoo": "NDX1.DE", "sector": "Industrials"},
    {"name": "Porsche AG", "ticker_yahoo": "P911.DE", "sector": "Automotive"},
    {"name": "Porsche Automobil Holding", "ticker_yahoo": "PAH3.DE", "sector": "Automotive"},
    {"name": "PUMA", "ticker_yahoo": "PUM.DE", "sector": "Consumer Cyclical"},
    {"name": "RATIONAL", "ticker_yahoo": "RAA.DE", "sector": "Industrials"},
    {"name": "RENK Group", "ticker_yahoo": "R3NK.DE", "sector": "Aerospace & Defence"},
    {"name": "RTL Group", "ticker_yahoo": "RRTL.DE", "sector": "Communication Services"},
    {"name": "Salzgitter", "ticker_yahoo": "SZG.DE", "sector": "Basic Materials"},
    {"name": "Sartorius", "ticker_yahoo": "SRT3.DE", "sector": "Healthcare"},
    {"name": "Schaeffler", "ticker_yahoo": "SHA0.DE", "sector": "Automotive"},
    {"name": "Siltronic", "ticker_yahoo": "WAF.DE", "sector": "Technology"},
    {"name": "SUSS MicroTec", "ticker_yahoo": "SMHN.DE", "sector": "Technology"},
    {"name": "TAG Immobilien", "ticker_yahoo": "TEG.DE", "sector": "Real Estate"},
    {"name": "Talanx", "ticker_yahoo": "TLX.DE", "sector": "Financial Services"},
    {"name": "thyssenkrupp", "ticker_yahoo": "TKA.DE", "sector": "Industrials"},
    {"name": "TKMS", "ticker_yahoo": "TKMS.DE", "sector": "Aerospace & Defence"},
    {"name": "TRATON", "ticker_yahoo": "8TRA.DE", "sector": "Automotive"},
    {"name": "TUI", "ticker_yahoo": "TUI1.DE", "sector": "Consumer Cyclical"},
    {"name": "United Internet", "ticker_yahoo": "UTDI.DE", "sector": "Communication Services"},
    {"name": "Wacker Chemie", "ticker_yahoo": "WCH.DE", "sector": "Chemicals"},
]

SDAX_STATIC_CONSTITUENTS = [
    {"name": "1&1", "ticker_yahoo": "1U1.DE", "sector": "Communication Services"},
    {"name": "Adtran Networks", "ticker_yahoo": "ADV.DE", "sector": "Technology"},
    {"name": "AlzChem Group", "ticker_yahoo": "ACT.DE", "sector": "Chemicals"},
    {"name": "Asta Energy Solutions", "ticker_yahoo": "1AST.DE", "sector": "Industrials"},
    {"name": "ATOSS Software", "ticker_yahoo": "AOF.DE", "sector": "Technology"},
    {"name": "Basler", "ticker_yahoo": "BSL.DE", "sector": "Technology"},
    {"name": "Befesa", "ticker_yahoo": "BFSA.DE", "sector": "Industrials"},
    {"name": "CANCOM", "ticker_yahoo": "COK.DE", "sector": "Technology"},
    {"name": "Carl Zeiss Meditec", "ticker_yahoo": "AFX.DE", "sector": "Healthcare"},
    {"name": "CEWE", "ticker_yahoo": "CWC.DE", "sector": "Consumer Cyclical"},
    {"name": "Dermapharm", "ticker_yahoo": "DMP.DE", "sector": "Healthcare"},
    {"name": "Deutsche Beteiligungs", "ticker_yahoo": "DBAN.DE", "sector": "Financial Services"},
    {"name": "Deutsche EuroShop", "ticker_yahoo": "DEQ.DE", "sector": "Real Estate"},
    {"name": "Deutsche Pfandbriefbank", "ticker_yahoo": "PBB.DE", "sector": "Financial Services"},
    {"name": "Douglas", "ticker_yahoo": "DOU.DE", "sector": "Consumer Cyclical"},
    {"name": "Drägerwerk", "ticker_yahoo": "DRW3.DE", "sector": "Healthcare"},
    {"name": "Dürr", "ticker_yahoo": "DUE.DE", "sector": "Industrials"},
    {"name": "Eckert & Ziegler", "ticker_yahoo": "EUZ.DE", "sector": "Healthcare"},
    {"name": "Einhell Germany", "ticker_yahoo": "EIN3.DE", "sector": "Consumer Cyclical"},
    {"name": "Energiekontor", "ticker_yahoo": "EKT.DE", "sector": "Utilities"},
    {"name": "Evotec", "ticker_yahoo": "EVT.DE", "sector": "Healthcare"},
    {"name": "Fielmann", "ticker_yahoo": "FIE.DE", "sector": "Consumer Cyclical"},
    {"name": "Friedrich Vorwerk", "ticker_yahoo": "VH2.DE", "sector": "Industrials"},
    {"name": "GFT Technologies", "ticker_yahoo": "GFT.DE", "sector": "Technology"},
    {"name": "Grand City Properties", "ticker_yahoo": "GYC.DE", "sector": "Real Estate"},
    {"name": "Grenke", "ticker_yahoo": "GLJ.DE", "sector": "Financial Services"},
    {"name": "Hamborner REIT", "ticker_yahoo": "HABA.DE", "sector": "Real Estate"},
    {"name": "Heidelberger Druckmaschinen", "ticker_yahoo": "HDP.DE", "sector": "Industrials"},
    {"name": "HelloFresh", "ticker_yahoo": "HFG.DE", "sector": "Consumer Cyclical"},
    {"name": "HORNBACH Holding", "ticker_yahoo": "HBM.DE", "sector": "Consumer Cyclical"},
    {"name": "Hypoport", "ticker_yahoo": "HYQ.DE", "sector": "Financial Services"},
    {"name": "INDUS Holding", "ticker_yahoo": "INH.DE", "sector": "Industrials"},
    {"name": "INIT", "ticker_yahoo": "IXX.DE", "sector": "Technology"},
    {"name": "JOST Werke", "ticker_yahoo": "JST.DE", "sector": "Industrials"},
    {"name": "Jungheinrich", "ticker_yahoo": "JUN3.DE", "sector": "Industrials"},
    {"name": "Klöckner & Co", "ticker_yahoo": "KCO.DE", "sector": "Basic Materials"},
    {"name": "Kontron", "ticker_yahoo": "KTN.DE", "sector": "Technology"},
    {"name": "KSB", "ticker_yahoo": "KSB.DE", "sector": "Industrials"},
    {"name": "KWS SAAT", "ticker_yahoo": "KWS.DE", "sector": "Consumer Defensive"},
    {"name": "LPKF Laser & Electronics", "ticker_yahoo": "LPK.DE", "sector": "Technology"},
    {"name": "MBB", "ticker_yahoo": "MBB.DE", "sector": "Industrials"},
    {"name": "Medios", "ticker_yahoo": "ILM1.DE", "sector": "Healthcare"},
    {"name": "MLP", "ticker_yahoo": "MLP.DE", "sector": "Financial Services"},
    {"name": "Mutares", "ticker_yahoo": "MUX.DE", "sector": "Financial Services"},
    {"name": "Nagarro", "ticker_yahoo": "NA9.DE", "sector": "Technology"},
    {"name": "NORMA Group", "ticker_yahoo": "NOEJ.DE", "sector": "Industrials"},
    {"name": "Ottobock", "ticker_yahoo": "OBCK.DE", "sector": "Healthcare"},
    {"name": "PATRIZIA", "ticker_yahoo": "PAT.DE", "sector": "Real Estate"},
    {"name": "PNE", "ticker_yahoo": "PNE3.DE", "sector": "Utilities"},
    {"name": "PVA TePla", "ticker_yahoo": "TPE.DE", "sector": "Technology"},
    {"name": "Redcare Pharmacy", "ticker_yahoo": "RDC.DE", "sector": "Healthcare"},
    {"name": "SAF-HOLLAND", "ticker_yahoo": "SFQ.DE", "sector": "Industrials"},
    {"name": "SCHOTT Pharma", "ticker_yahoo": "1SXP.DE", "sector": "Healthcare"},
    {"name": "secunet Security Networks", "ticker_yahoo": "YSN.DE", "sector": "Technology"},
    {"name": "SFC Energy", "ticker_yahoo": "F3C.DE", "sector": "Industrials"},
    {"name": "Shelly Group", "ticker_yahoo": "SLYG.DE", "sector": "Technology"},
    {"name": "Sixt", "ticker_yahoo": "SIX2.DE", "sector": "Industrials"},
    {"name": "SMA Solar Technology", "ticker_yahoo": "S92.DE", "sector": "Technology"},
    {"name": "Springer Nature", "ticker_yahoo": "SPG.DE", "sector": "Communication Services"},
    {"name": "Stabilus", "ticker_yahoo": "STM.DE", "sector": "Industrials"},
    {"name": "STO", "ticker_yahoo": "STO3.DE", "sector": "Basic Materials"},
    {"name": "STRATEC", "ticker_yahoo": "SBS.DE", "sector": "Healthcare"},
    {"name": "Ströer", "ticker_yahoo": "SAX.DE", "sector": "Communication Services"},
    {"name": "Südzucker", "ticker_yahoo": "SZU.DE", "sector": "Consumer Defensive"},
    {"name": "TeamViewer", "ticker_yahoo": "TMV.DE", "sector": "Technology"},
    {"name": "tonies", "ticker_yahoo": "TNIE.DE", "sector": "Consumer Cyclical"},
    {"name": "VERBIO", "ticker_yahoo": "VBK.DE", "sector": "Energy"},
    {"name": "Vincorion", "ticker_yahoo": "V1NC.DE", "sector": "Aerospace & Defence"},
    {"name": "Vossloh", "ticker_yahoo": "VOS.DE", "sector": "Industrials"},
    {"name": "Wacker Neuson", "ticker_yahoo": "WAC.DE", "sector": "Industrials"},
]

STATIC_INDEX_CONSTITUENTS = {
    "DAX 40": DAX40_STATIC_CONSTITUENTS,
    "MDAX": MDAX_STATIC_CONSTITUENTS,
    "SDAX": SDAX_STATIC_CONSTITUENTS,
}

INDEX_EXPECTED_COUNTS = {
    "DAX 40": 40,
    "MDAX": 50,
    "SDAX": 70,
    "S&P 500": 500,
}

INDEX_STATIC_AS_OF = "22.06.2026"

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


def deduplicate_dataframe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Entfernt doppelte Spalten robust und behält die zuletzt berechnete Variante.

    Profil- und Score-Spalten können bereits als leere Platzhalter im Rohdaten-Frame
    vorhanden sein. Werden berechnete Spalten anschließend per ``pd.concat`` ergänzt,
    entstehen identische Spaltennamen. Bei ``row.get(...)`` liefert Pandas dann eine
    Series statt eines Einzelwerts. Diese Funktion verhindert genau diesen Zustand.
    """
    if frame is None:
        return frame
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else frame
    if not frame.columns.duplicated().any():
        return frame.copy()
    return frame.loc[:, ~frame.columns.duplicated(keep="last")].copy()


def merge_computed_columns(base: pd.DataFrame, computed: Any) -> pd.DataFrame:
    """Schreibt berechnete Spalten in einen DataFrame und überschreibt Platzhalter.

    Im Gegensatz zu ``pd.concat(..., axis=1)`` werden vorhandene Spalten mit gleichem
    Namen ersetzt. Dadurch bleiben alle Spaltennamen eindeutig – auch bei erneuten
    Streamlit-Reruns, Slideränderungen und separat geladenen Watchlist-Daten.
    """
    result = deduplicate_dataframe_columns(base)
    if computed is None:
        return result
    if isinstance(computed, pd.Series):
        computed = computed.to_frame()
    if not isinstance(computed, pd.DataFrame) or computed.empty:
        return result
    computed = deduplicate_dataframe_columns(computed)
    for column in computed.columns:
        result[column] = computed[column].reindex(result.index)
    return result


def display_text(value: Any, fallback: str = "–") -> str:
    """Bereitet Einzelwerte, Listen und versehentlich gelieferte Series fürs UI auf."""
    if isinstance(value, pd.Series):
        parts = []
        for item in value.tolist():
            rendered = display_text(item, fallback="")
            if rendered and rendered not in parts:
                parts.append(rendered)
        return " | ".join(parts) if parts else fallback
    if isinstance(value, (list, tuple, set)):
        parts = [display_text(item, fallback="") for item in value]
        parts = [part for part in parts if part]
        return " | ".join(parts) if parts else fallback
    if isinstance(value, dict):
        if not value:
            return fallback
        return " | ".join(f"{key}: {display_text(item, fallback='')}" for key, item in value.items())
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text_value = str(value).strip()
    return text_value if text_value else fallback


def safe_session_date(
    value: Any,
    fallback: Any,
    min_date: Any = None,
    max_date: Any = None,
):
    """Liest ein Datum aus dem Streamlit Session State robust ein.

    ``pd.Timestamp(None)`` liefert ``NaT`` statt eine Exception auszulösen.
    Ohne explizite NaT-Prüfung führt ein anschließender Vergleich mit
    ``datetime.date`` zu ``TypeError: Cannot compare NaT with datetime.date``.
    """
    try:
        fallback_date = pd.Timestamp(fallback).date()
    except Exception:
        fallback_date = fallback

    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return fallback_date
        result = pd.Timestamp(parsed).date()
    except Exception:
        return fallback_date

    if min_date is not None and result < min_date:
        return fallback_date
    if max_date is not None and result > max_date:
        return fallback_date
    return result


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


def company_selectbox(
    label: str,
    df: pd.DataFrame,
    key: str,
    *,
    ticker_col: str = "ticker_yahoo",
    name_col: str = "name",
    sort_by_name: bool = True,
    **selectbox_kwargs: Any,
) -> str:
    """Zeigt Firmenname und Ticker, gibt intern aber nur den Ticker zurück.

    Beispiel in der Oberfläche: ``Bayer (BAYN.DE)``. Bestehende Berechnungen
    arbeiten unverändert mit ``BAYN.DE`` weiter. Der Helper räumt außerdem
    veraltete Session-State-Werte nach einem Index- oder Filterwechsel auf.
    """
    if df is None or df.empty or ticker_col not in df.columns:
        raise ValueError("Für die Aktienauswahl sind keine Ticker verfügbar.")

    columns = [ticker_col]
    if name_col in df.columns:
        columns.append(name_col)

    choices = df[columns].copy()
    choices[ticker_col] = choices[ticker_col].fillna("").astype(str).str.strip()
    choices = choices[choices[ticker_col].ne("")].drop_duplicates(subset=[ticker_col])
    if choices.empty:
        raise ValueError("Für die Aktienauswahl sind keine gültigen Ticker verfügbar.")

    if name_col not in choices.columns:
        choices[name_col] = choices[ticker_col]
    else:
        choices[name_col] = choices[name_col].fillna("").astype(str).str.strip()
        choices.loc[choices[name_col].eq(""), name_col] = choices.loc[choices[name_col].eq(""), ticker_col]

    if sort_by_name:
        choices = choices.assign(_sort_name=choices[name_col].str.casefold()).sort_values(
            ["_sort_name", ticker_col], kind="stable"
        )

    options = choices[ticker_col].tolist()
    labels = {
        row[ticker_col]: (
            f"{row[name_col]} ({row[ticker_col]})"
            if row[name_col] != row[ticker_col]
            else row[ticker_col]
        )
        for _, row in choices.iterrows()
    }

    # Nach Index-/Filterwechseln kann ein alter Ticker im Session State liegen.
    # Vor Erzeugung des Widgets entfernen, damit Streamlit keinen ungültigen Wert hält.
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]

    return st.selectbox(
        label,
        options=options,
        format_func=lambda ticker: labels.get(ticker, str(ticker)),
        key=key,
        **selectbox_kwargs,
    )


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
        "cashflow_dividend_coverage", "debt_to_equity", "net_debt_ebitda", "beta",
        "revenue_growth", "earnings_growth", "earnings_quarterly_growth",
        "gross_margin", "ebitda_margin", "current_ratio", "quick_ratio",
        "change_1m", "change_3m", "change_6m", "price_vs_sma50_pct", "price_vs_sma200_pct",
        "growth_score", "growth_coverage", "growth_components",
        "momentum_score", "momentum_coverage", "momentum_components",
        "safety_score", "safety_coverage", "safety_components",
        "data_updated_at",
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


def _valid_local_index(index_name: str) -> Optional[pd.DataFrame]:
    """Lädt eine lokale Indexdatei nur, wenn sie strukturell plausibel ist."""
    local_path = INDEX_LOCAL_FILES.get(index_name)
    if local_path is None or not local_path.exists():
        return None
    try:
        frame = validate_constituents(pd.read_csv(local_path))
    except Exception:
        return None

    expected = INDEX_EXPECTED_COUNTS.get(index_name, 0)
    minimum = expected if index_name in STATIC_INDEX_CONSTITUENTS else max(1, int(expected * 0.8))
    if len(frame) < minimum:
        return None
    return frame.reset_index(drop=True)


def index_source_description(index_name: str) -> str:
    """Kurze, nutzerfreundliche Beschreibung der verwendeten Indexquelle."""
    if _valid_local_index(index_name) is not None:
        return f"Lokale CSV: {INDEX_LOCAL_FILES[index_name]}"
    if index_name in STATIC_INDEX_CONSTITUENTS:
        return f"Integrierte Offline-Liste · Stand {INDEX_STATIC_AS_OF}"
    return "Online-Quelle mit lokalem Cache"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_index_constituents(index_name: str) -> pd.DataFrame:
    """Lädt Indizes robust, ohne deutsche Listen automatisch von Wikipedia abzurufen.

    Reihenfolge:
    1. valide lokale CSV unter data/indices/
    2. integrierte Offline-Liste für DAX, MDAX und SDAX
    3. S&P 500: lokaler Cache, danach Online-Fallback
    """
    local_frame = _valid_local_index(index_name)
    if local_frame is not None:
        return local_frame

    if index_name in STATIC_INDEX_CONSTITUENTS:
        frame = validate_constituents(pd.DataFrame(STATIC_INDEX_CONSTITUENTS[index_name]))
        expected = INDEX_EXPECTED_COUNTS[index_name]
        if len(frame) != expected:
            raise RuntimeError(
                f"Interne {index_name}-Liste ist unvollständig: {len(frame)} statt {expected} Werte."
            )
        return frame.reset_index(drop=True)

    if index_name == "S&P 500":
        cache_path = CACHE_DIR / "sp500_constituents.csv"
        if cache_path.exists():
            try:
                cached = validate_constituents(pd.read_csv(cache_path))
                if len(cached) >= 400:
                    return cached.reset_index(drop=True)
            except Exception:
                pass

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            frame = parse_sp500_tables(read_html_tables(fetch_html(url)))
            try:
                frame.to_csv(cache_path, index=False)
            except Exception:
                pass
            return frame.reset_index(drop=True)
        except Exception as error:
            raise RuntimeError(
                "S&P 500 konnte online nicht geladen werden und es liegt noch kein lokaler Cache vor. "
                "Lege alternativ data/indices/sp500.csv an."
            ) from error

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

        # Wachstum, Liquidität und zusätzliche Qualitätskennzahlen
        "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
        "grossMargins", "ebitdaMargins", "currentRatio", "quickRatio",

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
        "revenue_growth": to_percent(info.get("revenueGrowth")),
        "earnings_growth": to_percent(info.get("earningsGrowth")),
        "earnings_quarterly_growth": to_percent(info.get("earningsQuarterlyGrowth")),
        "gross_margin": to_percent(info.get("grossMargins")),
        "ebitda_margin": to_percent(info.get("ebitdaMargins")),
        "current_ratio": safe_float(info.get("currentRatio")),
        "quick_ratio": safe_float(info.get("quickRatio")),
        "change_1m": None,
        "change_3m": None,
        "change_6m": None,
        "price_vs_sma50_pct": None,
        "price_vs_sma200_pct": None,
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
            record["change_1m"] = series_change(close, 21)
            record["change_3m"] = series_change(close, 63)
            record["change_6m"] = series_change(close, 126)
            record["change_1y"] = series_change(close, 252) if len(close) >= 253 else None

            last_close = safe_float(close.iloc[-1])
            if last_close not in (None, 0):
                if len(close) >= 50:
                    sma50 = safe_float(close.tail(50).mean())
                    if sma50 not in (None, 0):
                        record["price_vs_sma50_pct"] = (last_close / sma50 - 1) * 100
                if len(close) >= 200:
                    sma200 = safe_float(close.tail(200).mean())
                    if sma200 not in (None, 0):
                        record["price_vs_sma200_pct"] = (last_close / sma200 - 1) * 100

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



def _normalised_component_score(
    components: list[tuple[str, Optional[float], float]],
) -> pd.Series:
    """Normalisiert Profilpunkte und weist die Datenabdeckung transparent aus."""
    available = [(name, points, weight) for name, points, weight in components if points is not None]
    total_possible = sum(weight for _, _, weight in components)
    available_weight = sum(weight for _, _, weight in available)
    if available_weight <= 0 or total_possible <= 0:
        return pd.Series({"score": None, "coverage": 0.0, "components": "Keine ausreichenden Daten"})
    raw = sum(float(points) for _, points, _ in available) / available_weight * 100.0
    coverage = available_weight / total_possible * 100.0
    adjusted = raw * (0.65 + 0.35 * coverage / 100.0)
    details = " | ".join(f"{name}: {float(points):.0f}/{weight:.0f}" for name, points, weight in available)
    return pd.Series({"score": round(adjusted, 1), "coverage": round(coverage, 1), "components": details})


def _growth_points(value: Optional[float], max_points: float) -> Optional[float]:
    if value is None:
        return None
    if value >= 25:
        return max_points
    if value >= 15:
        return max_points * 0.85
    if value >= 8:
        return max_points * 0.65
    if value >= 3:
        return max_points * 0.45
    if value >= 0:
        return max_points * 0.25
    if value >= -10:
        return max_points * 0.10
    return 0.0


def compute_growth_score(row: pd.Series) -> pd.Series:
    """Wachstumsprofil aus Umsatz, Ergebnis, Margen und Kapitalrendite."""
    revenue_growth = safe_float(row.get("revenue_growth"))
    earnings_growth = safe_float(row.get("earnings_growth"))
    if earnings_growth is None:
        earnings_growth = safe_float(row.get("earnings_quarterly_growth"))
    operating_margin = safe_float(row.get("operating_margin"))
    gross_margin = safe_float(row.get("gross_margin"))
    roe = safe_float(row.get("roe"))

    margin_points = None
    margin_value = operating_margin if operating_margin is not None else gross_margin
    if margin_value is not None:
        margin_points = 20 if margin_value >= 25 else 16 if margin_value >= 15 else 12 if margin_value >= 8 else 7 if margin_value >= 3 else 2
    roe_points = None
    if roe is not None:
        roe_points = 15 if roe >= 25 else 12 if roe >= 15 else 9 if roe >= 10 else 5 if roe >= 5 else 1

    result = _normalised_component_score([
        ("Umsatzwachstum", _growth_points(revenue_growth, 30), 30),
        ("Ergebniswachstum", _growth_points(earnings_growth, 35), 35),
        ("Marge", margin_points, 20),
        ("ROE", roe_points, 15),
    ])
    return pd.Series({
        "growth_score": result["score"],
        "growth_coverage": result["coverage"],
        "growth_components": result["components"],
    })


def _momentum_return_points(value: Optional[float], max_points: float) -> Optional[float]:
    if value is None:
        return None
    if value >= 20:
        return max_points
    if value >= 10:
        return max_points * 0.8
    if value >= 3:
        return max_points * 0.6
    if value >= 0:
        return max_points * 0.45
    if value >= -5:
        return max_points * 0.25
    if value >= -15:
        return max_points * 0.10
    return 0.0


def compute_momentum_score(row: pd.Series) -> pd.Series:
    """Trendprofil aus mehreren Zeithorizonten und Abstand zu SMA 50/200."""
    sma50 = safe_float(row.get("price_vs_sma50_pct"))
    sma200 = safe_float(row.get("price_vs_sma200_pct"))
    sma_points = None
    if sma50 is not None or sma200 is not None:
        sma_points = 0.0
        if sma50 is not None:
            sma_points += 8 if sma50 >= 5 else 6 if sma50 >= 0 else 2 if sma50 >= -5 else 0
        if sma200 is not None:
            sma_points += 12 if sma200 >= 10 else 9 if sma200 >= 0 else 3 if sma200 >= -10 else 0
        if sma50 is None:
            sma_points = min(sma_points / 12 * 20, 20)
        elif sma200 is None:
            sma_points = min(sma_points / 8 * 20, 20)

    result = _normalised_component_score([
        ("1M-Trend", _momentum_return_points(safe_float(row.get("change_1m")), 15), 15),
        ("3M-Trend", _momentum_return_points(safe_float(row.get("change_3m")), 20), 20),
        ("6M-Trend", _momentum_return_points(safe_float(row.get("change_6m")), 25), 25),
        ("12M-Trend", _momentum_return_points(safe_float(row.get("change_1y")), 20), 20),
        ("SMA 50/200", sma_points, 20),
    ])
    return pd.Series({
        "momentum_score": result["score"],
        "momentum_coverage": result["coverage"],
        "momentum_components": result["components"],
    })


def compute_safety_score(row: pd.Series) -> pd.Series:
    """Sicherheitsprofil aus Schwankung, Drawdown, Bilanz, Cashflow und Ausschüttung."""
    volatility = safe_float(row.get("vol_1y"))
    drawdown = safe_float(row.get("max_drawdown_1y"))
    net_debt = safe_float(row.get("net_debt_ebitda"))
    debt_to_equity = safe_float(row.get("debt_to_equity"))
    payout = safe_float(row.get("payout_ratio"))
    free_cashflow = safe_float(row.get("free_cashflow"))
    operating_cashflow = safe_float(row.get("operating_cashflow"))
    current_ratio = safe_float(row.get("current_ratio"))

    vol_points = None if volatility is None else 25 if volatility <= 18 else 20 if volatility <= 25 else 14 if volatility <= 35 else 7 if volatility <= 50 else 0
    dd_abs = abs(drawdown) if drawdown is not None else None
    drawdown_points = None if dd_abs is None else 20 if dd_abs <= 15 else 16 if dd_abs <= 25 else 10 if dd_abs <= 35 else 4 if dd_abs <= 50 else 0

    debt_points = None
    if is_financial_sector(row.get("sector")):
        if current_ratio is not None:
            debt_points = 20 if current_ratio >= 1.5 else 15 if current_ratio >= 1.1 else 8 if current_ratio >= 0.8 else 2
    elif net_debt is not None:
        debt_points = 20 if net_debt <= 1 else 16 if net_debt <= 2 else 10 if net_debt <= 3 else 4 if net_debt <= 4 else 0
    elif debt_to_equity is not None:
        debt_points = 20 if debt_to_equity <= 0.5 else 16 if debt_to_equity <= 1 else 10 if debt_to_equity <= 2 else 4 if debt_to_equity <= 3 else 0

    payout_points = None
    if payout is not None:
        payout_points = 15 if 0 <= payout <= 60 else 12 if payout <= 80 else 7 if payout <= 100 else 1

    cashflow = free_cashflow if free_cashflow is not None else operating_cashflow
    cashflow_points = None if cashflow is None else 15 if cashflow > 0 else 0
    beta = safe_float(row.get("beta"))
    beta_points = None if beta is None else 5 if beta <= 0.8 else 4 if beta <= 1.0 else 2 if beta <= 1.3 else 0

    result = _normalised_component_score([
        ("Volatilität", vol_points, 25),
        ("Drawdown", drawdown_points, 20),
        ("Verschuldung/Liquidität", debt_points, 20),
        ("Ausschüttung", payout_points, 15),
        ("Cashflow", cashflow_points, 15),
        ("Beta", beta_points, 5),
    ])
    return pd.Series({
        "safety_score": result["score"],
        "safety_coverage": result["coverage"],
        "safety_components": result["components"],
    })


def enrich_with_profile_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return metrics
    result = deduplicate_dataframe_columns(metrics)
    for calculator in (compute_growth_score, compute_momentum_score, compute_safety_score):
        profile = result.apply(calculator, axis=1)
        result = merge_computed_columns(result, profile)
    return result


def enrich_with_scores(
    metrics: pd.DataFrame,
    drawdown_trigger: float,
    payout_max: float,
    score_min: float,
    yield_min: float,
) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return empty_metrics_frame()
    result = deduplicate_dataframe_columns(metrics)
    total_score = result.apply(compute_total_score, axis=1)
    result = merge_computed_columns(result, total_score)
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
    result = merge_computed_columns(result, value)
    return enrich_with_profile_scores(result)


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
    clean_df = deduplicate_dataframe_columns(df)
    sentiment_map = load_recent_sentiment_map(days_back=120)
    scored = clean_df.apply(
        lambda row: compute_special_situation_score(
            row, sentiment_map.get(clean_ticker(row.get("ticker_yahoo")), {})
        ),
        axis=1,
    )
    return merge_computed_columns(clean_df, scored)


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
        "dividend_per_share", "payout_ratio", "dividend_growth_5y", "fcf_dividend_coverage",
        "net_debt_ebitda", "total_score", "score_coverage", "value_score", "value_trigger",
        "growth_score", "growth_coverage", "momentum_score", "momentum_coverage",
        "safety_score", "safety_coverage", "special_situation_score", "special_situation_trigger",
        "vol_1y", "total_return_1y", "max_drawdown_1y",
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


def _sentiment_phrase_present(text: str, phrase: str) -> bool:
    """Prüft normalisierte Phrasen mit Wortgrenzen statt riskanter Teilstrings."""
    normalized_text = f" {normalize_for_search(text)} "
    normalized_phrase = normalize_for_search(phrase)
    return bool(normalized_phrase and f" {normalized_phrase} " in normalized_text)


def analyze_sentiment(title: str, summary: str = "") -> dict[str, Any]:
    """Gewichtete, nachvollziehbare News-Sentiment-Heuristik.

    Überschriften werden doppelt gewichtet, weil RSS-Beschreibungen oft fremde
    Navigationstexte oder Wiederholungen enthalten. Sobald klare positive und
    negative Signale gleichzeitig vorkommen, lautet das Label bewusst
    ``gemischt`` statt einer irreführenden Neutralisierung.
    """
    positive_points = 0
    negative_points = 0
    positive_hits: list[str] = []
    negative_hits: list[str] = []

    for phrase, weight in POSITIVE_SENTIMENT_PHRASES.items():
        if _sentiment_phrase_present(title, phrase):
            positive_points += int(weight) * 2
            positive_hits.append(phrase)
        elif summary and _sentiment_phrase_present(summary, phrase):
            positive_points += int(weight)
            positive_hits.append(phrase)

    for phrase, weight in NEGATIVE_SENTIMENT_PHRASES.items():
        if _sentiment_phrase_present(title, phrase):
            negative_points += int(weight) * 2
            negative_hits.append(phrase)
        elif summary and _sentiment_phrase_present(summary, phrase):
            negative_points += int(weight)
            negative_hits.append(phrase)

    raw_score = max(-10, min(10, positive_points - negative_points))

    if positive_points >= 2 and negative_points >= 2:
        label = "gemischt"
    elif raw_score >= 2:
        label = "positiv"
    elif raw_score <= -2:
        label = "negativ"
    else:
        label = "neutral"

    strongest = max(positive_points, negative_points)
    if label == "gemischt" and min(positive_points, negative_points) >= 4:
        confidence = "hoch"
    elif strongest >= 6:
        confidence = "hoch"
    elif strongest >= 3:
        confidence = "mittel"
    else:
        confidence = "niedrig"

    reason_parts: list[str] = []
    if positive_hits:
        reason_parts.append("Positiv: " + ", ".join(dict.fromkeys(positive_hits[:3])))
    if negative_hits:
        reason_parts.append("Negativ: " + ", ".join(dict.fromkeys(negative_hits[:3])))
    reason = " | ".join(reason_parts) if reason_parts else "Keine eindeutige richtungsweisende Phrase erkannt."

    return {
        "score": int(raw_score),
        "label": label,
        "confidence": confidence,
        "reason": reason,
        "positive_points": int(positive_points),
        "negative_points": int(negative_points),
    }


def simple_sentiment(text: str) -> tuple[int, str]:
    """Kompatibilitäts-Wrapper für ältere Aufrufstellen."""
    result = analyze_sentiment(text, "")
    return int(result["score"]), str(result["label"])


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
                "sentiment_confidence": "niedrig",
                "sentiment_reason": "Legacy-RSS-Auswertung",
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

# Event-Kalender 2.0: optionale offizielle IR-Feeds/Kalender und manuell
# bestätigte Termine. Die CSV-Dateien werden automatisch als Vorlagen angelegt.
IR_SOURCES_PATH = DATA_DIR / "ir_sources.csv"
MANUAL_EVENTS_PATH = DATA_DIR / "manual_events.csv"
SEC_TICKER_CACHE_PATH = CACHE_DIR / "sec_company_tickers.json"

SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "contact@example.com")
SEC_HEADERS = {
    "User-Agent": f"AktienExplorer/5.6 {SEC_CONTACT_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json,text/plain,*/*",
}

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
            "matched_alias", "sentiment_score", "sentiment_label", "sentiment_confidence",
            "sentiment_reason", "event_type", "relevance_score", "relevance_label",
            "relevance_reason", "is_relevant",
        ]
    )


def empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date", "ticker_yahoo", "event_type", "title", "source", "link",
            "sentiment_score", "sentiment_label", "sentiment_confidence",
            "sentiment_reason", "importance", "is_future_event",
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


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {normalize_for_search(text)} "
    normalized_phrase = normalize_for_search(phrase)
    return bool(normalized_phrase and f" {normalized_phrase} " in normalized_text)


def classify_event_from_text(text: str) -> str:
    """Ordnet nur konkrete Finanzereignisse mit Wortgrenzen ein.

    Das verhindert z. B., dass das Kürzel ``AGM`` versehentlich innerhalb des
    Wortes ``Dienstagmittag`` als Hauptversammlung erkannt wird.
    """
    earnings_phrases = [
        "quarterly results", "quarterly earnings", "earnings results", "earnings release",
        "quartalszahlen", "quartalsbericht", "geschaeftszahlen", "jahreszahlen",
        "full year results", "half year results", "halbjahreszahlen",
    ]
    dividend_phrases = [
        "ex dividend", "ex dividende", "dividend payment", "dividendenzahlung",
        "dividend increase", "dividend cut", "dividende", "dividend",
    ]
    meeting_phrases = [
        "hauptversammlung", "annual general meeting", "annual meeting", "agm",
    ]
    report_phrases = [
        "geschaeftsbericht", "annual report", "sustainability report", "nachhaltigkeitsbericht",
    ]
    analyst_phrases = [
        "analyst", "analysts", "analysten", "kursziel", "price target", "rating",
        "upgrade", "downgrade", "outperform", "underperform",
    ]

    if any(_contains_normalized_phrase(text, phrase) for phrase in earnings_phrases):
        return "earnings"
    if any(_contains_normalized_phrase(text, phrase) for phrase in dividend_phrases):
        return "dividend"
    if any(_contains_normalized_phrase(text, phrase) for phrase in meeting_phrases):
        return "annual_meeting"
    if any(_contains_normalized_phrase(text, phrase) for phrase in report_phrases):
        return "report"
    if any(_contains_normalized_phrase(text, phrase) for phrase in analyst_phrases):
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
            sentiment_result = analyze_sentiment(entry["title"], entry["summary"])
            sentiment_score = int(sentiment_result["score"])
            sentiment_label = str(sentiment_result["label"])
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
                    "sentiment_confidence": sentiment_result["confidence"],
                    "sentiment_reason": sentiment_result["reason"],
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
            "sentiment_confidence": eligible.get("sentiment_confidence", "niedrig"),
            "sentiment_reason": eligible.get("sentiment_reason", ""),
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
        "sentiment_confidence": "niedrig",
        "sentiment_reason": "",
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


def sentiment_badge(label: Any) -> str:
    normalized = str(label or "neutral").lower()
    return {
        "positiv": "🟢 Positiv",
        "negativ": "🔴 Negativ",
        "gemischt": "🟡 Gemischt",
        "neutral": "⚪ Neutral",
    }.get(normalized, "⚪ Neutral")


def sentiment_cell_style(value: Any) -> str:
    normalized = str(value or "").lower()
    if "positiv" in normalized:
        return "color: #166534; background-color: #dcfce7; font-weight: 700"
    if "negativ" in normalized:
        return "color: #991b1b; background-color: #fee2e2; font-weight: 700"
    if "gemischt" in normalized:
        return "color: #854d0e; background-color: #fef9c3; font-weight: 700"
    return "color: #475569; background-color: #f1f5f9"


def _event_type_label(value: Any) -> str:
    event_type = str(value or "news")
    icon = {
        "news": "📰",
        "earnings": "📊",
        "dividend": "💶",
        "annual_meeting": "🗳️",
        "report": "📄",
        "analyst": "🎯",
    }.get(event_type, "•")
    label = EVENT_META.get(event_type, {}).get("label", event_type)
    return f"{icon} {label}"


def _friendly_event_date(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "–"
    if timestamp.hour == 0 and timestamp.minute == 0:
        return timestamp.strftime("%d.%m.%Y")
    return timestamp.strftime("%d.%m.%Y, %H:%M")


def _render_event_table(frame: pd.DataFrame, empty_message: str) -> None:
    if frame.empty:
        st.info(empty_message)
        return

    display = frame.copy()
    display["Datum"] = display["date"].map(_friendly_event_date)
    display["Typ"] = display["event_type"].map(_event_type_label)
    display["Titel"] = display.get("title", "").fillna("")
    display["Quelle"] = display.get("source", "").fillna("")
    display["Sentiment"] = display.get("sentiment_label", "neutral").map(sentiment_badge)
    display["Sicherheit"] = display.get("sentiment_confidence", "niedrig").fillna("niedrig").astype(str).str.capitalize()
    display["Begründung"] = display.get("sentiment_reason", "").fillna("")
    display["Artikel"] = display.get("link", "").fillna("")

    visible = ["Datum", "Typ", "Titel", "Quelle", "Sentiment", "Sicherheit", "Begründung", "Artikel"]
    styled = display[visible].style.map(sentiment_cell_style, subset=["Sentiment"])
    try:
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Artikel": st.column_config.LinkColumn("Artikel", display_text="Öffnen"),
                "Titel": st.column_config.TextColumn("Titel", width="large"),
                "Begründung": st.column_config.TextColumn("Warum?", width="large"),
            },
        )
    except (TypeError, AttributeError):
        st.dataframe(styled, use_container_width=True, hide_index=True)


def render_event_calendar(events: pd.DataFrame) -> None:
    st.markdown("#### Ereigniskalender")
    if events is None or events.empty:
        st.info("Noch keine Ereignisse gespeichert. Aktualisiere im News-Tab eine Aktie.")
        return

    display = events.copy()
    display["date"] = pd.to_datetime(display["date"], errors="coerce")
    display = display.dropna(subset=["date"])
    if display.empty:
        st.info("Die gespeicherten Ereignisse enthalten keine gültigen Datumswerte.")
        return

    event_options = sorted(display["event_type"].dropna().astype(str).unique().tolist())
    sentiment_options = ["positiv", "negativ", "gemischt", "neutral"]
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_types = st.multiselect(
            "Ereignistypen",
            options=event_options,
            default=event_options,
            format_func=lambda value: _event_type_label(value),
            key="calendar_event_types",
        )
    with filter_col2:
        selected_sentiments = st.multiselect(
            "Sentiment",
            options=sentiment_options,
            default=sentiment_options,
            format_func=sentiment_badge,
            key="calendar_sentiments",
        )

    if selected_types:
        display = display[display["event_type"].astype(str).isin(selected_types)]
    else:
        display = display.iloc[0:0]
    if selected_sentiments:
        display = display[display.get("sentiment_label", "neutral").fillna("neutral").astype(str).str.lower().isin(selected_sentiments)]
    else:
        display = display.iloc[0:0]

    today = pd.Timestamp.now().normalize()
    future_flags = display.get("is_future_event", False).astype(str).str.lower().isin({"true", "1", "yes", "ja"})
    upcoming_mask = future_flags | (display["date"].dt.normalize() >= today)
    upcoming = display[upcoming_mask].sort_values("date", ascending=True)
    past = display[~upcoming_mask].sort_values("date", ascending=False)

    counts = st.columns(4)
    counts[0].metric("Kommende Termine", len(upcoming))
    counts[1].metric("Positive Ereignisse", int((display.get("sentiment_label", "neutral").astype(str).str.lower() == "positiv").sum()))
    counts[2].metric("Negative Ereignisse", int((display.get("sentiment_label", "neutral").astype(str).str.lower() == "negativ").sum()))
    counts[3].metric("Gemischt/Neutral", int(display.get("sentiment_label", "neutral").astype(str).str.lower().isin(["gemischt", "neutral"]).sum()))

    st.markdown("##### Kommende Termine")
    _render_event_table(upcoming, "Keine kommenden Termine für die gewählten Filter.")

    with st.expander(f"Vergangene Ereignisse ({len(past)})", expanded=upcoming.empty):
        _render_event_table(past, "Keine vergangenen Ereignisse für die gewählten Filter.")

    with st.expander("Wie werden Ereignistyp und Sentiment bestimmt?", expanded=False):
        st.markdown(
            """
- **Ereignistyp** und **Sentiment** sind getrennt: Eine Quartalszahl oder Dividende ist zunächst neutral. Erst Formulierungen wie „Erwartungen übertroffen“, „Prognose angehoben“ oder „Dividende gekürzt“ erzeugen eine Richtung.
- **Positiv**: u. a. Erwartungsübertreffen, Prognoseanhebung, Dividendenerhöhung, Aktienrückkauf, Analysten-Upgrade, Schuldenabbau.
- **Negativ**: u. a. Gewinnwarnung, Prognosesenkung, Dividendenkürzung, Erwartungsverfehlung, Analysten-Downgrade, Klage/Untersuchung oder Insolvenz.
- **Gemischt**: klare positive und negative Signale stehen gleichzeitig in Überschrift/Beschreibung, etwa „Erwartungen übertroffen, aber Prognose gesenkt“.
- **Neutral**: kein eindeutiges Richtungssignal; ein bloßer Termin oder ein allgemeiner Analystenartikel bleibt neutral.

Die Bewertung nutzt nur RSS-Überschrift und -Beschreibung. Ironie, komplexe Zusammenhänge und der vollständige Artikel können dadurch nicht sicher erfasst werden.
            """
        )







def _weighted_portfolio_average(portfolio_view: pd.DataFrame, column: str) -> Optional[float]:
    """Gewichteter Durchschnitt einer Kennzahl mit den aktuellen Portfoliogewichten."""
    if portfolio_view is None or portfolio_view.empty or column not in portfolio_view.columns:
        return None
    values = pd.to_numeric(portfolio_view[column], errors="coerce")
    weights = pd.to_numeric(portfolio_view.get("weight_pct"), errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return None
    weight_sum = float(weights[valid].sum())
    if weight_sum <= 0:
        return None
    return float((values[valid] * weights[valid]).sum() / weight_sum)


def _concentration_points(value: Optional[float], thresholds: list[tuple[float, float]]) -> Optional[float]:
    if value is None:
        return None
    for upper, points in thresholds:
        if value <= upper:
            return float(points)
    return 0.0


def compute_portfolio_health(portfolio_view: pd.DataFrame) -> dict[str, Any]:
    """Berechnet einen nachvollziehbaren Portfolio-Health-Score von 0 bis 100.

    Der Score bewertet Diversifikation, Konzentration sowie die gewichteten
    Qualitäts-, Safety- und Risikokennzahlen. Fehlende Daten werden über eine
    separate Abdeckung ausgewiesen und nicht stillschweigend als gut behandelt.
    """
    empty = {
        "score": None,
        "coverage": 0.0,
        "label": "nicht berechenbar",
        "confidence": "gering",
        "components": pd.DataFrame(),
        "metrics": {},
        "strengths": [],
        "risks": ["Keine ausreichenden Portfoliodaten vorhanden."],
        "profiles": {},
    }
    if portfolio_view is None or portfolio_view.empty:
        return empty

    frame = portfolio_view.copy()
    frame["weight_pct"] = pd.to_numeric(frame.get("weight_pct"), errors="coerce")
    frame = frame.dropna(subset=["weight_pct"])
    frame = frame[frame["weight_pct"] > 0]
    if frame.empty:
        return empty

    weights = frame["weight_pct"] / frame["weight_pct"].sum() * 100.0
    frame["_health_weight"] = weights
    top_position = float(weights.max())
    hhi = float(((weights / 100.0) ** 2).sum())
    effective_positions = float(1.0 / hhi) if hhi > 0 else None

    sector_weights = (
        frame.assign(_sector=frame.get("sector", "Unbekannt").fillna("Unbekannt").astype(str))
        .groupby("_sector", dropna=False)["_health_weight"].sum()
        .sort_values(ascending=False)
    )
    currency_weights = (
        frame.assign(_currency=frame.get("asset_currency", BASE_CURRENCY).fillna(BASE_CURRENCY).astype(str))
        .groupby("_currency", dropna=False)["_health_weight"].sum()
        .sort_values(ascending=False)
    )
    top_sector = float(sector_weights.iloc[0]) if not sector_weights.empty else None
    top_sector_name = str(sector_weights.index[0]) if not sector_weights.empty else "–"
    top_currency = float(currency_weights.iloc[0]) if not currency_weights.empty else None
    top_currency_name = str(currency_weights.index[0]) if not currency_weights.empty else "–"

    top_position_points = _concentration_points(
        top_position,
        [(10, 20), (15, 17), (20, 13), (25, 8), (35, 3)],
    )
    effective_points = _concentration_points(
        effective_positions,
        [(1.5, 0), (2.5, 4), (4, 8), (6, 13), (9, 17), (10_000, 20)],
    )
    # Für die effektive Anzahl gilt: höher ist besser. Die Hilfsfunktion bewertet
    # kleine Werte, deshalb wird hier bewusst direkt klassifiziert.
    if effective_positions is not None:
        effective_points = (
            20.0 if effective_positions >= 10 else
            17.0 if effective_positions >= 7 else
            13.0 if effective_positions >= 5 else
            8.0 if effective_positions >= 3 else
            4.0 if effective_positions >= 2 else 0.0
        )
    position_points = None
    if top_position_points is not None and effective_points is not None:
        position_points = (top_position_points + effective_points) / 2.0

    sector_points = _concentration_points(top_sector, [(25, 15), (35, 12), (45, 8), (60, 4)])
    currency_points = _concentration_points(top_currency, [(45, 10), (60, 8), (75, 5), (90, 2)])

    weighted_safety = _weighted_portfolio_average(frame, "safety_score")
    weighted_quality = _weighted_portfolio_average(frame, "total_score")
    weighted_value = _weighted_portfolio_average(frame, "value_score")
    weighted_growth = _weighted_portfolio_average(frame, "growth_score")
    weighted_momentum = _weighted_portfolio_average(frame, "momentum_score")
    weighted_deep_value = _weighted_portfolio_average(frame, "special_situation_score")
    weighted_volatility = _weighted_portfolio_average(frame, "vol_1y")
    weighted_drawdown = _weighted_portfolio_average(frame, "max_drawdown_1y")

    safety_points = None if weighted_safety is None else max(0.0, min(20.0, weighted_safety / 100.0 * 20.0))
    quality_points = None if weighted_quality is None else max(0.0, min(15.0, weighted_quality / 100.0 * 15.0))
    volatility_points = _concentration_points(weighted_volatility, [(18, 10), (25, 8), (35, 5), (45, 2)])
    drawdown_abs = abs(weighted_drawdown) if weighted_drawdown is not None else None
    drawdown_points = _concentration_points(drawdown_abs, [(15, 10), (25, 8), (35, 5), (50, 2)])

    component_rows = [
        ("Positions-Diversifikation", position_points, 20.0, f"Größte Position {top_position:.1f} %, effektive Positionen {effective_positions:.1f}" if effective_positions is not None else f"Größte Position {top_position:.1f} %"),
        ("Sektor-Diversifikation", sector_points, 15.0, f"Größter Sektor: {top_sector_name} ({top_sector:.1f} %)" if top_sector is not None else "Keine Sektordaten"),
        ("Währungs-Diversifikation", currency_points, 10.0, f"Größte Währung: {top_currency_name} ({top_currency:.1f} %)" if top_currency is not None else "Keine Währungsdaten"),
        ("Safety-Profil", safety_points, 20.0, f"Gewichteter Safety-Score {weighted_safety:.1f}" if weighted_safety is not None else "Safety-Daten fehlen"),
        ("Qualitätsprofil", quality_points, 15.0, f"Gewichteter Qualitäts-Score {weighted_quality:.1f}" if weighted_quality is not None else "Qualitätsdaten fehlen"),
        ("Volatilität", volatility_points, 10.0, f"Gewichtet {weighted_volatility:.1f} %" if weighted_volatility is not None else "Volatilitätsdaten fehlen"),
        ("Drawdown", drawdown_points, 10.0, f"Gewichtet {weighted_drawdown:.1f} %" if weighted_drawdown is not None else "Drawdown-Daten fehlen"),
    ]

    available = [(name, points, maximum, detail) for name, points, maximum, detail in component_rows if points is not None]
    available_max = sum(maximum for _, _, maximum, _ in available)
    total_max = sum(maximum for _, _, maximum, _ in component_rows)
    coverage = available_max / total_max * 100.0 if total_max else 0.0
    raw_score = sum(points for _, points, _, _ in available) / available_max * 100.0 if available_max else None
    score = None if raw_score is None else round(raw_score * (0.70 + 0.30 * coverage / 100.0), 1)

    if score is None:
        label = "nicht berechenbar"
    elif score >= 80:
        label = "sehr robust"
    elif score >= 65:
        label = "solide"
    elif score >= 50:
        label = "gemischt"
    else:
        label = "erhöhte Risiken"

    if coverage >= 85 and len(frame) >= 7:
        confidence = "hoch"
    elif coverage >= 65 and len(frame) >= 4:
        confidence = "mittel"
    else:
        confidence = "gering"

    components = pd.DataFrame([
        {
            "Baustein": name,
            "Punkte": round(points, 1) if points is not None else None,
            "Maximum": maximum,
            "Erfüllung": round(points / maximum * 100.0, 1) if points is not None and maximum else None,
            "Einordnung": detail,
        }
        for name, points, maximum, detail in component_rows
    ])

    strengths: list[str] = []
    risks: list[str] = []
    if top_position <= 15:
        strengths.append(f"Keine einzelne Position überschreitet 15 % (Maximum {top_position:.1f} %).")
    else:
        risks.append(f"Die größte Position wiegt {top_position:.1f} % und erhöht das Einzelwertrisiko.")
    if top_sector is not None and top_sector <= 30:
        strengths.append(f"Die größte Sektorquote liegt mit {top_sector:.1f} % im moderaten Bereich.")
    elif top_sector is not None:
        risks.append(f"Der Sektor {top_sector_name} bündelt {top_sector:.1f} % des Portfolios.")
    if weighted_safety is not None and weighted_safety >= 65:
        strengths.append(f"Der gewichtete Safety-Score ist mit {weighted_safety:.1f} solide.")
    elif weighted_safety is not None and weighted_safety < 45:
        risks.append(f"Der gewichtete Safety-Score ist mit {weighted_safety:.1f} niedrig.")
    if weighted_volatility is not None and weighted_volatility > 35:
        risks.append(f"Die gewichtete Einzeltitel-Volatilität ist mit {weighted_volatility:.1f} % hoch.")
    if effective_positions is not None and effective_positions < 4:
        risks.append(f"Die effektive Diversifikation entspricht nur etwa {effective_positions:.1f} gleichgewichteten Positionen.")
    if coverage < 70:
        risks.append(f"Nur {coverage:.0f} % der Health-Komponenten sind mit Daten belegt.")

    profiles = {
        "Value": weighted_value,
        "Growth": weighted_growth,
        "Momentum": weighted_momentum,
        "Safety": weighted_safety,
        "Deep Value": weighted_deep_value,
        "Qualität": weighted_quality,
    }
    metrics = {
        "positions": int(len(frame)),
        "top_position_pct": top_position,
        "effective_positions": effective_positions,
        "top_sector_pct": top_sector,
        "top_sector_name": top_sector_name,
        "top_currency_pct": top_currency,
        "top_currency_name": top_currency_name,
        "weighted_safety": weighted_safety,
        "weighted_quality": weighted_quality,
        "weighted_volatility": weighted_volatility,
        "weighted_drawdown": weighted_drawdown,
    }
    return {
        "score": score,
        "coverage": round(coverage, 1),
        "label": label,
        "confidence": confidence,
        "components": components,
        "metrics": metrics,
        "strengths": strengths,
        "risks": risks,
        "profiles": profiles,
    }


def build_rebalancing_suggestions(
    portfolio_view: pd.DataFrame,
    max_position_pct: float = 15.0,
    max_sector_pct: float = 30.0,
    max_currency_pct: float = 70.0,
    min_safety_score: float = 40.0,
) -> pd.DataFrame:
    """Erzeugt regelbasierte Prüfhinweise, keine automatischen Handelsaufträge."""
    columns = ["Priorität", "Kategorie", "Betroffen", "Aktuell", "Grenze", "Prüfhinweis", "Richtwert EUR"]
    if portfolio_view is None or portfolio_view.empty:
        return pd.DataFrame(columns=columns)

    frame = portfolio_view.copy()
    frame["weight_pct"] = pd.to_numeric(frame.get("weight_pct"), errors="coerce")
    frame["market_value_eur"] = pd.to_numeric(frame.get("market_value_eur"), errors="coerce")
    total_value = float(frame["market_value_eur"].sum(skipna=True))
    rows: list[dict[str, Any]] = []

    for _, row in frame.dropna(subset=["weight_pct"]).iterrows():
        weight = float(row["weight_pct"])
        if weight > max_position_pct:
            excess = max(0.0, total_value * (weight - max_position_pct) / 100.0)
            rows.append({
                "Priorität": "hoch" if weight >= max_position_pct + 10 else "mittel",
                "Kategorie": "Einzelposition",
                "Betroffen": f"{row.get('name', row.get('ticker_yahoo', '–'))} ({row.get('ticker_yahoo', '–')})",
                "Aktuell": f"{weight:.1f} %",
                "Grenze": f"{max_position_pct:.1f} %",
                "Prüfhinweis": "Positionsgröße und Investmentthese prüfen; eine Reduktion wäre eine mögliche, aber nicht zwingende Maßnahme.",
                "Richtwert EUR": round(excess, 2),
            })

    sector_group = (
        frame.assign(_sector=frame.get("sector", "Unbekannt").fillna("Unbekannt").astype(str))
        .groupby("_sector", dropna=False)["weight_pct"].sum()
        .sort_values(ascending=False)
    )
    for sector, weight in sector_group.items():
        weight = float(weight)
        if weight > max_sector_pct:
            excess = max(0.0, total_value * (weight - max_sector_pct) / 100.0)
            rows.append({
                "Priorität": "hoch" if weight >= max_sector_pct + 15 else "mittel",
                "Kategorie": "Sektor",
                "Betroffen": str(sector),
                "Aktuell": f"{weight:.1f} %",
                "Grenze": f"{max_sector_pct:.1f} %",
                "Prüfhinweis": "Sektorkonzentration prüfen; neue Käufe könnten bevorzugt andere Sektoren ergänzen.",
                "Richtwert EUR": round(excess, 2),
            })

    currency_group = (
        frame.assign(_currency=frame.get("asset_currency", BASE_CURRENCY).fillna(BASE_CURRENCY).astype(str))
        .groupby("_currency", dropna=False)["weight_pct"].sum()
        .sort_values(ascending=False)
    )
    for currency, weight in currency_group.items():
        weight = float(weight)
        if weight > max_currency_pct:
            excess = max(0.0, total_value * (weight - max_currency_pct) / 100.0)
            rows.append({
                "Priorität": "mittel",
                "Kategorie": "Währung",
                "Betroffen": str(currency),
                "Aktuell": f"{weight:.1f} %",
                "Grenze": f"{max_currency_pct:.1f} %",
                "Prüfhinweis": "Währungskonzentration und persönliche Ausgabenwährung prüfen; dies ist nicht automatisch ein Nachteil.",
                "Richtwert EUR": round(excess, 2),
            })

    if "safety_score" in frame.columns:
        safety_values = pd.to_numeric(frame["safety_score"], errors="coerce")
        for index, row in frame[safety_values.notna() & (safety_values < min_safety_score) & (frame["weight_pct"] >= 3)].iterrows():
            safety = float(safety_values.loc[index])
            rows.append({
                "Priorität": "hoch" if safety < 25 else "mittel",
                "Kategorie": "Safety",
                "Betroffen": f"{row.get('name', row.get('ticker_yahoo', '–'))} ({row.get('ticker_yahoo', '–')})",
                "Aktuell": f"Safety {safety:.1f} / Gewicht {float(row.get('weight_pct')):.1f} %",
                "Grenze": f"Safety ≥ {min_safety_score:.0f}",
                "Prüfhinweis": "Bilanz, Cashflow, Ausschüttung und Drawdown dieser Position vertieft prüfen.",
                "Richtwert EUR": None,
            })

    if len(frame) < 5:
        rows.append({
            "Priorität": "hoch" if len(frame) <= 2 else "mittel",
            "Kategorie": "Diversifikation",
            "Betroffen": "Gesamtportfolio",
            "Aktuell": f"{len(frame)} Positionen",
            "Grenze": "mindestens 5 als grober Orientierungswert",
            "Prüfhinweis": "Die geringe Zahl an Positionen erhöht das unternehmensspezifische Risiko.",
            "Richtwert EUR": None,
        })

    suggestions = pd.DataFrame(rows, columns=columns)
    if suggestions.empty:
        return suggestions
    priority_order = {"hoch": 0, "mittel": 1, "niedrig": 2}
    suggestions["_order"] = suggestions["Priorität"].map(priority_order).fillna(9)
    return suggestions.sort_values(["_order", "Kategorie", "Betroffen"]).drop(columns="_order").reset_index(drop=True)


def render_portfolio_health(portfolio_view: pd.DataFrame) -> None:
    st.markdown("#### Portfolio Health Score & Rebalancing")
    st.caption(
        "Der Health Score bündelt Diversifikation, Konzentration, Safety, Qualität, Volatilität und Drawdown. "
        "Die Hinweise sind regelbasierte Research-Hilfen und keine automatischen Kauf- oder Verkaufsaufträge."
    )

    health = compute_portfolio_health(portfolio_view)
    score = health.get("score")
    metrics = health.get("metrics", {})
    score_columns = st.columns(6)
    score_columns[0].metric("Health Score", format_number(score, 1) if score is not None else "–")
    score_columns[1].metric("Einordnung", str(health.get("label", "–")))
    score_columns[2].metric("Datenabdeckung", format_percent(health.get("coverage"), 0))
    score_columns[3].metric("Aussagekraft", str(health.get("confidence", "–")))
    score_columns[4].metric("Größte Position", format_percent(metrics.get("top_position_pct"), 1))
    score_columns[5].metric("Effektive Positionen", format_number(metrics.get("effective_positions"), 1))

    if score is None:
        st.warning("Der Portfolio Health Score konnte mangels Daten nicht berechnet werden.")
        return
    if score >= 80:
        st.success("Das Portfolio wirkt nach den gewählten Regeln breit und vergleichsweise robust aufgestellt.")
    elif score >= 65:
        st.info("Das Portfolio wirkt insgesamt solide, einzelne Konzentrationen oder Risikofaktoren sollten dennoch geprüft werden.")
    elif score >= 50:
        st.warning("Das Portfolio zeigt ein gemischtes Risikoprofil. Die unten genannten Konzentrationen sollten bewusst entschieden sein.")
    else:
        st.error("Das Portfolio weist nach den aktuellen Regeln erhöhte Konzentrations- oder Qualitätsrisiken auf.")

    profile_data = pd.DataFrame([
        {"Profil": name, "Score": value}
        for name, value in health.get("profiles", {}).items()
        if value is not None
    ])
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("##### Health-Bausteine")
        components = health.get("components", pd.DataFrame())
        if not components.empty:
            st.dataframe(
                components.style.format({"Punkte": "{:.1f}", "Maximum": "{:.0f}", "Erfüllung": "{:.1f}%"}, na_rep="–"),
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.markdown("##### Gewichtetes Aktienprofil")
        if not profile_data.empty:
            profile_chart = alt.Chart(profile_data).mark_bar().encode(
                x=alt.X("Score:Q", title="Score", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Profil:N", sort="-x", title=None),
                tooltip=["Profil:N", alt.Tooltip("Score:Q", format=".1f")],
            ).properties(height=max(220, 36 * len(profile_data)))
            st.altair_chart(profile_chart, use_container_width=True)
        else:
            st.info("Für das gewichtete Profil fehlen noch Score-Daten.")

    strengths = health.get("strengths", [])
    risks = health.get("risks", [])
    text_columns = st.columns(2)
    with text_columns[0]:
        st.markdown("##### Stärken")
        if strengths:
            for item in strengths:
                st.markdown(f"- ✓ {item}")
        else:
            st.caption("Keine eindeutige Stärke aus den verfügbaren Daten abgeleitet.")
    with text_columns[1]:
        st.markdown("##### Prüfpunkte")
        if risks:
            for item in risks:
                st.markdown(f"- ⚠ {item}")
        else:
            st.caption("Keine auffälligen Prüfpunkte aus den gewählten Regeln.")

    with st.expander("Grenzwerte für Rebalancing-Hinweise", expanded=False):
        controls = st.columns(4)
        max_position = controls[0].slider("Max. Einzelposition (%)", 5.0, 40.0, 15.0, 1.0, key="health_max_position")
        max_sector = controls[1].slider("Max. Sektor (%)", 15.0, 70.0, 30.0, 1.0, key="health_max_sector")
        max_currency = controls[2].slider("Max. Währung (%)", 30.0, 100.0, 70.0, 1.0, key="health_max_currency")
        min_safety = controls[3].slider("Min. Safety-Score", 0.0, 80.0, 40.0, 5.0, key="health_min_safety")

    suggestions = build_rebalancing_suggestions(
        portfolio_view,
        max_position_pct=float(max_position),
        max_sector_pct=float(max_sector),
        max_currency_pct=float(max_currency),
        min_safety_score=float(min_safety),
    )
    st.markdown("##### Rebalancing- und Prüfhinweise")
    if suggestions.empty:
        st.success("Unter den gewählten Grenzwerten wurde kein konkreter Rebalancing-Hinweis ausgelöst.")
    else:
        display = suggestions.copy()
        display["Richtwert EUR"] = display["Richtwert EUR"].map(lambda value: format_eur(value, 0) if safe_float(value) is not None else "–")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            "Der Richtwert zeigt nur den rechnerischen Betrag oberhalb des gewählten Grenzwerts. "
            "Er berücksichtigt weder Steuern noch Gebühren, persönliche Ziele oder die Qualität möglicher Ersatzanlagen."
        )
        st.download_button(
            "Prüfhinweise als CSV herunterladen",
            data=suggestions.to_csv(index=False).encode("utf-8"),
            file_name="portfolio_health_hinweise.csv",
            mime="text/csv",
            key="download_portfolio_health_suggestions",
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





# -----------------------------------------------------------------------------
# Historisches Deep-Value-Backtesting 2.0
# -----------------------------------------------------------------------------


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_long_history(ticker: str) -> pd.DataFrame:
    """Lädt eine lange Preisreihe für eine einzelne historische Fallstudie."""
    symbol = clean_ticker(ticker)
    if not symbol:
        return pd.DataFrame()
    try:
        history = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=False)
    except Exception:
        history = pd.DataFrame()
    if history is None or history.empty:
        return pd.DataFrame()
    history = ensure_datetime_index(history)
    columns = [column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in history.columns]
    return history[columns].copy()


def _fetch_statement(ticker_object: Any, getter_name: str, fallback_name: str) -> pd.DataFrame:
    """Ruft einen Yahoo-Jahresabschluss robust über neue und ältere yfinance-APIs ab."""
    statement = pd.DataFrame()
    getter = getattr(ticker_object, getter_name, None)
    if callable(getter):
        for kwargs in ({"freq": "yearly"}, {}):
            try:
                candidate = getter(**kwargs)
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    statement = candidate.copy()
                    break
            except Exception:
                continue
    if statement.empty:
        try:
            candidate = getattr(ticker_object, fallback_name)
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                statement = candidate.copy()
        except Exception:
            pass
    return statement


def _normalise_statement_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _statement_values(statement: pd.DataFrame, aliases: list[str]) -> dict[pd.Timestamp, float]:
    """Extrahiert eine Kennzahlenzeile unabhängig von Yahoo-Bezeichnungsvarianten."""
    if statement is None or statement.empty:
        return {}
    normalised_aliases = [_normalise_statement_label(alias) for alias in aliases]
    selected_index: Any = None
    for index_value in statement.index:
        label = _normalise_statement_label(index_value)
        if label in normalised_aliases:
            selected_index = index_value
            break
    if selected_index is None:
        for index_value in statement.index:
            label = _normalise_statement_label(index_value)
            if any(alias and (alias in label or label in alias) for alias in normalised_aliases):
                selected_index = index_value
                break
    if selected_index is None:
        return {}

    values: dict[pd.Timestamp, float] = {}
    row = statement.loc[selected_index]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    for column, value in row.items():
        date_value = pd.to_datetime(column, errors="coerce")
        number = safe_float(value)
        if pd.notna(date_value) and number is not None:
            values[pd.Timestamp(date_value).tz_localize(None).normalize()] = number
    return values


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_historical_financials(ticker: str) -> pd.DataFrame:
    """Lädt verfügbare Jahresabschlüsse und vereinheitlicht zentrale BAT-Kennzahlen.

    Wichtig: Yahoo stellt keine garantierte Point-in-Time-Datenbank bereit. Die
    Werte können nachträglich angepasst worden sein. Im Backtest wird deshalb
    zusätzlich ein konfigurierbarer Veröffentlichungsverzug verwendet.
    """
    symbol = clean_ticker(ticker)
    columns = [
        "fiscal_date", "revenue", "net_income", "diluted_eps", "ebitda",
        "operating_cashflow", "capital_expenditure", "free_cashflow",
        "cash_dividends_paid", "total_debt", "cash", "diluted_average_shares",
        "net_debt", "net_debt_ebitda", "cashflow_dividend_coverage",
    ]
    if not symbol:
        return pd.DataFrame(columns=columns)

    try:
        ticker_object = yf.Ticker(symbol)
        income = _fetch_statement(ticker_object, "get_income_stmt", "income_stmt")
        cashflow = _fetch_statement(ticker_object, "get_cash_flow", "cashflow")
        balance = _fetch_statement(ticker_object, "get_balance_sheet", "balance_sheet")
    except Exception:
        return pd.DataFrame(columns=columns)

    series_by_field = {
        "revenue": _statement_values(income, ["Total Revenue", "Operating Revenue"]),
        "net_income": _statement_values(income, ["Net Income Common Stockholders", "Net Income", "Net Income Continuous Operations"]),
        "diluted_eps": _statement_values(income, ["Diluted EPS", "Basic EPS"]),
        "ebitda": _statement_values(income, ["Normalized EBITDA", "EBITDA"]),
        "diluted_average_shares": _statement_values(income, ["Diluted Average Shares", "Basic Average Shares"]),
        "operating_cashflow": _statement_values(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"]),
        "capital_expenditure": _statement_values(cashflow, ["Capital Expenditure", "Capital Expenditures"]),
        "free_cashflow": _statement_values(cashflow, ["Free Cash Flow"]),
        "cash_dividends_paid": _statement_values(cashflow, ["Cash Dividends Paid", "Common Stock Dividend Paid", "Cash Dividends Paid To Common Stockholders"]),
        "total_debt": _statement_values(balance, ["Total Debt"]),
        "cash": _statement_values(balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash Financial"]),
    }

    all_dates = sorted({date for values in series_by_field.values() for date in values})
    rows: list[dict[str, Any]] = []
    for fiscal_date in all_dates:
        row: dict[str, Any] = {"fiscal_date": fiscal_date}
        for field, values in series_by_field.items():
            row[field] = values.get(fiscal_date)

        operating_cashflow = safe_float(row.get("operating_cashflow"))
        capital_expenditure = safe_float(row.get("capital_expenditure"))
        free_cashflow = safe_float(row.get("free_cashflow"))
        if free_cashflow is None and operating_cashflow is not None and capital_expenditure is not None:
            # Capex wird von Yahoo meist negativ ausgewiesen; bei positiven Werten abziehen.
            free_cashflow = operating_cashflow + capital_expenditure if capital_expenditure < 0 else operating_cashflow - capital_expenditure
        row["free_cashflow"] = free_cashflow

        debt = safe_float(row.get("total_debt"))
        cash = safe_float(row.get("cash"))
        ebitda = safe_float(row.get("ebitda"))
        net_debt = None if debt is None else debt - (cash or 0.0)
        row["net_debt"] = net_debt
        row["net_debt_ebitda"] = net_debt / ebitda if net_debt is not None and ebitda not in (None, 0) else None

        dividends_paid = safe_float(row.get("cash_dividends_paid"))
        row["cashflow_dividend_coverage"] = (
            free_cashflow / abs(dividends_paid)
            if free_cashflow is not None and dividends_paid not in (None, 0)
            else None
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["fiscal_date"] = pd.to_datetime(result["fiscal_date"], errors="coerce")
    result = result.dropna(subset=["fiscal_date"]).sort_values("fiscal_date").drop_duplicates("fiscal_date", keep="last")
    for column in columns:
        if column not in result.columns:
            result[column] = None
    return result[columns].reset_index(drop=True)


def _nearest_series_value(series: pd.Series, target: pd.Timestamp, direction: str = "nearest", tolerance_days: int = 10) -> tuple[Optional[pd.Timestamp], Optional[float]]:
    if series is None or series.empty:
        return None, None
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return None, None
    target = pd.Timestamp(target).tz_localize(None).normalize()
    index = clean.index
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
        clean.index = index

    if direction == "backward":
        candidates = clean[clean.index <= target]
        if candidates.empty:
            return None, None
        date_value = candidates.index[-1]
    elif direction == "forward":
        candidates = clean[clean.index >= target]
        if candidates.empty:
            return None, None
        date_value = candidates.index[0]
    else:
        position = index.get_indexer([target], method="nearest")[0]
        if position < 0:
            return None, None
        date_value = index[position]

    if abs((pd.Timestamp(date_value) - target).days) > tolerance_days:
        return None, None
    return pd.Timestamp(date_value), safe_float(clean.loc[date_value])


def _trailing_dividend_sum(dividends: pd.DataFrame, as_of: pd.Timestamp, days: int = 365) -> Optional[float]:
    if dividends is None or dividends.empty:
        return None
    frame = dividends.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.dropna(subset=["date", "amount"])
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=days)
    value = frame.loc[(frame["date"] > cutoff) & (frame["date"] <= pd.Timestamp(as_of)), "amount"].sum()
    return safe_float(value) if value > 0 else None


def _historical_yield_average(dividends: pd.DataFrame, close: pd.Series, as_of: pd.Timestamp, years: int = 5) -> Optional[float]:
    if dividends is None or dividends.empty or close is None or close.empty:
        return None
    start = pd.Timestamp(as_of) - pd.DateOffset(years=years)
    yields: list[float] = []
    for year in range(start.year, pd.Timestamp(as_of).year):
        year_end = pd.Timestamp(year=year, month=12, day=31)
        dps = _trailing_dividend_sum(dividends, year_end)
        _, price = _nearest_series_value(close, year_end, direction="backward", tolerance_days=12)
        if dps is not None and price not in (None, 0):
            yields.append(dps / price * 100)
    return sum(yields) / len(yields) if yields else None


def _available_financial_snapshot(financials: pd.DataFrame, as_of: pd.Timestamp, lag_days: int) -> Optional[pd.Series]:
    if financials is None or financials.empty:
        return None
    frame = financials.copy()
    frame["available_date"] = pd.to_datetime(frame["fiscal_date"], errors="coerce") + pd.to_timedelta(int(lag_days), unit="D")
    available = frame[frame["available_date"] <= pd.Timestamp(as_of)]
    if available.empty:
        return None
    return available.sort_values("available_date").iloc[-1]


def historical_bat_snapshot(
    ticker: str,
    as_of: Any,
    history: pd.DataFrame,
    dividends: pd.DataFrame,
    financials: pd.DataFrame,
    info_lag_days: int = 120,
    negative_news_shock: bool = False,
    special_effect_suspected: bool = False,
) -> dict[str, Any]:
    """Erstellt eine historische, konservativ verzögerte BAT-Pattern-Momentaufnahme."""
    date_value = pd.Timestamp(as_of).tz_localize(None).normalize()
    result: dict[str, Any] = {"ticker_yahoo": clean_ticker(ticker), "as_of": date_value}
    if history is None or history.empty or "Close" not in history.columns:
        result["error"] = "Keine historische Preisreihe verfügbar."
        return result

    frame = ensure_datetime_index(history.copy())
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    adjusted = pd.to_numeric(frame.get("Adj Close", frame["Close"]), errors="coerce").dropna()
    price_date, price = _nearest_series_value(close, date_value, direction="backward", tolerance_days=10)
    if price is None or price_date is None:
        result["error"] = "Am gewählten Stichtag wurde kein naher Handelstag gefunden."
        return result

    result["price_date"] = price_date
    result["price"] = price
    result["currency"] = (fetch_ticker_info(clean_ticker(ticker)).get("currency") or "–")

    trailing_dividend = _trailing_dividend_sum(dividends, price_date)
    result["trailing_dividend"] = trailing_dividend
    result["dividend_yield"] = trailing_dividend / price * 100 if trailing_dividend is not None and price else None
    average_yield = _historical_yield_average(dividends, close, price_date, years=5)
    result["dividend_yield_5y_avg"] = average_yield
    result["dividend_yield_vs_5y_avg_pct"] = (
        (result["dividend_yield"] / average_yield - 1) * 100
        if result.get("dividend_yield") is not None and average_yield not in (None, 0)
        else None
    )

    historical_close = close[close.index <= price_date]
    for years, column in ((1, "drawdown_1y_high_pct"), (3, "drawdown_3y_high_pct"), (5, "drawdown_5y_high_pct")):
        window = historical_close[historical_close.index >= price_date - pd.DateOffset(years=years)]
        high_value = safe_float(window.max()) if not window.empty else None
        result[column] = (price / high_value - 1) * 100 if high_value not in (None, 0) else None
    drawdowns = [safe_float(result.get("drawdown_3y_high_pct")), safe_float(result.get("drawdown_5y_high_pct"))]
    drawdowns = [value for value in drawdowns if value is not None]
    result["deepest_drawdown_3_5y_pct"] = min(drawdowns) if drawdowns else None

    one_year = adjusted[(adjusted.index <= price_date) & (adjusted.index >= price_date - pd.DateOffset(years=1))]
    returns = one_year.pct_change().dropna()
    result["volatility_1y"] = float(returns.std() * (252 ** 0.5) * 100) if len(returns) >= 30 else None

    annual = _available_financial_snapshot(financials, price_date, info_lag_days)
    result["financial_fiscal_date"] = annual.get("fiscal_date") if annual is not None else None
    for field in [
        "net_income", "diluted_eps", "operating_cashflow", "free_cashflow",
        "cashflow_dividend_coverage", "net_debt_ebitda", "ebitda", "total_debt", "cash",
    ]:
        result[field] = annual.get(field) if annual is not None else None

    cashflow_positive = (
        (safe_float(result.get("free_cashflow")) is not None and safe_float(result.get("free_cashflow")) > 0)
        or (safe_float(result.get("operating_cashflow")) is not None and safe_float(result.get("operating_cashflow")) > 0)
    )
    result["cashflow_positive"] = cashflow_positive
    eps = safe_float(result.get("diluted_eps"))
    result["reported_eps_negative"] = eps is not None and eps < 0
    result["negative_news_shock"] = bool(negative_news_shock)
    result["special_effect_suspected"] = bool(special_effect_suspected)

    score = 0.0
    components: list[dict[str, Any]] = []

    dividend_yield = safe_float(result.get("dividend_yield"))
    if dividend_yield is None:
        points = 0
    elif dividend_yield >= 7:
        points = 20
    elif dividend_yield >= 5:
        points = 15
    elif dividend_yield >= 3.5:
        points = 9
    else:
        points = 0
    score += points
    components.append({"Kriterium": "Dividendenrendite", "Punkte": points, "Maximum": 20, "Messwert": format_percent(dividend_yield, 2)})

    drawdown = safe_float(result.get("deepest_drawdown_3_5y_pct"))
    drawdown_abs = abs(min(drawdown, 0)) if drawdown is not None else 0
    points = 20 if drawdown_abs >= 35 else 15 if drawdown_abs >= 25 else 8 if drawdown_abs >= 15 else 0
    score += points
    components.append({"Kriterium": "Drawdown vom 3J-/5J-Hoch", "Punkte": points, "Maximum": 20, "Messwert": format_percent(drawdown, 1, signed=True)})

    points = 15 if cashflow_positive else 0
    score += points
    components.append({"Kriterium": "Cashflow positiv", "Punkte": points, "Maximum": 15, "Messwert": "Ja" if cashflow_positive else "Nein / keine Daten"})

    coverage = safe_float(result.get("cashflow_dividend_coverage"))
    points = 15 if coverage is not None and coverage >= 1.2 else 11 if coverage is not None and coverage >= 1.0 else 5 if coverage is not None and coverage >= 0.8 else 0
    score += points
    components.append({"Kriterium": "FCF deckt Dividende", "Punkte": points, "Maximum": 15, "Messwert": f"{format_number(coverage, 2)}x"})

    leverage = safe_float(result.get("net_debt_ebitda"))
    points = 10 if leverage is not None and leverage < 3.5 else 5 if leverage is not None and leverage < 5 else 0
    score += points
    components.append({"Kriterium": "Net Debt / EBITDA", "Punkte": points, "Maximum": 10, "Messwert": f"{format_number(leverage, 2)}x"})

    pe_approx = None
    if eps not in (None, 0):
        historical_price_for_pe = price
        try:
            info = fetch_ticker_info(clean_ticker(ticker))
            raw_price_currency = str(info.get("currency") or "")
            financial_currency = str(info.get("financialCurrency") or "").upper()
            if raw_price_currency in {"GBp", "GBX", "GBPence"} and financial_currency == "GBP" and price > 100:
                historical_price_for_pe = price / 100.0
        except Exception:
            pass
        pe_approx = historical_price_for_pe / eps
    result["pe_approx"] = pe_approx

    if special_effect_suspected and cashflow_positive:
        points = 10
        valuation_text = "Sondereffekt manuell markiert"
    elif result.get("reported_eps_negative") and cashflow_positive:
        points = 8
        valuation_text = "EPS negativ, Cashflow positiv"
    elif pe_approx is not None and 0 < pe_approx <= 12:
        points = 10
        valuation_text = f"KGV ca. {format_number(pe_approx, 1)}"
    elif pe_approx is not None and 0 < pe_approx <= 18:
        points = 6
        valuation_text = f"KGV ca. {format_number(pe_approx, 1)}"
    else:
        points = 0
        valuation_text = "keine günstige/verzerrte Bewertung erkannt"
    score += points
    components.append({"Kriterium": "Bewertung / Sondereffekt", "Punkte": points, "Maximum": 10, "Messwert": valuation_text})

    points = 10 if negative_news_shock and cashflow_positive else 0
    score += points
    components.append({"Kriterium": "Negativer News-Schock bei stabilem Cashflow", "Punkte": points, "Maximum": 10, "Messwert": "Manuell bestätigt" if negative_news_shock else "Nicht berücksichtigt"})

    result["bat_pattern_score"] = round(min(max(score, 0), 100), 1)
    result["bat_pattern_trigger"] = bool(
        score >= 70
        and dividend_yield is not None and dividend_yield >= 5
        and drawdown is not None and drawdown <= -25
        and cashflow_positive
    )
    result["components"] = pd.DataFrame(components)
    return result



def _future_value_trap_metrics(
    history: pd.DataFrame,
    dividends: pd.DataFrame,
    financials: pd.DataFrame,
    entry_date: pd.Timestamp,
    info_lag_days: int,
    use_adjusted: bool = True,
) -> dict[str, Any]:
    """Bewertet nachträglich typische Value-Trap-Signale über zwölf Monate.

    Die Prüfung nutzt ausschließlich Daten nach dem Signal und beeinflusst den
    historischen Score nicht. Sie dient nur der späteren Ergebnisbewertung.
    """
    result: dict[str, Any] = {
        "max_decline_12m": None,
        "dividend_change_12m": None,
        "future_cashflow_positive": None,
        "value_trap": False,
        "value_trap_reason": "",
    }
    if history is None or history.empty:
        return result

    frame = ensure_datetime_index(history.copy())
    price_column = "Adj Close" if use_adjusted and "Adj Close" in frame.columns else "Close"
    if price_column not in frame.columns:
        return result

    prices = pd.to_numeric(frame[price_column], errors="coerce").dropna()
    entry_ts = pd.Timestamp(entry_date).tz_localize(None).normalize()
    entry_price_date, entry_price = _nearest_series_value(
        prices, entry_ts, direction="backward", tolerance_days=10
    )
    if entry_price_date is None or entry_price in (None, 0):
        return result

    target_12m = pd.Timestamp(entry_price_date) + pd.DateOffset(months=12)
    future_window = prices[(prices.index > entry_price_date) & (prices.index <= target_12m)]
    if not future_window.empty:
        result["max_decline_12m"] = (float(future_window.min()) / float(entry_price) - 1) * 100

    dividend_at_entry = _trailing_dividend_sum(dividends, pd.Timestamp(entry_price_date))
    dividend_after_12m = _trailing_dividend_sum(dividends, target_12m)
    if dividend_at_entry not in (None, 0) and dividend_after_12m is not None:
        result["dividend_change_12m"] = (
            float(dividend_after_12m) / float(dividend_at_entry) - 1
        ) * 100

    future_snapshot = _available_financial_snapshot(financials, target_12m, info_lag_days)
    if future_snapshot is not None:
        future_fcf = safe_float(future_snapshot.get("free_cashflow"))
        future_ocf = safe_float(future_snapshot.get("operating_cashflow"))
        if future_fcf is not None or future_ocf is not None:
            result["future_cashflow_positive"] = bool(
                (future_fcf is not None and future_fcf > 0)
                or (future_ocf is not None and future_ocf > 0)
            )

    reasons: list[str] = []
    if safe_float(result.get("max_decline_12m")) is not None and float(result["max_decline_12m"]) <= -30:
        reasons.append("nach dem Signal weitere mindestens 30 % gefallen")
    if safe_float(result.get("dividend_change_12m")) is not None and float(result["dividend_change_12m"]) <= -20:
        reasons.append("Ausschüttung innerhalb von zwölf Monaten deutlich gesunken")
    if result.get("future_cashflow_positive") is False:
        reasons.append("später verfügbarer Cashflow nicht mehr positiv")

    result["value_trap"] = bool(reasons)
    result["value_trap_reason"] = "; ".join(reasons)
    return result


def _component_summary(snapshot: dict[str, Any]) -> str:
    components = snapshot.get("components", pd.DataFrame())
    if not isinstance(components, pd.DataFrame) or components.empty:
        return ""
    parts: list[str] = []
    for _, row in components.iterrows():
        criterion = str(row.get("Kriterium", "Kriterium"))
        points = safe_float(row.get("Punkte")) or 0
        maximum = safe_float(row.get("Maximum")) or 0
        measurement = str(row.get("Messwert", "–"))
        parts.append(f"{criterion}: {points:.0f}/{maximum:.0f} ({measurement})")
    return " | ".join(parts)


def forward_performance_table(
    history: pd.DataFrame,
    benchmark: pd.DataFrame,
    entry_date: pd.Timestamp,
    use_adjusted: bool = True,
) -> pd.DataFrame:
    """Berechnet nachgelagerte 3/6/12/24-Monatsrenditen ohne Zukunftsdaten im Signal."""
    if history is None or history.empty:
        return pd.DataFrame()
    stock_column = "Adj Close" if use_adjusted and "Adj Close" in history.columns else "Close"
    benchmark_column = "Adj Close" if use_adjusted and benchmark is not None and "Adj Close" in benchmark.columns else "Close"
    stock = pd.to_numeric(ensure_datetime_index(history)[stock_column], errors="coerce").dropna()
    bench = (
        pd.to_numeric(ensure_datetime_index(benchmark)[benchmark_column], errors="coerce").dropna()
        if benchmark is not None and not benchmark.empty and benchmark_column in benchmark.columns
        else pd.Series(dtype=float)
    )
    entry_stock_date, entry_stock = _nearest_series_value(stock, entry_date, direction="backward", tolerance_days=10)
    _, entry_bench = _nearest_series_value(bench, entry_date, direction="backward", tolerance_days=10)
    if entry_stock is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for months in (3, 6, 12, 24):
        target = pd.Timestamp(entry_stock_date) + pd.DateOffset(months=months)
        _, exit_stock = _nearest_series_value(stock, target, direction="forward", tolerance_days=15)
        _, exit_bench = _nearest_series_value(bench, target, direction="forward", tolerance_days=15)
        stock_return = (exit_stock / entry_stock - 1) * 100 if exit_stock not in (None, 0) else None
        bench_return = (
            (exit_bench / entry_bench - 1) * 100
            if exit_bench not in (None, 0) and entry_bench not in (None, 0)
            else None
        )
        excess_return = stock_return - bench_return if stock_return is not None and bench_return is not None else None
        rows.append({
            "Horizont": f"{months} Monate",
            "Aktienrendite": stock_return,
            "Benchmark": bench_return,
            "Überrendite": excess_return,
            "Positiv": stock_return is not None and stock_return > 0,
            "Benchmark geschlagen": excess_return is not None and excess_return > 0,
            "Daten verfügbar": exit_stock is not None,
        })
    return pd.DataFrame(rows)


def historical_signal_candidates(
    ticker: str,
    history: pd.DataFrame,
    dividends: pd.DataFrame,
    financials: pd.DataFrame,
    benchmark: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    info_lag_days: int,
) -> pd.DataFrame:
    """Berechnet alle monatlichen Deep-Value-Momentaufnahmen einmalig.

    Die spätere Schwellen- und Cooldown-Auswahl wird separat angewendet. Dadurch
    können mehrere Score-Schwellen verglichen werden, ohne die Historie mehrfach
    vollständig zu berechnen.
    """
    if history is None or history.empty:
        return pd.DataFrame()

    dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="ME")
    rows: list[dict[str, Any]] = []
    for date_value in dates:
        snapshot = historical_bat_snapshot(
            ticker=ticker,
            as_of=date_value,
            history=history,
            dividends=dividends,
            financials=financials,
            info_lag_days=info_lag_days,
            negative_news_shock=False,
            special_effect_suspected=False,
        )
        if snapshot.get("error"):
            continue

        signal_date = pd.Timestamp(snapshot["price_date"])
        forward = forward_performance_table(history, benchmark, signal_date, use_adjusted=True)
        horizon_values: dict[str, Any] = {}
        for months in (3, 6, 12, 24):
            horizon = forward[forward["Horizont"] == f"{months} Monate"] if not forward.empty else pd.DataFrame()
            horizon_values[f"Rendite {months}M"] = safe_float(horizon["Aktienrendite"].iloc[0]) if not horizon.empty else None
            horizon_values[f"Benchmark {months}M"] = safe_float(horizon["Benchmark"].iloc[0]) if not horizon.empty else None
            horizon_values[f"Überrendite {months}M"] = safe_float(horizon["Überrendite"].iloc[0]) if not horizon.empty else None
            horizon_values[f"Benchmark geschlagen {months}M"] = bool(horizon["Benchmark geschlagen"].iloc[0]) if not horizon.empty else False

        trap = _future_value_trap_metrics(
            history=history,
            dividends=dividends,
            financials=financials,
            entry_date=signal_date,
            info_lag_days=info_lag_days,
            use_adjusted=True,
        )
        row = {
            "Signal-Datum": signal_date,
            "Deep-Value-Score": safe_float(snapshot.get("bat_pattern_score")) or 0.0,
            "Kurs": snapshot.get("price"),
            "Dividendenrendite": snapshot.get("dividend_yield"),
            "Drawdown": snapshot.get("deepest_drawdown_3_5y_pct"),
            "FCF/Dividende": snapshot.get("cashflow_dividend_coverage"),
            "Net Debt/EBITDA": snapshot.get("net_debt_ebitda"),
            "Max. Rückgang 12M": trap.get("max_decline_12m"),
            "Dividendenänderung 12M": trap.get("dividend_change_12m"),
            "Cashflow nach 12M positiv": trap.get("future_cashflow_positive"),
            "Value Trap": bool(trap.get("value_trap")),
            "Value-Trap-Grund": trap.get("value_trap_reason", ""),
            "Score-Erklärung": _component_summary(snapshot),
            **horizon_values,
        }
        row["Positiv nach 12M"] = row.get("Rendite 12M") is not None and float(row["Rendite 12M"]) > 0
        row["Benchmark geschlagen 12M"] = row.get("Überrendite 12M") is not None and float(row["Überrendite 12M"]) > 0
        rows.append(row)
    return pd.DataFrame(rows)


def filter_historical_signals(
    candidates: pd.DataFrame,
    threshold: float,
    cooldown_months: int,
) -> pd.DataFrame:
    """Wendet Score-Schwelle und Mindestabstand auf vorberechnete Kandidaten an."""
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    frame = candidates.copy()
    frame["Signal-Datum"] = pd.to_datetime(frame["Signal-Datum"], errors="coerce")
    frame["Deep-Value-Score"] = pd.to_numeric(frame["Deep-Value-Score"], errors="coerce")
    frame = frame.dropna(subset=["Signal-Datum", "Deep-Value-Score"])
    frame = frame[frame["Deep-Value-Score"] >= float(threshold)].sort_values("Signal-Datum")
    if frame.empty:
        return frame.reset_index(drop=True)

    selected_indices: list[Any] = []
    last_signal: Optional[pd.Timestamp] = None
    for idx, row in frame.iterrows():
        current = pd.Timestamp(row["Signal-Datum"])
        if last_signal is not None and current < last_signal + pd.DateOffset(months=int(cooldown_months)):
            continue
        selected_indices.append(idx)
        last_signal = current
    return frame.loc[selected_indices].reset_index(drop=True)


def historical_signal_backtest(
    ticker: str,
    history: pd.DataFrame,
    dividends: pd.DataFrame,
    financials: pd.DataFrame,
    benchmark: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    threshold: float,
    info_lag_days: int,
    cooldown_months: int,
) -> pd.DataFrame:
    """Kompatibilitäts-Wrapper für einen einzelnen Schwellenwert."""
    candidates = historical_signal_candidates(
        ticker=ticker,
        history=history,
        dividends=dividends,
        financials=financials,
        benchmark=benchmark,
        start_date=start_date,
        end_date=end_date,
        info_lag_days=info_lag_days,
    )
    return filter_historical_signals(candidates, threshold, cooldown_months)


def backtest_threshold_summary(results: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in sorted(results):
        frame = results[threshold]
        valid = frame.dropna(subset=["Rendite 12M"]) if isinstance(frame, pd.DataFrame) and not frame.empty else pd.DataFrame()
        rows.append({
            "Schwelle": float(threshold),
            "Signale": len(frame) if isinstance(frame, pd.DataFrame) else 0,
            "Mit 12M-Daten": len(valid),
            "Positive Treffer 12M": float(valid["Positiv nach 12M"].mean() * 100) if not valid.empty else None,
            "Benchmark geschlagen 12M": float(valid["Benchmark geschlagen 12M"].mean() * 100) if not valid.empty else None,
            "Value-Trap-Quote": float(valid["Value Trap"].mean() * 100) if not valid.empty else None,
            "Ø Rendite 12M": safe_float(valid["Rendite 12M"].mean()) if not valid.empty else None,
            "Ø Überrendite 12M": safe_float(valid["Überrendite 12M"].mean()) if not valid.empty else None,
        })
    return pd.DataFrame(rows)


def model_signal_equity_curve(signals: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt eine transparente, modellhafte Kurve aus sequenziellen 12M-Ergebnissen.

    Jeder Signal-Ausgang wird nacheinander auf einen Startwert von 100 angewendet.
    Das ist keine kalendergenaue Portfoliosimulation und wird im UI entsprechend
    gekennzeichnet.
    """
    if signals is None or signals.empty:
        return pd.DataFrame()
    valid = signals.dropna(subset=["Rendite 12M"]).sort_values("Signal-Datum").copy()
    if valid.empty:
        return pd.DataFrame()
    strategy = 100.0
    benchmark_value = 100.0
    rows = [{"Signal": 0, "Datum": None, "Deep-Value-Strategie": strategy, "Benchmark": benchmark_value}]
    for number, (_, row) in enumerate(valid.iterrows(), start=1):
        strategy *= 1 + float(row["Rendite 12M"]) / 100
        benchmark_return = safe_float(row.get("Benchmark 12M"))
        if benchmark_return is not None:
            benchmark_value *= 1 + benchmark_return / 100
        rows.append({
            "Signal": number,
            "Datum": pd.Timestamp(row["Signal-Datum"]),
            "Deep-Value-Strategie": strategy,
            "Benchmark": benchmark_value,
        })
    return pd.DataFrame(rows)



def deep_value_snapshot_coverage(snapshot: dict[str, Any]) -> tuple[int, int, float]:
    """Bewertet, wie viele zentrale Deep-Value-Bausteine für eine Momentaufnahme vorliegen."""
    available = [
        safe_float(snapshot.get("dividend_yield")) is not None,
        safe_float(snapshot.get("deepest_drawdown_3_5y_pct")) is not None,
        safe_float(snapshot.get("free_cashflow")) is not None
        or safe_float(snapshot.get("operating_cashflow")) is not None,
        safe_float(snapshot.get("cashflow_dividend_coverage")) is not None,
        safe_float(snapshot.get("net_debt_ebitda")) is not None,
        safe_float(snapshot.get("pe_approx")) is not None
        or safe_float(snapshot.get("diluted_eps")) is not None,
    ]
    count = sum(bool(value) for value in available)
    total = len(available)
    return count, total, round(count / total * 100.0, 1) if total else 0.0


def expectation_evidence_label(sample_size: int) -> tuple[str, str]:
    """Ordnet die statistische Aussagekraft anhand unabhängiger 12M-Fälle ein."""
    if sample_size <= 2:
        return "sehr gering", "Nur sehr wenige historische Fälle – keine belastbare Statistik."
    if sample_size <= 5:
        return "gering", "Die Stichprobe ist klein; einzelne Ausreißer prägen das Ergebnis stark."
    if sample_size <= 14:
        return "mittel", "Mehrere Vergleichsfälle sind vorhanden, die Streuung bleibt aber wichtig."
    return "höher", "Die Stichprobe ist größer, bleibt aber eine historische Orientierung und keine Prognose."


def _cooldown_sample(frame: pd.DataFrame, cooldown_months: int) -> pd.DataFrame:
    """Reduziert monatliche Kandidaten auf zeitlich getrennte Vergleichsfälle."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    sample = frame.copy()
    sample["Signal-Datum"] = pd.to_datetime(sample["Signal-Datum"], errors="coerce")
    sample = sample.dropna(subset=["Signal-Datum"]).sort_values("Signal-Datum")
    selected: list[Any] = []
    last_date: Optional[pd.Timestamp] = None
    for idx, row in sample.iterrows():
        current = pd.Timestamp(row["Signal-Datum"])
        if last_date is not None and current < last_date + pd.DateOffset(months=int(cooldown_months)):
            continue
        selected.append(idx)
        last_date = current
    return sample.loc[selected].reset_index(drop=True)


def historical_expectation_for_current_score(
    candidates: pd.DataFrame,
    current_score: float,
    score_band: float = 10.0,
    cooldown_months: int = 12,
) -> dict[str, Any]:
    """Leitet historische 12M-Bandbreiten für ein heutiges Deep-Value-Setup ab.

    Zuerst werden Fälle innerhalb eines Score-Bands um den heutigen Score gesucht.
    Bei weniger als drei Fällen wird die Auswahl konservativ erweitert. Alle Fälle
    werden zeitlich ausgedünnt, damit dieselbe lange Krise nicht monatlich mehrfach
    als unabhängige Beobachtung zählt.
    """
    result: dict[str, Any] = {
        "sample": pd.DataFrame(),
        "valid": pd.DataFrame(),
        "expanded": False,
        "selection_text": "",
    }
    if candidates is None or candidates.empty:
        return result

    frame = candidates.copy()
    frame["Deep-Value-Score"] = pd.to_numeric(frame.get("Deep-Value-Score"), errors="coerce")
    frame["Signal-Datum"] = pd.to_datetime(frame.get("Signal-Datum"), errors="coerce")
    frame = frame.dropna(subset=["Deep-Value-Score", "Signal-Datum"])
    if frame.empty:
        return result

    lower = max(0.0, float(current_score) - float(score_band))
    upper = min(100.0, float(current_score) + float(score_band))
    sample = frame[frame["Deep-Value-Score"].between(lower, upper, inclusive="both")].copy()
    selection_text = f"Score {lower:.0f} bis {upper:.0f}"

    # Bei sehr kleiner Stichprobe erweitern, ohne schwache Scores beliebig einzubeziehen.
    if len(sample) < 3:
        expanded_lower = max(45.0, float(current_score) - max(15.0, float(score_band)))
        sample = frame[frame["Deep-Value-Score"] >= expanded_lower].copy()
        result["expanded"] = True
        selection_text = f"erweitert: Score ab {expanded_lower:.0f}"

    sample = _cooldown_sample(sample, max(12, int(cooldown_months)))
    valid = sample.dropna(subset=["Rendite 12M"]).copy() if not sample.empty else pd.DataFrame()
    result["sample"] = sample
    result["valid"] = valid
    result["selection_text"] = selection_text

    if valid.empty:
        return result

    returns = pd.to_numeric(valid["Rendite 12M"], errors="coerce").dropna()
    excess = pd.to_numeric(valid.get("Überrendite 12M"), errors="coerce").dropna()
    max_declines = pd.to_numeric(valid.get("Max. Rückgang 12M"), errors="coerce").dropna()
    trap_values = valid.get("Value Trap", pd.Series(False, index=valid.index)).fillna(False).astype(bool)
    benchmark_hits = valid.get("Benchmark geschlagen 12M", pd.Series(False, index=valid.index)).fillna(False).astype(bool)

    result.update({
        "count": len(valid),
        "positive_rate": float((returns > 0).mean() * 100) if not returns.empty else None,
        "benchmark_rate": float(benchmark_hits.mean() * 100) if len(benchmark_hits) else None,
        "trap_rate": float(trap_values.mean() * 100) if len(trap_values) else None,
        "average_return": safe_float(returns.mean()) if not returns.empty else None,
        "median_return": safe_float(returns.median()) if not returns.empty else None,
        "average_excess": safe_float(excess.mean()) if not excess.empty else None,
        "weak_scenario": safe_float(returns.quantile(0.20)) if not returns.empty else None,
        "middle_scenario": safe_float(returns.quantile(0.50)) if not returns.empty else None,
        "strong_scenario": safe_float(returns.quantile(0.80)) if not returns.empty else None,
        "worst_return": safe_float(returns.min()) if not returns.empty else None,
        "best_return": safe_float(returns.max()) if not returns.empty else None,
        "typical_drawdown": safe_float(max_declines.median()) if not max_declines.empty else None,
    })
    return result


def classify_historical_expectation(expectation: dict[str, Any]) -> tuple[str, str]:
    """Erzeugt eine vorsichtige, rein historische Einordnung der Vergleichsfälle."""
    count = int(expectation.get("count") or 0)
    if count < 3:
        return "nicht belastbar", "Es gibt zu wenige unabhängige Vergleichsfälle."

    median_return = safe_float(expectation.get("median_return"))
    positive_rate = safe_float(expectation.get("positive_rate"))
    benchmark_rate = safe_float(expectation.get("benchmark_rate"))
    trap_rate = safe_float(expectation.get("trap_rate"))

    if (
        median_return is not None and median_return > 5
        and benchmark_rate is not None and benchmark_rate >= 55
        and (trap_rate is None or trap_rate <= 25)
    ):
        return "historisch positiv", "Die Mehrzahl ähnlicher Fälle war positiv und schlug häufig die Benchmark."
    if (
        median_return is not None and median_return > 0
        and positive_rate is not None and positive_rate >= 55
        and (trap_rate is None or trap_rate < 40)
    ):
        return "vorsichtig positiv", "Ähnliche Fälle waren überwiegend positiv, aber nicht eindeutig überlegen."
    if (
        median_return is not None and median_return < 0
        and ((benchmark_rate is not None and benchmark_rate < 40) or (trap_rate is not None and trap_rate >= 40))
    ):
        return "historisch schwach", "Ähnliche Fälle zeigten häufig Verluste, Unterrendite oder Value Traps."
    return "gemischt", "Die historischen Ergebnisse sind uneinheitlich; Chancen und Risiken liegen eng beieinander."


def current_setup_strengths_and_risks(
    snapshot: dict[str, Any],
    expectation: dict[str, Any],
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    risks: list[str] = []

    dividend_yield = safe_float(snapshot.get("dividend_yield"))
    drawdown = safe_float(snapshot.get("deepest_drawdown_3_5y_pct"))
    coverage = safe_float(snapshot.get("cashflow_dividend_coverage"))
    leverage = safe_float(snapshot.get("net_debt_ebitda"))
    cashflow_positive = bool(snapshot.get("cashflow_positive"))
    pe_approx = safe_float(snapshot.get("pe_approx"))

    if dividend_yield is not None and dividend_yield >= 5:
        strengths.append(f"Hohe historische Dividendenrendite von {format_percent(dividend_yield, 2)}.")
    elif dividend_yield is None:
        risks.append("Dividendenrendite konnte nicht zuverlässig bestimmt werden.")

    if drawdown is not None and drawdown <= -25:
        strengths.append(f"Deutlicher Abstand zum 3J-/5J-Hoch ({format_percent(drawdown, 1, signed=True)}).")
    elif drawdown is not None and drawdown > -15:
        risks.append("Kein ausgeprägter langfristiger Drawdown – das Setup ist weniger typisch für Deep Value.")

    if cashflow_positive:
        strengths.append("Der zuletzt konservativ verfügbare operative beziehungsweise freie Cashflow ist positiv.")
    else:
        risks.append("Cashflow ist negativ oder nicht ausreichend belegt.")

    if coverage is not None and coverage >= 1.2:
        strengths.append(f"Free Cashflow deckt die Ausschüttung mit rund {format_number(coverage, 2)}x.")
    elif coverage is not None and coverage < 1.0:
        risks.append(f"Free-Cashflow-Deckung der Ausschüttung ist mit {format_number(coverage, 2)}x schwach.")
    elif coverage is None:
        risks.append("FCF-Dividendendeckung ist nicht verfügbar.")

    if leverage is not None and leverage < 3.5:
        strengths.append(f"Net Debt/EBITDA liegt mit {format_number(leverage, 2)}x unter der Deep-Value-Warngrenze.")
    elif leverage is not None and leverage >= 5:
        risks.append(f"Net Debt/EBITDA ist mit {format_number(leverage, 2)}x hoch.")
    elif leverage is None:
        risks.append("Verschuldungskennzahl Net Debt/EBITDA fehlt.")

    if pe_approx is not None and 0 < pe_approx <= 12:
        strengths.append(f"Historisch angenähertes KGV ist mit etwa {format_number(pe_approx, 1)} niedrig.")
    if snapshot.get("reported_eps_negative") and cashflow_positive:
        risks.append("Berichtetes EPS ist negativ; ein Sondereffekt muss manuell geprüft werden.")

    count = int(expectation.get("count") or 0)
    trap_rate = safe_float(expectation.get("trap_rate"))
    benchmark_rate = safe_float(expectation.get("benchmark_rate"))
    if count < 6:
        risks.append(f"Nur {count} unabhängige historische 12M-Vergleichsfälle.")
    if trap_rate is not None and trap_rate >= 30:
        risks.append(f"Historische Value-Trap-Quote liegt bei {format_percent(trap_rate, 1)}.")
    if benchmark_rate is not None and benchmark_rate >= 55:
        strengths.append(f"Ähnliche Fälle schlugen in {format_percent(benchmark_rate, 1)} der Fälle die Benchmark.")

    return strengths[:5], risks[:5]

def persist_backtest_run(
    ticker: str,
    company_name: str,
    benchmark_label: str,
    benchmark_ticker: str,
    start_date: Any,
    end_date: Any,
    threshold: float,
    cooldown_months: int,
    result: pd.DataFrame,
) -> tuple[bool, str]:
    if result is None or result.empty:
        return False, "Es gibt keine Backtest-Ergebnisse zum Speichern."
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]+", "_", clean_ticker(ticker))
    detail_path = BACKTEST_DIR / f"{safe_ticker}_threshold_{int(threshold)}_{stamp}.csv"
    success, message = safe_write_csv(result, detail_path)

    valid = result.dropna(subset=["Rendite 12M"])
    registry_row = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "ticker_yahoo": clean_ticker(ticker),
        "company_name": company_name,
        "benchmark_label": benchmark_label,
        "benchmark_ticker": benchmark_ticker,
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "end_date": pd.Timestamp(end_date).date().isoformat(),
        "threshold": float(threshold),
        "cooldown_months": int(cooldown_months),
        "signals": len(result),
        "signals_with_12m": len(valid),
        "positive_hit_rate_12m": float(valid["Positiv nach 12M"].mean() * 100) if not valid.empty else None,
        "benchmark_hit_rate_12m": float(valid["Benchmark geschlagen 12M"].mean() * 100) if not valid.empty else None,
        "value_trap_rate": float(valid["Value Trap"].mean() * 100) if not valid.empty else None,
        "average_return_12m": safe_float(valid["Rendite 12M"].mean()) if not valid.empty else None,
        "average_excess_return_12m": safe_float(valid["Überrendite 12M"].mean()) if not valid.empty else None,
        "detail_file": detail_path.name,
    }
    registry = pd.DataFrame(columns=list(registry_row))
    if BACKTEST_RUNS_PATH.exists():
        try:
            registry = pd.read_csv(BACKTEST_RUNS_PATH)
        except Exception:
            registry = pd.DataFrame(columns=list(registry_row))
    updated = pd.concat([registry, pd.DataFrame([registry_row])], ignore_index=True)
    registry_success, registry_message = safe_write_csv(updated, BACKTEST_RUNS_PATH)
    if success and registry_success:
        return True, f"Backtest wurde unter data/backtests/{detail_path.name} gespeichert."
    return False, message or registry_message


RESEARCH_CASE_COLUMNS = [
    "case_id", "saved_at", "ticker_yahoo", "company_name", "sector", "as_of",
    "price", "currency", "deep_value_score", "bat_pattern_score", "trigger",
    "return_3m", "return_6m", "return_12m", "return_24m",
    "benchmark_12m", "excess_return_12m", "label", "notes",
    "dividend_yield", "drawdown_3_5y", "fcf_dividend_coverage",
    "net_debt_ebitda", "pe_approx", "volatility_1y", "cashflow_positive",
    "reported_eps_negative", "special_effect_suspected", "negative_news_shock",
    "quality_score", "value_score", "growth_score", "momentum_score", "safety_score",
]


def load_research_cases() -> pd.DataFrame:
    """Lädt und migriert das persönliche Lernjournal ohne alte Dateien zu zerstören."""
    frame = pd.DataFrame(columns=RESEARCH_CASE_COLUMNS)
    if RESEARCH_CASES_PATH.exists():
        try:
            frame = pd.read_csv(RESEARCH_CASES_PATH)
        except Exception:
            frame = pd.DataFrame(columns=RESEARCH_CASE_COLUMNS)
    frame = deduplicate_dataframe_columns(frame)
    for column in RESEARCH_CASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    # Alte Versionen speicherten nur bat_pattern_score. Beide Namen bleiben zur
    # Rückwärtskompatibilität erhalten, der neue Name wird bevorzugt.
    deep = pd.to_numeric(frame["deep_value_score"], errors="coerce")
    legacy = pd.to_numeric(frame["bat_pattern_score"], errors="coerce")
    frame["deep_value_score"] = deep.fillna(legacy)
    frame["bat_pattern_score"] = legacy.fillna(deep)

    if frame["case_id"].isna().any() or frame["case_id"].astype(str).str.strip().eq("").any():
        for idx in frame.index:
            current = str(frame.at[idx, "case_id"] or "").strip()
            if not current or current.lower() == "nan":
                ticker = clean_ticker(frame.at[idx, "ticker_yahoo"])
                date_part = str(frame.at[idx, "as_of"] or "unknown").replace("-", "")
                frame.at[idx, "case_id"] = f"{ticker or 'CASE'}_{date_part}_{idx + 1}"
    return frame[RESEARCH_CASE_COLUMNS].copy()


def _save_research_case(record: dict[str, Any]) -> tuple[bool, str]:
    existing = load_research_cases()
    case_id = str(record.get("case_id") or "").strip()
    if not case_id:
        ticker = clean_ticker(record.get("ticker_yahoo")) or "CASE"
        date_part = str(record.get("as_of") or datetime.now().date().isoformat()).replace("-", "")
        case_id = f"{ticker}_{date_part}_{int(time.time())}"
    payload = {column: record.get(column) for column in RESEARCH_CASE_COLUMNS}
    payload["case_id"] = case_id
    payload["deep_value_score"] = record.get("deep_value_score", record.get("bat_pattern_score"))
    payload["bat_pattern_score"] = record.get("bat_pattern_score", record.get("deep_value_score"))
    updated = pd.concat([existing, pd.DataFrame([payload])], ignore_index=True)
    return safe_write_csv(updated[RESEARCH_CASE_COLUMNS], RESEARCH_CASES_PATH)



# -----------------------------------------------------------------------------
# Universumsweiter historischer Mustervergleich
# -----------------------------------------------------------------------------


def _similarity_component(current_value: Any, historical_value: Any, scale: float) -> Optional[float]:
    """Gibt eine Ähnlichkeit von 0 bis 1 für zwei numerische Merkmale zurück."""
    current = safe_float(current_value)
    historical = safe_float(historical_value)
    if current is None or historical is None or scale <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - abs(current - historical) / float(scale)))


def deep_value_pattern_similarity(
    current_snapshot: dict[str, Any],
    historical_row: pd.Series,
    current_sector: Optional[str] = None,
    historical_sector: Optional[str] = None,
) -> dict[str, Any]:
    """Vergleicht ein heutiges Deep-Value-Setup mit einem historischen Fall.

    Die Ähnlichkeit ist bewusst transparent und regelbasiert. Sie ersetzt kein
    statistisches Modell, sondern gewichtet Score, Rendite, Drawdown,
    Cashflow-Deckung und Verschuldung. Ein gleicher Sektor liefert einen kleinen
    Bonus, weil Kennzahlen innerhalb eines Geschäftsmodells besser vergleichbar
    sind.
    """
    definitions = [
        ("Deep-Value-Score", current_snapshot.get("bat_pattern_score"), historical_row.get("Deep-Value-Score"), 25.0, 0.30),
        ("Dividendenrendite", current_snapshot.get("dividend_yield"), historical_row.get("Dividendenrendite"), 5.0, 0.20),
        ("Drawdown", current_snapshot.get("deepest_drawdown_3_5y_pct"), historical_row.get("Drawdown"), 30.0, 0.20),
        ("FCF/Dividende", current_snapshot.get("cashflow_dividend_coverage"), historical_row.get("FCF/Dividende"), 1.5, 0.15),
        ("Net Debt/EBITDA", current_snapshot.get("net_debt_ebitda"), historical_row.get("Net Debt/EBITDA"), 2.5, 0.10),
    ]
    weighted_sum = 0.0
    available_weight = 0.0
    details: list[str] = []
    for label, current_value, historical_value, scale, weight in definitions:
        similarity = _similarity_component(current_value, historical_value, scale)
        if similarity is None:
            details.append(f"{label}: keine vollständigen Vergleichsdaten")
            continue
        weighted_sum += similarity * weight
        available_weight += weight
        details.append(f"{label}: {similarity * 100:.0f} % ähnlich")

    if available_weight <= 0:
        return {"similarity": None, "coverage": 0.0, "explanation": "Keine vergleichbaren Merkmale verfügbar."}

    score = weighted_sum / available_weight * 100.0
    same_sector = bool(
        current_sector and historical_sector
        and str(current_sector).strip().lower() == str(historical_sector).strip().lower()
    )
    if same_sector:
        score = min(100.0, score + 5.0)
        details.append("Sektorbonus: +5 Punkte")

    return {
        "similarity": round(score, 1),
        "coverage": round(available_weight / 0.95 * 100.0, 1),
        "same_sector": same_sector,
        "explanation": " | ".join(details),
    }


def _sample_dates(start_date: Any, end_date: Any, sampling_months: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date).tz_localize(None).normalize()
    end = pd.Timestamp(end_date).tz_localize(None).normalize()
    if end < start:
        return pd.DatetimeIndex([])
    monthly = pd.date_range(start, end, freq="ME")
    step = max(1, int(sampling_months))
    return monthly[::step]


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def historical_pattern_candidates_cached(
    ticker: str,
    company_name: str,
    sector: str,
    benchmark_ticker: str,
    start_date_iso: str,
    end_date_iso: str,
    info_lag_days: int,
    sampling_months: int,
) -> pd.DataFrame:
    """Erstellt zeitlich gesampelte historische Deep-Value-Fälle je Unternehmen."""
    symbol = clean_ticker(ticker)
    if not symbol:
        return pd.DataFrame()

    history = fetch_long_history(symbol)
    if history.empty or "Close" not in history.columns:
        return pd.DataFrame()
    dividends = fetch_dividends(symbol)
    financials = fetch_historical_financials(symbol)
    benchmark = fetch_benchmark_history(benchmark_ticker, period="max")

    rows: list[dict[str, Any]] = []
    for date_value in _sample_dates(start_date_iso, end_date_iso, sampling_months):
        snapshot = historical_bat_snapshot(
            ticker=symbol,
            as_of=date_value,
            history=history,
            dividends=dividends,
            financials=financials,
            info_lag_days=int(info_lag_days),
            negative_news_shock=False,
            special_effect_suspected=False,
        )
        if snapshot.get("error"):
            continue

        signal_date = pd.Timestamp(snapshot["price_date"])
        forward = forward_performance_table(history, benchmark, signal_date, use_adjusted=True)
        horizon = forward[forward["Horizont"] == "12 Monate"] if not forward.empty else pd.DataFrame()
        stock_return = safe_float(horizon["Aktienrendite"].iloc[0]) if not horizon.empty else None
        benchmark_return = safe_float(horizon["Benchmark"].iloc[0]) if not horizon.empty else None
        excess_return = safe_float(horizon["Überrendite"].iloc[0]) if not horizon.empty else None

        trap = _future_value_trap_metrics(
            history=history,
            dividends=dividends,
            financials=financials,
            entry_date=signal_date,
            info_lag_days=int(info_lag_days),
            use_adjusted=True,
        )
        rows.append({
            "Ticker": symbol,
            "Unternehmen": company_name or symbol,
            "Sektor": sector or "Unbekannt",
            "Signal-Datum": signal_date,
            "Deep-Value-Score": safe_float(snapshot.get("bat_pattern_score")) or 0.0,
            "Kurs": snapshot.get("price"),
            "Dividendenrendite": snapshot.get("dividend_yield"),
            "Drawdown": snapshot.get("deepest_drawdown_3_5y_pct"),
            "FCF/Dividende": snapshot.get("cashflow_dividend_coverage"),
            "Net Debt/EBITDA": snapshot.get("net_debt_ebitda"),
            "Rendite 12M": stock_return,
            "Benchmark 12M": benchmark_return,
            "Überrendite 12M": excess_return,
            "Benchmark geschlagen 12M": excess_return is not None and excess_return > 0,
            "Max. Rückgang 12M": trap.get("max_decline_12m"),
            "Dividendenänderung 12M": trap.get("dividend_change_12m"),
            "Value Trap": bool(trap.get("value_trap")),
            "Value-Trap-Grund": trap.get("value_trap_reason", ""),
            "Score-Erklärung": _component_summary(snapshot),
        })
    return pd.DataFrame(rows)


def _independent_pattern_cases(frame: pd.DataFrame, cooldown_months: int) -> pd.DataFrame:
    """Wählt je Ticker die ähnlichsten zeitlich unabhängigen Fälle aus."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    selected_rows: list[pd.Series] = []
    for _, group in frame.groupby("Ticker", dropna=False):
        group = group.copy()
        group["Signal-Datum"] = pd.to_datetime(group["Signal-Datum"], errors="coerce")
        group = group.dropna(subset=["Signal-Datum", "Ähnlichkeit"]).sort_values("Ähnlichkeit", ascending=False)
        chosen_dates: list[pd.Timestamp] = []
        for _, row in group.iterrows():
            current_date = pd.Timestamp(row["Signal-Datum"])
            if any(abs((current_date - previous).days) < int(cooldown_months * 30.44) for previous in chosen_dates):
                continue
            selected_rows.append(row)
            chosen_dates.append(current_date)
    if not selected_rows:
        return pd.DataFrame(columns=frame.columns)
    return pd.DataFrame(selected_rows).sort_values(["Ähnlichkeit", "Signal-Datum"], ascending=[False, False]).reset_index(drop=True)


def summarise_universe_pattern_cases(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"count": 0}
    if frame is None or frame.empty:
        return result
    valid = frame.dropna(subset=["Rendite 12M"]).copy()
    if valid.empty:
        return result
    returns = pd.to_numeric(valid["Rendite 12M"], errors="coerce").dropna()
    excess = pd.to_numeric(valid.get("Überrendite 12M"), errors="coerce").dropna()
    drawdowns = pd.to_numeric(valid.get("Max. Rückgang 12M"), errors="coerce").dropna()
    trap = valid.get("Value Trap", pd.Series(False, index=valid.index)).fillna(False).astype(bool)
    benchmark_hits = valid.get("Benchmark geschlagen 12M", pd.Series(False, index=valid.index)).fillna(False).astype(bool)
    result.update({
        "count": len(valid),
        "positive_rate": float((returns > 0).mean() * 100) if not returns.empty else None,
        "benchmark_rate": float(benchmark_hits.mean() * 100) if len(benchmark_hits) else None,
        "trap_rate": float(trap.mean() * 100) if len(trap) else None,
        "average_return": safe_float(returns.mean()) if not returns.empty else None,
        "median_return": safe_float(returns.median()) if not returns.empty else None,
        "average_excess": safe_float(excess.mean()) if not excess.empty else None,
        "weak_scenario": safe_float(returns.quantile(0.20)) if not returns.empty else None,
        "middle_scenario": safe_float(returns.quantile(0.50)) if not returns.empty else None,
        "strong_scenario": safe_float(returns.quantile(0.80)) if not returns.empty else None,
        "typical_drawdown": safe_float(drawdowns.median()) if not drawdowns.empty else None,
        "worst_return": safe_float(returns.min()) if not returns.empty else None,
        "best_return": safe_float(returns.max()) if not returns.empty else None,
        "valid": valid,
    })
    return result


def render_universe_pattern_comparison(df: pd.DataFrame, index_name: str) -> None:
    st.subheader("Universumsweiter historischer Mustervergleich")
    st.caption(
        "Vergleicht das heutige Deep-Value-Setup einer Aktie mit historischen Situationen "
        "mehrerer Unternehmen des aktuell geladenen Index. Dadurch entsteht eine breitere "
        "Stichprobe als bei der reinen Einzelaktien-Historie."
    )

    with st.expander("Methodik und wichtige Grenzen", expanded=False):
        st.markdown(
            """
            **Verglichen werden:** Deep-Value-Score, Dividendenrendite, langfristiger Drawdown,
            Free-Cashflow-Deckung der Dividende und Net Debt/EBITDA. Ein gleicher Sektor erhält
            einen kleinen Bonus.

            **Ergebnis:** historische 12-Monatsrendite, Überrendite zur Benchmark,
            zwischenzeitlicher Rückgang und Value-Trap-Kennzeichnung.

            **Wichtige Grenze:** Die Auswahl nutzt die *heutigen* Indexbestandteile. Frühere
            ausgeschiedene oder insolvente Unternehmen fehlen. Dadurch besteht Survivorship Bias.
            Yahoo-Jahresabschlüsse sind zudem keine garantierte Point-in-Time-Datenbank. Das Modul
            ist ein Research-Werkzeug und keine Renditeprognose.
            """
        )

    if df is None or df.empty:
        st.info("Bitte zuerst links Marktdaten für einen Index laden.")
        return

    target_ticker = company_selectbox("Heutiges Zielunternehmen", df, key="universe_pattern_target")
    target_row = df[df["ticker_yahoo"].astype(str) == str(target_ticker)].iloc[0]
    target_name = str(target_row.get("name") or target_ticker)
    target_sector = str(target_row.get("sector") or "Unbekannt")
    benchmark_ticker = benchmark_for_index(index_name)

    with st.spinner(f"Berechne heutiges Deep-Value-Setup für {target_name} ({target_ticker}) …"):
        target_history = fetch_long_history(target_ticker)
        target_dividends = fetch_dividends(target_ticker)
        target_financials = fetch_historical_financials(target_ticker)
    if target_history.empty:
        st.error("Für das Zielunternehmen konnten keine langen Kursdaten geladen werden.")
        return
    latest_date = target_history.index.max()
    current_snapshot = historical_bat_snapshot(
        ticker=target_ticker,
        as_of=latest_date,
        history=target_history,
        dividends=target_dividends,
        financials=target_financials,
        info_lag_days=120,
        negative_news_shock=False,
        special_effect_suspected=False,
    )
    if current_snapshot.get("error"):
        st.warning(str(current_snapshot.get("error")))
        return

    target_cols = st.columns(6)
    target_cols[0].metric("Zielaktie", target_name)
    target_cols[1].metric("Deep-Value-Score heute", format_number(current_snapshot.get("bat_pattern_score"), 1))
    target_cols[2].metric("Dividendenrendite", format_percent(current_snapshot.get("dividend_yield"), 2))
    target_cols[3].metric("Drawdown 3J/5J", format_percent(current_snapshot.get("deepest_drawdown_3_5y_pct"), 1, signed=True))
    target_cols[4].metric("FCF/Dividende", f"{format_number(current_snapshot.get('cashflow_dividend_coverage'), 2)}x")
    target_cols[5].metric("Net Debt/EBITDA", f"{format_number(current_snapshot.get('net_debt_ebitda'), 2)}x")
    st.caption(f"Sektor: {target_sector} · Benchmark: {benchmark_ticker} · Stichtag: {pd.Timestamp(latest_date).strftime('%d.%m.%Y')}")

    st.markdown("#### Vergleichsuniversum einstellen")
    settings_a, settings_b, settings_c = st.columns(3)
    sector_mode = settings_a.selectbox(
        "Sektorvergleich",
        ["Alle Sektoren · gleicher Sektor bevorzugt", "Nur gleicher Sektor", "Alle Sektoren ohne Bonus"],
        key="universe_pattern_sector_mode",
    )
    sampling_label = settings_b.selectbox(
        "Historische Abtastung",
        ["Quartalsweise", "Halbjährlich"],
        key="universe_pattern_sampling",
        help="Quartalsweise liefert mehr Fälle, braucht aber länger.",
    )
    sampling_months = 3 if sampling_label == "Quartalsweise" else 6
    max_available = max(1, len(df.drop_duplicates("ticker_yahoo")))
    max_companies = settings_c.slider(
        "Max. Unternehmen",
        min_value=1,
        max_value=min(100, max_available),
        value=min(20, max_available),
        step=1,
        key="universe_pattern_max_companies",
        help="Für den ersten Test 10 bis 20 Unternehmen verwenden. DAX 40 kann anschließend vollständig geprüft werden.",
    )

    settings_d, settings_e, settings_f = st.columns(3)
    similarity_min = settings_d.slider("Mindestähnlichkeit", 40, 95, 65, 5, key="universe_pattern_similarity_min")
    min_coverage = settings_e.slider("Min. Merkmalsabdeckung", 20, 100, 60, 10, key="universe_pattern_min_coverage")
    cooldown_months = settings_f.slider("Mindestabstand je Aktie (Monate)", 6, 36, 12, 1, key="universe_pattern_cooldown")

    history_min = datetime(2005, 1, 1).date()
    default_start = datetime(2014, 1, 1).date()
    latest_complete = (pd.Timestamp.today().normalize() - pd.DateOffset(months=12)).date()
    date_a, date_b, date_c = st.columns(3)
    start_date = date_a.date_input("Historischer Beginn", value=default_start, min_value=history_min, max_value=latest_complete, key="universe_pattern_start")
    end_date = date_b.date_input("Historisches Ende", value=latest_complete, min_value=history_min, max_value=latest_complete, key="universe_pattern_end")
    max_cases = date_c.slider("Max. Vergleichsfälle", 10, 200, 60, 10, key="universe_pattern_max_cases")

    universe = df[["ticker_yahoo", "name", "sector", "deep_value_score"]].copy() if "deep_value_score" in df.columns else df[["ticker_yahoo", "name", "sector"]].copy()
    universe = universe.dropna(subset=["ticker_yahoo"]).drop_duplicates("ticker_yahoo")
    if sector_mode == "Nur gleicher Sektor":
        universe = universe[universe["sector"].astype(str).str.lower() == target_sector.lower()]
    if "deep_value_score" in universe.columns:
        universe["deep_value_score"] = pd.to_numeric(universe["deep_value_score"], errors="coerce")
        universe = universe.sort_values(["deep_value_score", "name"], ascending=[False, True], na_position="last")
    else:
        universe = universe.sort_values("name")
    universe = universe.head(int(max_companies)).reset_index(drop=True)

    st.caption(
        f"Geplant: {len(universe)} Unternehmen · {sampling_label.lower()} · "
        f"{pd.Timestamp(start_date).strftime('%d.%m.%Y')} bis {pd.Timestamp(end_date).strftime('%d.%m.%Y')}. "
        "Bereits gecachte Unternehmen werden deutlich schneller verarbeitet."
    )

    if st.button("Universumsvergleich starten", type="primary", key="run_universe_pattern_comparison"):
        if pd.Timestamp(end_date) <= pd.Timestamp(start_date):
            st.error("Das Enddatum muss nach dem Startdatum liegen.")
        else:
            progress = st.progress(0.0, text="Bereite Vergleich vor …")
            frames: list[pd.DataFrame] = []
            errors: list[str] = []
            total = max(1, len(universe))
            for position, row in universe.iterrows():
                symbol = clean_ticker(row.get("ticker_yahoo"))
                name = str(row.get("name") or symbol)
                sector = str(row.get("sector") or "Unbekannt")
                progress.progress(position / total, text=f"{name} ({symbol}) · historische Fälle …")
                try:
                    candidates = historical_pattern_candidates_cached(
                        ticker=symbol,
                        company_name=name,
                        sector=sector,
                        benchmark_ticker=benchmark_ticker,
                        start_date_iso=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                        end_date_iso=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                        info_lag_days=120,
                        sampling_months=sampling_months,
                    )
                    if candidates is not None and not candidates.empty:
                        frames.append(candidates)
                    else:
                        errors.append(f"{symbol}: keine ausreichenden historischen Fälle")
                except Exception as exc:
                    errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            progress.progress(1.0, text="Berechne Ähnlichkeiten …")

            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not combined.empty:
                similarity_rows: list[dict[str, Any]] = []
                for _, candidate in combined.iterrows():
                    candidate_sector = str(candidate.get("Sektor") or "Unbekannt")
                    comparison_sector = None if sector_mode == "Alle Sektoren ohne Bonus" else target_sector
                    similarity = deep_value_pattern_similarity(
                        current_snapshot,
                        candidate,
                        current_sector=comparison_sector,
                        historical_sector=candidate_sector,
                    )
                    item = candidate.to_dict()
                    item["Ähnlichkeit"] = similarity.get("similarity")
                    item["Merkmalsabdeckung"] = similarity.get("coverage")
                    item["Gleicher Sektor"] = bool(similarity.get("same_sector"))
                    item["Ähnlichkeits-Erklärung"] = similarity.get("explanation")
                    similarity_rows.append(item)
                combined = pd.DataFrame(similarity_rows)
                combined = combined[
                    (pd.to_numeric(combined["Ähnlichkeit"], errors="coerce") >= float(similarity_min))
                    & (pd.to_numeric(combined["Merkmalsabdeckung"], errors="coerce") >= float(min_coverage))
                ].copy()
                combined = _independent_pattern_cases(combined, int(cooldown_months))
                combined = combined.sort_values("Ähnlichkeit", ascending=False).head(int(max_cases)).reset_index(drop=True)

            st.session_state["universe_pattern_result"] = combined
            st.session_state["universe_pattern_errors"] = errors
            st.session_state["universe_pattern_meta"] = {
                "target_ticker": target_ticker,
                "target_name": target_name,
                "target_sector": target_sector,
                "benchmark": benchmark_ticker,
                "companies": len(universe),
                "start_date": str(start_date),
                "end_date": str(end_date),
                "sampling": sampling_label,
                "similarity_min": similarity_min,
                "coverage_min": min_coverage,
                "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
            }
            progress.empty()
            st.rerun()

    result = st.session_state.get("universe_pattern_result")
    meta = st.session_state.get("universe_pattern_meta", {})
    if not isinstance(result, pd.DataFrame):
        st.info("Starte den Vergleich, um historische Muster aus mehreren Unternehmen auszuwerten.")
        return

    if meta.get("target_ticker") != target_ticker:
        st.warning("Das angezeigte Ergebnis gehört zu einer zuvor gewählten Zielaktie. Starte den Vergleich erneut.")

    errors = st.session_state.get("universe_pattern_errors", [])
    if errors:
        with st.expander(f"Nicht vollständig verarbeitete Unternehmen ({len(errors)})", expanded=False):
            for message in errors[:100]:
                st.write(f"- {message}")

    if result.empty:
        st.warning(
            "Keine historischen Fälle erfüllen die gewählte Ähnlichkeit und Datenabdeckung. "
            "Reduziere die Mindestähnlichkeit, erweitere den Zeitraum oder erhöhe die Zahl der Unternehmen."
        )
        return

    summary = summarise_universe_pattern_cases(result)
    count = int(summary.get("count") or 0)
    evidence_label, evidence_text = expectation_evidence_label(count)
    st.markdown("#### Historische Erwartung aus dem Vergleichsuniversum")
    summary_cols = st.columns(7)
    summary_cols[0].metric("Vergleichsfälle", count)
    summary_cols[1].metric("Aussagekraft", evidence_label)
    summary_cols[2].metric("Positiv nach 12M", format_percent(summary.get("positive_rate"), 1))
    summary_cols[3].metric("Benchmark geschlagen", format_percent(summary.get("benchmark_rate"), 1))
    summary_cols[4].metric("Medianrendite", format_percent(summary.get("median_return"), 2, signed=True))
    summary_cols[5].metric("Value-Trap-Anteil", format_percent(summary.get("trap_rate"), 1))
    summary_cols[6].metric("Typischer Rückgang", format_percent(summary.get("typical_drawdown"), 1, signed=True))

    if count < 5:
        st.error(f"Nur {count} unabhängige Fälle. Die Statistik ist nicht belastbar. {evidence_text}")
    else:
        st.caption(evidence_text)

    scenario_cols = st.columns(3)
    scenario_cols[0].metric("Schwaches Szenario · 20 %", format_percent(summary.get("weak_scenario"), 2, signed=True))
    scenario_cols[1].metric("Mittleres Szenario · Median", format_percent(summary.get("middle_scenario"), 2, signed=True))
    scenario_cols[2].metric("Starkes Szenario · 80 %", format_percent(summary.get("strong_scenario"), 2, signed=True))

    valid = summary.get("valid", pd.DataFrame())
    if isinstance(valid, pd.DataFrame) and not valid.empty:
        plot_frame = valid.dropna(subset=["Ähnlichkeit", "Rendite 12M"]).copy()
        if not plot_frame.empty:
            plot_frame["Fall"] = plot_frame["Unternehmen"].astype(str) + " · " + pd.to_datetime(plot_frame["Signal-Datum"]).dt.strftime("%m/%Y")
            scatter = alt.Chart(plot_frame).mark_circle(size=90).encode(
                x=alt.X("Ähnlichkeit:Q", scale=alt.Scale(domain=[40, 100]), title="Ähnlichkeit zum heutigen Setup (%)"),
                y=alt.Y("Rendite 12M:Q", title="Rendite nach 12 Monaten (%)"),
                shape=alt.Shape("Value Trap:N", title="Value Trap"),
                tooltip=[
                    "Fall:N", "Ticker:N", "Sektor:N",
                    alt.Tooltip("Ähnlichkeit:Q", format=".1f"),
                    alt.Tooltip("Rendite 12M:Q", format="+.2f"),
                    alt.Tooltip("Überrendite 12M:Q", format="+.2f"),
                    "Value Trap:N",
                ],
            ).properties(height=360)
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart(scatter + zero, use_container_width=True)
            st.caption("Jeder Punkt ist ein historischer, zeitlich unabhängiger Fall. Weiter rechts bedeutet ähnlicher zum heutigen Setup.")

    st.markdown("#### Ähnlichste historische Fälle")
    display_columns = [
        "Unternehmen", "Ticker", "Sektor", "Signal-Datum", "Ähnlichkeit", "Merkmalsabdeckung",
        "Deep-Value-Score", "Dividendenrendite", "Drawdown", "FCF/Dividende",
        "Net Debt/EBITDA", "Rendite 12M", "Überrendite 12M", "Max. Rückgang 12M",
        "Value Trap", "Value-Trap-Grund",
    ]
    display = result[[column for column in display_columns if column in result.columns]].copy()
    if "Signal-Datum" in display.columns:
        display["Signal-Datum"] = pd.to_datetime(display["Signal-Datum"], errors="coerce").dt.strftime("%d.%m.%Y")
    st.dataframe(
        display.style.format(
            {
                "Ähnlichkeit": "{:.1f} %",
                "Merkmalsabdeckung": "{:.1f} %",
                "Deep-Value-Score": "{:.1f}",
                "Dividendenrendite": lambda value: format_percent(value, 2),
                "Drawdown": lambda value: format_percent(value, 1, signed=True),
                "FCF/Dividende": lambda value: f"{format_number(value, 2)}x",
                "Net Debt/EBITDA": lambda value: f"{format_number(value, 2)}x",
                "Rendite 12M": lambda value: format_percent(value, 2, signed=True),
                "Überrendite 12M": lambda value: format_percent(value, 2, signed=True),
                "Max. Rückgang 12M": lambda value: format_percent(value, 1, signed=True),
            },
            na_rep="–",
        ).map(
            colorize_change,
            subset=[column for column in ["Rendite 12M", "Überrendite 12M"] if column in display.columns],
        ),
        use_container_width=True,
        hide_index=True,
    )

    selected_options = result.index.tolist()
    selected_case = st.selectbox(
        "Historischen Fall genauer ansehen",
        options=selected_options,
        format_func=lambda idx: (
            f"{result.loc[idx, 'Unternehmen']} ({result.loc[idx, 'Ticker']}) · "
            f"{pd.Timestamp(result.loc[idx, 'Signal-Datum']).strftime('%m/%Y')} · "
            f"{safe_float(result.loc[idx, 'Ähnlichkeit']) or 0:.1f} % ähnlich"
        ),
        key="universe_pattern_case_detail",
    )
    case = result.loc[selected_case]
    detail_left, detail_right = st.columns(2)
    detail_left.info(f"**Ähnlichkeitslogik**\n\n{case.get('Ähnlichkeits-Erklärung', '–')}")
    detail_right.info(f"**Historischer Score**\n\n{case.get('Score-Erklärung', '–')}")

    csv_data = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Vergleichsfälle als CSV herunterladen",
        data=csv_data,
        file_name=f"mustervergleich_{clean_ticker(target_ticker).replace('.', '_')}.csv",
        mime="text/csv",
        key="download_universe_pattern_cases",
    )
    st.caption(
        f"Ergebnis erstellt: {meta.get('created_at', '–')} · {meta.get('companies', '–')} Unternehmen geprüft · "
        f"Mindestähnlichkeit {meta.get('similarity_min', '–')} % · Mindestabdeckung {meta.get('coverage_min', '–')} %."
    )


def render_bat_backtesting(df: pd.DataFrame, index_name: str) -> None:
    st.subheader("Historisches Deep-Value-Backtesting")
    st.caption(
        "Prüft, ob ein BAT-ähnliches Deep-Value-Muster an einem historischen Stichtag "
        "sichtbar gewesen wäre. Preise und Dividenden sind historisch; Jahresabschlüsse "
        "werden mit einem konservativen Veröffentlichungsverzug angenähert."
    )

    with st.expander("Was dieser Backtest kann – und was nicht", expanded=False):
        st.markdown(
            """
            **Enthalten:** historischer Kurs, 3J-/5J-Drawdown, damalige Ausschüttungen,
            verfügbare Jahres-Cashflows, FCF-Dividendendeckung, Verschuldung sowie spätere
            3/6/12/24-Monatsrenditen und Benchmarkvergleiche.

            **Nachträgliche Value-Trap-Prüfung:** weiterer Kursrückgang von mindestens 30 %,
            deutliche Dividendenkürzung oder später negativer Cashflow. Diese Informationen
            fließen nicht in den damaligen Score ein, sondern bewerten nur das Ergebnis.

            **Manuell in der Fallstudie:** einmalige Abschreibungen und damalige negative
            Nachrichten müssen bestätigt werden, weil RSS-Feeds kein belastbares Archiv sind.

            **Nicht enthalten:** Steuern, Gebühren, exakte Point-in-Time-Veröffentlichungsdaten,
            historische Analystenschätzungen und eine kalendergenaue Portfoliosimulation.
            """
        )

    if st.button("BAT-Beispiel Ende 2023 laden", key="load_bat_case_preset"):
        st.session_state["backtest_source_mode"] = "Eigener Yahoo-Ticker"
        st.session_state["backtest_custom_ticker"] = "BATS.L"
        st.session_state["backtest_as_of"] = datetime(2023, 12, 6).date()
        st.session_state["backtest_negative_news"] = True
        st.session_state["backtest_special_effect"] = True
        st.rerun()

    source_mode = st.radio(
        "Aktie wählen",
        ["Aus geladenem Index", "Eigener Yahoo-Ticker"],
        horizontal=True,
        key="backtest_source_mode",
    )
    if source_mode == "Aus geladenem Index":
        ticker = company_selectbox("Unternehmen", df, key="backtest_index_ticker")
    else:
        st.session_state.setdefault("backtest_custom_ticker", "BATS.L")
        ticker = clean_ticker(st.text_input("Yahoo-Ticker", key="backtest_custom_ticker"))
        st.caption("Beispiele: BATS.L (London), BTI (US-ADR), ALV.DE oder MSFT.")

    if not ticker:
        st.info("Bitte einen gültigen Yahoo-Ticker eingeben.")
        return

    company_name = ticker
    company_sector = "Unbekannt"
    if df is not None and not df.empty and {"ticker_yahoo", "name"}.issubset(df.columns):
        match = df[df["ticker_yahoo"].astype(str) == ticker]
        if not match.empty:
            company_name = str(match.iloc[0].get("name") or ticker)
            company_sector = str(match.iloc[0].get("sector") or "Unbekannt")
    if company_name == ticker or company_sector == "Unbekannt":
        try:
            ticker_info = fetch_ticker_info(ticker)
            company_name = str(ticker_info.get("longName") or ticker_info.get("shortName") or company_name)
            company_sector = str(ticker_info.get("sector") or company_sector)
        except Exception:
            company_name = company_name or ticker

    benchmark_choices = {
        f"Aktueller Index ({index_name})": benchmark_for_index(index_name),
        "MSCI World ETF (URTH)": "URTH",
        "S&P 500": "^GSPC",
        "DAX": "^GDAXI",
        "FTSE 100": "^FTSE",
    }
    benchmark_label = st.selectbox("Vergleichsbenchmark", list(benchmark_choices), key="backtest_benchmark_label")
    benchmark_ticker = benchmark_choices[benchmark_label]

    with st.spinner(f"Lade historische Research-Daten für {company_name} ({ticker}) …"):
        history = fetch_long_history(ticker)
        dividends = fetch_dividends(ticker)
        financials = fetch_historical_financials(ticker)
        benchmark = fetch_benchmark_history(benchmark_ticker, period="max")

    if history.empty or "Close" not in history.columns:
        st.error(f"Für {company_name} ({ticker}) konnten keine historischen Kursdaten geladen werden.")
        return

    min_date = history.index.min().date()
    max_date = history.index.max().date()
    suggested_date = max(min_date, min(max_date, datetime(2023, 12, 6).date()))

    stored_as_of = safe_session_date(
        st.session_state.get("backtest_as_of"),
        fallback=suggested_date,
        min_date=min_date,
        max_date=max_date,
    )
    st.session_state["backtest_as_of"] = stored_as_of
    st.session_state.setdefault("backtest_lag_days", 120)
    st.session_state.setdefault("backtest_use_adjusted", True)
    st.session_state.setdefault("backtest_negative_news", False)
    st.session_state.setdefault("backtest_special_effect", False)

    settings_left, settings_middle, settings_right = st.columns([1.2, 1, 1])
    as_of = settings_left.date_input(
        "Historischer Stichtag",
        min_value=min_date,
        max_value=max_date,
        key="backtest_as_of",
    )
    lag_days = settings_middle.slider(
        "Berichts-Verzug (Tage)", 60, 180, step=15,
        help="Verhindert, dass Jahresabschlussdaten vor ihrer wahrscheinlichen Veröffentlichung verwendet werden.",
        key="backtest_lag_days",
    )
    use_adjusted = settings_right.checkbox("Adj Close für Folgerendite", key="backtest_use_adjusted")

    qualitative_left, qualitative_right = st.columns(2)
    negative_news = qualitative_left.checkbox(
        "Damals klar negativer News-/Stimmungsschock",
        help="Zum Beispiel Gewinnwarnung, regulatorischer Schock oder extremer Pessimismus.",
        key="backtest_negative_news",
    )
    special_effect = qualitative_right.checkbox(
        "Reported EPS durch einmaligen Sondereffekt verzerrt",
        help="Zum Beispiel eine große, nicht zahlungswirksame Abschreibung bei weiterhin positivem Cashflow.",
        key="backtest_special_effect",
    )

    identity_cols = st.columns(3)
    identity_cols[0].info(f"**Geprüfte Aktie**\n\n{company_name} ({ticker})")
    identity_cols[1].info(f"**Benchmark**\n\n{benchmark_label} ({benchmark_ticker})")
    identity_cols[2].info(f"**Historischer Stichtag**\n\n{pd.Timestamp(as_of).strftime('%d.%m.%Y')}")

    snapshot = historical_bat_snapshot(
        ticker=ticker,
        as_of=as_of,
        history=history,
        dividends=dividends,
        financials=financials,
        info_lag_days=int(lag_days),
        negative_news_shock=negative_news,
        special_effect_suspected=special_effect,
    )
    if snapshot.get("error"):
        st.warning(str(snapshot["error"]))
        return

    score = safe_float(snapshot.get("bat_pattern_score")) or 0.0
    trigger = bool(snapshot.get("bat_pattern_trigger"))
    score_cols = st.columns(6)
    score_cols[0].metric("Deep-Value-Score", format_number(score, 1))
    score_cols[1].metric("Status", "⚠ prüfen" if trigger else "Beobachten" if score >= 55 else "kein Muster")
    score_cols[2].metric("Historischer Kurs", f"{format_number(snapshot.get('price'), 2)} {snapshot.get('currency', '–')}")
    score_cols[3].metric("Dividendenrendite", format_percent(snapshot.get("dividend_yield"), 2))
    score_cols[4].metric("3J-/5J-Drawdown", format_percent(snapshot.get("deepest_drawdown_3_5y_pct"), 1, signed=True))
    score_cols[5].metric("FCF/Dividende", f"{format_number(snapshot.get('cashflow_dividend_coverage'), 2)}x")

    if trigger:
        st.warning(
            "Sondersituation erkannt: hohe Rendite + starker Drawdown + positiver Cashflow. "
            "Das ist ein Research-Hinweis, kein Kaufsignal. Prüfe Geschäftsmodell, "
            "Cashflow-Normalisierung und Risiken."
        )
    elif score >= 55:
        st.info("Mehrere Deep-Value-Merkmale sind vorhanden, aber der strenge Trigger ist noch nicht vollständig erfüllt.")

    st.markdown("#### Score-Bausteine am Stichtag")
    component_table = snapshot.get("components", pd.DataFrame())
    if isinstance(component_table, pd.DataFrame) and not component_table.empty:
        st.dataframe(
            component_table.style.format({"Punkte": "{:.0f}", "Maximum": "{:.0f}"}),
            use_container_width=True,
            hide_index=True,
        )

    financial_date = snapshot.get("financial_fiscal_date")
    st.caption(
        f"Verwendeter Kurs-Tag: {pd.Timestamp(snapshot['price_date']).strftime('%d.%m.%Y')} · "
        f"letzter konservativ verfügbarer Jahresabschluss: "
        f"{pd.Timestamp(financial_date).strftime('%d.%m.%Y') if pd.notna(financial_date) else 'nicht verfügbar'} · "
        f"Berichts-Verzug: {lag_days} Tage"
    )

    forward = forward_performance_table(history, benchmark, pd.Timestamp(snapshot["price_date"]), use_adjusted=use_adjusted)
    st.markdown("#### Was geschah danach?")
    if forward.empty:
        st.info("Für die gewählten Horizonte liegen noch keine ausreichenden Folgedaten vor.")
    else:
        st.dataframe(
            forward.style.format(
                {
                    "Aktienrendite": lambda value: format_percent(value, 2, signed=True),
                    "Benchmark": lambda value: format_percent(value, 2, signed=True),
                    "Überrendite": lambda value: format_percent(value, 2, signed=True),
                },
                na_rep="–",
            ).map(colorize_change, subset=["Aktienrendite", "Überrendite"]),
            use_container_width=True,
            hide_index=True,
        )

    chart_start = pd.Timestamp(snapshot["price_date"]) - pd.DateOffset(years=2)
    chart_end = pd.Timestamp(snapshot["price_date"]) + pd.DateOffset(years=2)
    chart_column = "Adj Close" if use_adjusted and "Adj Close" in history.columns else "Close"
    chart_frame = history.loc[(history.index >= chart_start) & (history.index <= chart_end), [chart_column]].copy()
    chart_frame = chart_frame.rename(columns={chart_column: "Kurs"}).reset_index()
    chart_frame = chart_frame.rename(columns={chart_frame.columns[0]: "Datum"})
    if not chart_frame.empty:
        line = alt.Chart(chart_frame).mark_line().encode(
            x=alt.X("Datum:T", title="Datum"),
            y=alt.Y("Kurs:Q", title="Kurs", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Datum:T", format="%d.%m.%Y"), alt.Tooltip("Kurs:Q", format=".2f")],
        )
        marker = alt.Chart(pd.DataFrame({"Datum": [pd.Timestamp(snapshot["price_date"])], "Kurs": [snapshot["price"]]})).mark_point(
            size=140, filled=True, color="#f59e0b"
        ).encode(x="Datum:T", y="Kurs:Q", tooltip=[alt.Tooltip("Datum:T", format="%d.%m.%Y"), alt.Tooltip("Kurs:Q", format=".2f")])
        st.altair_chart((line + marker).properties(height=330, title="Kursentwicklung rund um den historischen Stichtag"), use_container_width=True)

    st.divider()
    st.markdown(f"### Quantitative Deep-Value-Prüfung für {company_name} ({ticker})")
    st.caption(
        "Testet den Score an Monatsenden ohne manuelle News- und Sondereffekt-Punkte. "
        "Mehrere Schwellenwerte können mit identischen historischen Momentaufnahmen verglichen werden."
    )
    earliest_backtest = max(pd.Timestamp(history.index.min()), pd.Timestamp(history.index.max()) - pd.DateOffset(years=10))
    latest_backtest = pd.Timestamp(history.index.max()) - pd.DateOffset(months=12)
    bt_end_default = max(earliest_backtest.date(), latest_backtest.date())
    stored_bt_start = safe_session_date(
        st.session_state.get("historical_bt_start"),
        fallback=earliest_backtest.date(),
        min_date=min_date,
        max_date=max_date,
    )
    stored_bt_end = safe_session_date(
        st.session_state.get("historical_bt_end"),
        fallback=bt_end_default,
        min_date=min_date,
        max_date=max_date,
    )
    st.session_state["historical_bt_start"] = stored_bt_start
    st.session_state["historical_bt_end"] = stored_bt_end
    st.session_state.setdefault("historical_bt_threshold", 65)
    st.session_state.setdefault("historical_bt_cooldown", 6)
    st.session_state.setdefault("historical_bt_compare_thresholds", [55, 65, 75])

    bt_cols = st.columns(4)
    bt_start = bt_cols[0].date_input("Start", min_value=min_date, max_value=max_date, key="historical_bt_start")
    bt_end = bt_cols[1].date_input("Ende", min_value=min_date, max_value=max_date, key="historical_bt_end")
    primary_threshold = bt_cols[2].slider("Haupt-Schwelle", 45, 85, step=5, key="historical_bt_threshold")
    cooldown = bt_cols[3].slider(
        "Mindestabstand Signale", 1, 24, step=1, key="historical_bt_cooldown",
        help="Verhindert, dass derselbe lange Drawdown jeden Monat als neuer Treffer gezählt wird.",
    )
    compare_thresholds = st.multiselect(
        "Zusätzliche Score-Schwellen vergleichen",
        options=list(range(45, 90, 5)),
        key="historical_bt_compare_thresholds",
        help="Die Haupt-Schwelle wird automatisch ergänzt, auch wenn sie hier nicht ausgewählt ist.",
    )

    run_key = f"deep_value_backtest_bundle_{ticker}_{benchmark_ticker}"
    if st.button(f"Historischen Deep-Value-Score für {ticker} testen", type="primary", key="run_historical_bat_backtest"):
        if pd.Timestamp(bt_start) > pd.Timestamp(bt_end):
            st.error("Das Startdatum muss vor dem Enddatum liegen.")
        else:
            with st.spinner("Berechne historische Monats-Momentaufnahmen und Folgerenditen …"):
                candidates = historical_signal_candidates(
                    ticker=ticker,
                    history=history,
                    dividends=dividends,
                    financials=financials,
                    benchmark=benchmark,
                    start_date=pd.Timestamp(bt_start),
                    end_date=pd.Timestamp(bt_end),
                    info_lag_days=int(lag_days),
                )
                thresholds = sorted({float(primary_threshold), *[float(v) for v in compare_thresholds]})
                results = {
                    threshold: filter_historical_signals(candidates, threshold, int(cooldown))
                    for threshold in thresholds
                }
            st.session_state[run_key] = {
                "results": results,
                "candidates": candidates,
                "start": pd.Timestamp(bt_start),
                "end": pd.Timestamp(bt_end),
                "cooldown": int(cooldown),
                "benchmark_label": benchmark_label,
                "benchmark_ticker": benchmark_ticker,
            }

    bundle = st.session_state.get(run_key)
    if isinstance(bundle, dict) and isinstance(bundle.get("results"), dict):
        results: dict[float, pd.DataFrame] = bundle["results"]
        summary = backtest_threshold_summary(results)

        tested_info = st.columns(3)
        tested_info[0].success(f"**Geprüfte Aktie:** {company_name} ({ticker})")
        tested_info[1].success(f"**Benchmark:** {bundle.get('benchmark_label')} ({bundle.get('benchmark_ticker')})")
        tested_info[2].success(
            f"**Zeitraum:** {pd.Timestamp(bundle.get('start')).strftime('%d.%m.%Y')} – "
            f"{pd.Timestamp(bundle.get('end')).strftime('%d.%m.%Y')}"
        )

        st.markdown("#### Aktuelles Setup & historische Erwartung")
        st.caption(
            "Verbindet den heutigen quantitativen Deep-Value-Score mit früheren, ähnlich bewerteten "
            "12M-Fällen derselben Aktie. Die Bandbreiten sind historische Erfahrungswerte – keine Kursziele."
        )

        latest_date = pd.Timestamp(history.index.max()).tz_localize(None).normalize()
        current_snapshot = historical_bat_snapshot(
            ticker=ticker,
            as_of=latest_date,
            history=history,
            dividends=dividends,
            financials=financials,
            info_lag_days=int(lag_days),
            negative_news_shock=False,
            special_effect_suspected=False,
        )
        current_score = safe_float(current_snapshot.get("bat_pattern_score")) or 0.0
        current_trigger = bool(current_snapshot.get("bat_pattern_trigger"))
        coverage_count, coverage_total, coverage_pct = deep_value_snapshot_coverage(current_snapshot)

        expectation_band = st.slider(
            "Ähnlichkeit zum heutigen Score (± Punkte)",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
            key="current_expectation_score_band",
            help="Historische Fälle werden zunächst innerhalb dieses Score-Bands gesucht. Bei zu wenigen Fällen wird die Auswahl transparent erweitert.",
        )
        expectation = historical_expectation_for_current_score(
            candidates=bundle.get("candidates", pd.DataFrame()),
            current_score=current_score,
            score_band=float(expectation_band),
            cooldown_months=int(bundle.get("cooldown", 12)),
        )
        evidence_label, evidence_text = expectation_evidence_label(int(expectation.get("count") or 0))
        expectation_label, expectation_text = classify_historical_expectation(expectation)

        current_cols = st.columns(5)
        current_cols[0].metric("Deep-Value-Score heute", format_number(current_score, 1))
        current_cols[1].metric("Aktuelles Signal", "⚠ aktiv" if current_trigger else "nicht aktiv")
        current_cols[2].metric("Datenabdeckung", f"{coverage_pct:.0f} %")
        current_cols[3].metric("Historische Fälle", int(expectation.get("count") or 0))
        current_cols[4].metric("Aussagekraft", evidence_label)

        if int(expectation.get("count") or 0) < 5:
            st.error(
                f"Historische Warnung: Nur {int(expectation.get('count') or 0)} unabhängige Vergleichsfälle. "
                "Prozentwerte und Szenarien sind statistisch nicht belastbar und dürfen nicht als Prognose gelesen werden."
            )

        if current_trigger:
            st.warning(
                "Das heutige quantitative Setup erfüllt den strengen Deep-Value-Trigger. "
                "Das ist ein Research-Hinweis und kein Kauf- oder Renditeversprechen."
            )
        elif current_score >= 55:
            st.info("Mehrere heutige Deep-Value-Merkmale sind vorhanden, der strenge Trigger ist jedoch nicht vollständig erfüllt.")
        else:
            st.info("Heute liegt kein ausgeprägtes quantitatives Deep-Value-Setup vor. Die Historie dient nur als Kontext.")

        valid_expectation = expectation.get("valid", pd.DataFrame())
        if isinstance(valid_expectation, pd.DataFrame) and not valid_expectation.empty:
            result_cols = st.columns(6)
            result_cols[0].metric("Positiv nach 12M", format_percent(expectation.get("positive_rate"), 1))
            result_cols[1].metric("Benchmark geschlagen", format_percent(expectation.get("benchmark_rate"), 1))
            result_cols[2].metric("Medianrendite 12M", format_percent(expectation.get("median_return"), 2, signed=True))
            result_cols[3].metric("Ø Rendite 12M", format_percent(expectation.get("average_return"), 2, signed=True))
            result_cols[4].metric("Value-Trap-Anteil", format_percent(expectation.get("trap_rate"), 1))
            result_cols[5].metric("Typischer Rückgang", format_percent(expectation.get("typical_drawdown"), 1, signed=True))

            scenario_cols = st.columns(3)
            scenario_cols[0].metric("Schwaches Szenario · 20 %", format_percent(expectation.get("weak_scenario"), 2, signed=True))
            scenario_cols[1].metric("Mittleres Szenario · Median", format_percent(expectation.get("middle_scenario"), 2, signed=True))
            scenario_cols[2].metric("Starkes Szenario · 80 %", format_percent(expectation.get("strong_scenario"), 2, signed=True))

            scenario_frame = pd.DataFrame({
                "Historische Bandbreite": ["Schwach · 20. Perzentil", "Mitte · Median", "Stark · 80. Perzentil"],
                "Rendite 12M": [
                    expectation.get("weak_scenario"),
                    expectation.get("middle_scenario"),
                    expectation.get("strong_scenario"),
                ],
            }).dropna(subset=["Rendite 12M"])
            if not scenario_frame.empty:
                scenario_chart = alt.Chart(scenario_frame).mark_bar().encode(
                    x=alt.X("Historische Bandbreite:N", title=None, sort=None),
                    y=alt.Y("Rendite 12M:Q", title="Historische 12M-Rendite (%)"),
                    tooltip=["Historische Bandbreite:N", alt.Tooltip("Rendite 12M:Q", format="+.2f")],
                ).properties(height=240)
                zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
                st.altair_chart(scenario_chart + zero_rule, use_container_width=True)

            if expectation_label == "historisch positiv":
                st.success(f"**Historische Einordnung: {expectation_label}.** {expectation_text}")
            elif expectation_label == "vorsichtig positiv":
                st.info(f"**Historische Einordnung: {expectation_label}.** {expectation_text}")
            elif expectation_label == "historisch schwach":
                st.error(f"**Historische Einordnung: {expectation_label}.** {expectation_text}")
            else:
                st.warning(f"**Historische Einordnung: {expectation_label}.** {expectation_text}")

            st.caption(
                f"Vergleichsauswahl: {expectation.get('selection_text')}; zeitlicher Abstand mindestens "
                f"{max(12, int(bundle.get('cooldown', 12)))} Monate. {evidence_text}"
            )
            if expectation.get("expanded"):
                st.warning("Im engen Score-Band gab es weniger als drei Fälle. Die Vergleichsmenge wurde deshalb erweitert.")

            strengths, risks = current_setup_strengths_and_risks(current_snapshot, expectation)
            strength_col, risk_col = st.columns(2)
            with strength_col:
                st.markdown("**Aktuelle Stärken**")
                if strengths:
                    for item in strengths:
                        st.markdown(f"- ✓ {item}")
                else:
                    st.caption("Keine klaren Stärken aus den verfügbaren Kennzahlen ableitbar.")
            with risk_col:
                st.markdown("**Aktuelle Risiken / offene Prüfungen**")
                if risks:
                    for item in risks:
                        st.markdown(f"- ⚠ {item}")
                else:
                    st.caption("Keine zusätzlichen Warnhinweise aus den verfügbaren Kennzahlen.")

            with st.expander("Historische Vergleichsfälle anzeigen", expanded=False):
                expectation_columns = [
                    "Signal-Datum", "Deep-Value-Score", "Rendite 12M", "Überrendite 12M",
                    "Max. Rückgang 12M", "Value Trap", "Value-Trap-Grund",
                ]
                expectation_display = valid_expectation[
                    [column for column in expectation_columns if column in valid_expectation.columns]
                ].copy()
                if "Signal-Datum" in expectation_display.columns:
                    expectation_display["Signal-Datum"] = pd.to_datetime(
                        expectation_display["Signal-Datum"], errors="coerce"
                    ).dt.strftime("%d.%m.%Y")
                st.dataframe(
                    expectation_display.style.format(
                        {
                            "Deep-Value-Score": "{:.1f}",
                            "Rendite 12M": lambda value: format_percent(value, 2, signed=True),
                            "Überrendite 12M": lambda value: format_percent(value, 2, signed=True),
                            "Max. Rückgang 12M": lambda value: format_percent(value, 1, signed=True),
                        },
                        na_rep="–",
                    ).map(
                        colorize_change,
                        subset=[column for column in ["Rendite 12M", "Überrendite 12M"] if column in expectation_display.columns],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning(
                "Für den heutigen Score wurden keine unabhängigen historischen Fälle mit vollständigen 12M-Daten gefunden. "
                "Erweitere den Testzeitraum oder das Score-Band."
            )

        st.markdown("#### Schwellenvergleich")
        st.dataframe(
            summary.style.format(
                {
                    "Schwelle": "{:.0f}",
                    "Positive Treffer 12M": lambda value: format_percent(value, 1),
                    "Benchmark geschlagen 12M": lambda value: format_percent(value, 1),
                    "Value-Trap-Quote": lambda value: format_percent(value, 1),
                    "Ø Rendite 12M": lambda value: format_percent(value, 2, signed=True),
                    "Ø Überrendite 12M": lambda value: format_percent(value, 2, signed=True),
                },
                na_rep="–",
            ).map(colorize_change, subset=["Ø Rendite 12M", "Ø Überrendite 12M"]),
            use_container_width=True,
            hide_index=True,
        )

        thresholds_available = sorted(results)
        detail_default = float(primary_threshold) if float(primary_threshold) in thresholds_available else thresholds_available[0]
        stored_detail_threshold = st.session_state.get("historical_bt_detail_threshold")
        if stored_detail_threshold not in thresholds_available:
            st.session_state.pop("historical_bt_detail_threshold", None)
        detail_threshold = st.selectbox(
            "Detailansicht für Score-Schwelle",
            options=thresholds_available,
            index=thresholds_available.index(detail_default),
            format_func=lambda value: f"Score ab {value:.0f}",
            key="historical_bt_detail_threshold",
        )
        backtest_result = results[detail_threshold]

        if isinstance(backtest_result, pd.DataFrame) and not backtest_result.empty:
            valid_12m = backtest_result.dropna(subset=["Rendite 12M"])
            stats = st.columns(6)
            stats[0].metric("Signale", len(backtest_result))
            stats[1].metric("Mit 12M-Daten", len(valid_12m))
            positive_rate = float(valid_12m["Positiv nach 12M"].mean() * 100) if not valid_12m.empty else None
            benchmark_rate = float(valid_12m["Benchmark geschlagen 12M"].mean() * 100) if not valid_12m.empty else None
            trap_rate = float(valid_12m["Value Trap"].mean() * 100) if not valid_12m.empty else None
            stats[2].metric("Positiv nach 12M", format_percent(positive_rate, 1))
            stats[3].metric("Benchmark geschlagen", format_percent(benchmark_rate, 1))
            stats[4].metric("Value-Trap-Quote", format_percent(trap_rate, 1))
            stats[5].metric("Ø Überrendite 12M", format_percent(valid_12m["Überrendite 12M"].mean() if not valid_12m.empty else None, 2, signed=True))

            display_columns = [
                "Signal-Datum", "Deep-Value-Score", "Kurs", "Dividendenrendite", "Drawdown",
                "FCF/Dividende", "Net Debt/EBITDA",
                "Rendite 3M", "Rendite 6M", "Rendite 12M", "Rendite 24M",
                "Überrendite 12M", "Benchmark geschlagen 12M",
                "Max. Rückgang 12M", "Dividendenänderung 12M", "Value Trap", "Value-Trap-Grund",
            ]
            display_result = backtest_result[[column for column in display_columns if column in backtest_result.columns]].copy()
            display_result["Signal-Datum"] = pd.to_datetime(display_result["Signal-Datum"], errors="coerce").dt.strftime("%d.%m.%Y")
            st.dataframe(
                display_result.style.format(
                    {
                        "Deep-Value-Score": "{:.1f}",
                        "Kurs": "{:.2f}",
                        "Dividendenrendite": lambda value: format_percent(value, 2),
                        "Drawdown": lambda value: format_percent(value, 1, signed=True),
                        "FCF/Dividende": lambda value: f"{format_number(value, 2)}x",
                        "Net Debt/EBITDA": lambda value: f"{format_number(value, 2)}x",
                        "Rendite 3M": lambda value: format_percent(value, 2, signed=True),
                        "Rendite 6M": lambda value: format_percent(value, 2, signed=True),
                        "Rendite 12M": lambda value: format_percent(value, 2, signed=True),
                        "Rendite 24M": lambda value: format_percent(value, 2, signed=True),
                        "Überrendite 12M": lambda value: format_percent(value, 2, signed=True),
                        "Max. Rückgang 12M": lambda value: format_percent(value, 1, signed=True),
                        "Dividendenänderung 12M": lambda value: format_percent(value, 1, signed=True),
                    },
                    na_rep="–",
                ).map(colorize_change, subset=["Rendite 3M", "Rendite 6M", "Rendite 12M", "Rendite 24M", "Überrendite 12M"]),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Signal im Detail erklären")
            signal_options = list(backtest_result.index)
            selected_signal_index = st.selectbox(
                "Historisches Signal",
                options=signal_options,
                format_func=lambda idx: (
                    f"{pd.Timestamp(backtest_result.loc[idx, 'Signal-Datum']).strftime('%d.%m.%Y')} · "
                    f"Score {safe_float(backtest_result.loc[idx, 'Deep-Value-Score']) or 0:.1f}"
                ),
                key="historical_bt_signal_explanation",
            )
            selected_signal = backtest_result.loc[selected_signal_index]
            selected_snapshot = historical_bat_snapshot(
                ticker=ticker,
                as_of=pd.Timestamp(selected_signal["Signal-Datum"]),
                history=history,
                dividends=dividends,
                financials=financials,
                info_lag_days=int(lag_days),
                negative_news_shock=False,
                special_effect_suspected=False,
            )
            explanation_cols = st.columns(3)
            explanation_cols[0].metric("Score", format_number(selected_signal.get("Deep-Value-Score"), 1))
            explanation_cols[1].metric("12M-Überrendite", format_percent(selected_signal.get("Überrendite 12M"), 2, signed=True))
            explanation_cols[2].metric("Nachträgliche Einordnung", "Value Trap" if bool(selected_signal.get("Value Trap")) else "kein Trap-Signal")
            if bool(selected_signal.get("Value Trap")):
                st.warning(str(selected_signal.get("Value-Trap-Grund") or "Nachträgliches Value-Trap-Kriterium erfüllt."))
            components = selected_snapshot.get("components", pd.DataFrame())
            if isinstance(components, pd.DataFrame) and not components.empty:
                st.dataframe(components, use_container_width=True, hide_index=True)

            equity = model_signal_equity_curve(backtest_result)
            if not equity.empty:
                st.markdown("#### Modellhafte Signal-Kurve")
                st.caption(
                    "Startwert 100; jedes historische Signal wird nacheinander mit seiner 12M-Rendite verknüpft. "
                    "Die Darstellung ist ein Schwellenvergleich und keine kalendergenaue Portfoliosimulation."
                )
                equity_long = equity.melt(
                    id_vars=["Signal", "Datum"],
                    value_vars=["Deep-Value-Strategie", "Benchmark"],
                    var_name="Reihe",
                    value_name="Indexwert",
                )
                equity_chart = alt.Chart(equity_long).mark_line(point=True).encode(
                    x=alt.X("Signal:Q", title="Abgeschlossenes 12M-Signal"),
                    y=alt.Y("Indexwert:Q", title="Modellhafter Indexwert", scale=alt.Scale(zero=False)),
                    color=alt.Color("Reihe:N", title=None),
                    tooltip=["Reihe:N", "Signal:Q", alt.Tooltip("Indexwert:Q", format=".2f")],
                ).properties(height=320)
                st.altair_chart(equity_chart, use_container_width=True)

            action_cols = st.columns(2)
            action_cols[0].download_button(
                "Backtest als CSV herunterladen",
                data=backtest_result.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"deep_value_backtest_{ticker.replace('.', '_')}_{int(detail_threshold)}.csv",
                mime="text/csv",
                key="download_bat_backtest",
            )
            if action_cols[1].button("Backtest dauerhaft speichern", key="persist_deep_value_backtest"):
                success, message = persist_backtest_run(
                    ticker=ticker,
                    company_name=company_name,
                    benchmark_label=benchmark_label,
                    benchmark_ticker=benchmark_ticker,
                    start_date=bundle.get("start"),
                    end_date=bundle.get("end"),
                    threshold=float(detail_threshold),
                    cooldown_months=int(bundle.get("cooldown", cooldown)),
                    result=backtest_result,
                )
                if success:
                    st.success(message)
                else:
                    st.warning(message)
        else:
            st.info(f"Für Score ab {detail_threshold:.0f} wurden im gewählten Zeitraum keine Signale gefunden.")
    else:
        st.info("Starte die quantitative Prüfung, um Schwellenwerte, Trefferquoten und Value Traps zu vergleichen.")

    if BACKTEST_RUNS_PATH.exists():
        with st.expander("Gespeicherte Backtest-Läufe", expanded=False):
            try:
                saved_runs = pd.read_csv(BACKTEST_RUNS_PATH)
                st.dataframe(saved_runs.sort_values("saved_at", ascending=False), use_container_width=True, hide_index=True)
            except Exception as error:
                st.warning(f"Gespeicherte Backtests konnten nicht gelesen werden: {error}")

    st.divider()
    st.markdown("### Fall im eigenen Lernjournal speichern")
    journal_cols = st.columns([1, 2])
    label = journal_cols[0].selectbox("Eigene Einordnung", ["offen", "guter Treffer", "Value Trap", "zu riskant", "verpasst"], key="research_case_label")
    notes = journal_cols[1].text_input("Notiz", placeholder="Was war damals besonders?", key="research_case_notes")
    if st.button("Fall speichern", key="save_research_case"):
        def _forward_metric(horizon: str, column: str) -> Any:
            if forward is None or forward.empty:
                return None
            match_row = forward[forward["Horizont"] == horizon]
            return match_row.iloc[0].get(column) if not match_row.empty else None

        return_12m = _forward_metric("12 Monate", "Aktienrendite")
        benchmark_12m = _forward_metric("12 Monate", "Benchmark")
        excess_12m = _forward_metric("12 Monate", "Überrendite")
        success, message = _save_research_case({
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "ticker_yahoo": ticker,
            "company_name": company_name,
            "sector": company_sector,
            "as_of": pd.Timestamp(snapshot["price_date"]).date().isoformat(),
            "price": snapshot.get("price"),
            "currency": snapshot.get("currency"),
            "deep_value_score": score,
            "bat_pattern_score": score,
            "trigger": trigger,
            "return_3m": _forward_metric("3 Monate", "Aktienrendite"),
            "return_6m": _forward_metric("6 Monate", "Aktienrendite"),
            "return_12m": return_12m,
            "return_24m": _forward_metric("24 Monate", "Aktienrendite"),
            "benchmark_12m": benchmark_12m,
            "excess_return_12m": excess_12m,
            "label": label,
            "notes": notes,
            "dividend_yield": snapshot.get("dividend_yield"),
            "drawdown_3_5y": snapshot.get("deepest_drawdown_3_5y_pct"),
            "fcf_dividend_coverage": snapshot.get("cashflow_dividend_coverage"),
            "net_debt_ebitda": snapshot.get("net_debt_ebitda"),
            "pe_approx": snapshot.get("pe_approx"),
            "volatility_1y": snapshot.get("volatility_1y"),
            "cashflow_positive": snapshot.get("cashflow_positive"),
            "reported_eps_negative": snapshot.get("reported_eps_negative"),
            "special_effect_suspected": snapshot.get("special_effect_suspected"),
            "negative_news_shock": snapshot.get("negative_news_shock"),
        })
        if success:
            st.success("Fall wurde unter data/research_cases.csv gespeichert.")
        else:
            st.warning(message)

    if RESEARCH_CASES_PATH.exists():
        with st.expander("Gespeicherte Lernfälle", expanded=False):
            try:
                cases = load_research_cases()
                st.dataframe(cases, use_container_width=True, hide_index=True)
            except Exception as error:
                st.warning(f"Lernjournal konnte nicht gelesen werden: {error}")


# -----------------------------------------------------------------------------
# Persönliches Lernmodul
# -----------------------------------------------------------------------------

LEARNING_FEATURES: dict[str, dict[str, Any]] = {
    "deep_value_score": {"label": "Deep-Value-Score", "scale": 35.0, "format": "number"},
    "dividend_yield": {"label": "Dividendenrendite", "scale": 8.0, "format": "percent"},
    "drawdown_3_5y": {"label": "3J-/5J-Drawdown", "scale": 45.0, "format": "percent"},
    "fcf_dividend_coverage": {"label": "FCF/Dividende", "scale": 2.0, "format": "multiple"},
    "net_debt_ebitda": {"label": "Net Debt/EBITDA", "scale": 5.0, "format": "multiple"},
    "pe_approx": {"label": "Historisches KGV", "scale": 25.0, "format": "number"},
    "volatility_1y": {"label": "Volatilität 1J", "scale": 40.0, "format": "percent"},
    "quality_score": {"label": "Qualitäts-Score", "scale": 50.0, "format": "number"},
    "value_score": {"label": "Value-Score", "scale": 50.0, "format": "number"},
    "growth_score": {"label": "Growth-Score", "scale": 50.0, "format": "number"},
    "momentum_score": {"label": "Momentum-Score", "scale": 50.0, "format": "number"},
    "safety_score": {"label": "Safety-Score", "scale": 50.0, "format": "number"},
}


def _normalise_learning_label(value: Any) -> str:
    text_value = str(value or "offen").strip().lower()
    aliases = {
        "guter treffer": "guter Treffer",
        "erfolg": "guter Treffer",
        "value trap": "Value Trap",
        "value-trap": "Value Trap",
        "zu riskant": "zu riskant",
        "verpasst": "verpasst",
        "offen": "offen",
    }
    return aliases.get(text_value, str(value or "offen").strip() or "offen")


def prepare_learning_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Bereitet manuelle und automatisch auswertbare Lernfälle einheitlich auf."""
    if cases is None or cases.empty:
        return pd.DataFrame(columns=RESEARCH_CASE_COLUMNS)
    frame = deduplicate_dataframe_columns(cases).copy()
    for column in RESEARCH_CASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    numeric_columns = [
        "price", "deep_value_score", "bat_pattern_score", "return_3m", "return_6m",
        "return_12m", "return_24m", "benchmark_12m", "excess_return_12m",
        *LEARNING_FEATURES.keys(),
    ]
    for column in dict.fromkeys(numeric_columns):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["deep_value_score"] = frame["deep_value_score"].fillna(frame["bat_pattern_score"])
    frame["excess_return_12m"] = frame["excess_return_12m"].fillna(
        frame["return_12m"] - frame["benchmark_12m"]
    )
    frame["label"] = frame["label"].map(_normalise_learning_label)
    frame["as_of"] = pd.to_datetime(frame["as_of"], errors="coerce")

    def classify(row: pd.Series) -> str:
        label = _normalise_learning_label(row.get("label"))
        if label == "guter Treffer":
            return "Erfolg"
        if label == "Value Trap":
            return "Value Trap"
        return_12m = safe_float(row.get("return_12m"))
        excess = safe_float(row.get("excess_return_12m"))
        if return_12m is not None and excess is not None:
            if return_12m > 0 and excess > 0:
                return "Erfolg (automatisch)"
            if return_12m < 0 and excess < 0:
                return "Schwach (automatisch)"
        return "Offen"

    frame["learning_outcome"] = frame.apply(classify, axis=1)
    frame["binary_outcome"] = frame["learning_outcome"].map({
        "Erfolg": 1.0,
        "Erfolg (automatisch)": 1.0,
        "Value Trap": 0.0,
        "Schwach (automatisch)": 0.0,
    })
    available_feature_columns = [column for column in LEARNING_FEATURES if column in frame.columns]
    frame["feature_count"] = frame[available_feature_columns].notna().sum(axis=1) if available_feature_columns else 0
    frame["feature_coverage"] = frame["feature_count"] / max(len(LEARNING_FEATURES), 1) * 100
    return frame


def learning_evidence_label(count: int) -> tuple[str, str]:
    if count < 3:
        return "sehr gering", "Weniger als drei auswertbare Fälle – nur als Notizsammlung verwenden."
    if count < 6:
        return "gering", "Erste Tendenzen sind sichtbar, aber einzelne Fälle dominieren das Ergebnis."
    if count < 15:
        return "mittel", "Mehrere unabhängige Fälle erlauben eine vorsichtige statistische Einordnung."
    if count < 30:
        return "gut", "Die Stichprobe ist für ein persönliches Regelwerk brauchbar, bleibt aber selektiv."
    return "höher", "Die Stichprobe ist vergleichsweise breit; Datenqualität und Auswahlverzerrung bleiben wichtig."


def learning_feature_summary(cases: pd.DataFrame) -> pd.DataFrame:
    """Vergleicht robuste Mediane erfolgreicher und schwacher Fälle je Merkmal."""
    prepared = prepare_learning_cases(cases)
    binary = prepared.dropna(subset=["binary_outcome"]).copy()
    if binary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for column, config in LEARNING_FEATURES.items():
        if column not in binary.columns:
            continue
        success = pd.to_numeric(binary.loc[binary["binary_outcome"] == 1, column], errors="coerce").dropna()
        weak = pd.to_numeric(binary.loc[binary["binary_outcome"] == 0, column], errors="coerce").dropna()
        if success.empty or weak.empty:
            continue
        success_median = safe_float(success.median())
        weak_median = safe_float(weak.median())
        difference = success_median - weak_median if success_median is not None and weak_median is not None else None
        scale = float(config.get("scale", 1.0) or 1.0)
        separation = min(abs(difference or 0.0) / scale * 100.0, 100.0)
        direction = "bei Erfolgen höher" if (difference or 0) > 0 else "bei Erfolgen niedriger" if (difference or 0) < 0 else "kein Unterschied"
        count = len(success) + len(weak)
        evidence, _ = learning_evidence_label(count)
        rows.append({
            "Merkmal": config["label"],
            "Spalte": column,
            "Median gute Treffer": success_median,
            "Median schwache Fälle": weak_median,
            "Differenz": difference,
            "Trennschärfe": separation,
            "Beobachtung": direction,
            "Erfolg-Fälle": len(success),
            "Schwache Fälle": len(weak),
            "Aussagekraft": evidence,
        })
    return pd.DataFrame(rows).sort_values(["Trennschärfe", "Erfolg-Fälle"], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame()


def _current_learning_snapshot(row: pd.Series) -> dict[str, Any]:
    return {
        "ticker_yahoo": clean_ticker(row.get("ticker_yahoo")),
        "company_name": row.get("name") or row.get("ticker_yahoo"),
        "sector": row.get("sector") or "Unbekannt",
        "deep_value_score": safe_float(row.get("special_situation_score")),
        "dividend_yield": safe_float(row.get("dividend_yield")),
        "drawdown_3_5y": safe_float(row.get("deepest_drawdown_3_5y_pct")),
        "fcf_dividend_coverage": safe_float(row.get("cashflow_dividend_coverage")),
        "net_debt_ebitda": safe_float(row.get("net_debt_ebitda")),
        "pe_approx": safe_float(row.get("pe_ratio")),
        "volatility_1y": safe_float(row.get("vol_1y")),
        "quality_score": safe_float(row.get("total_score")),
        "value_score": safe_float(row.get("value_profile_score", row.get("value_score"))),
        "growth_score": safe_float(row.get("growth_score")),
        "momentum_score": safe_float(row.get("momentum_score")),
        "safety_score": safe_float(row.get("safety_score")),
    }


def learning_case_similarity(current: dict[str, Any], case: pd.Series) -> dict[str, Any]:
    components: list[float] = []
    details: list[str] = []
    for column, config in LEARNING_FEATURES.items():
        current_value = safe_float(current.get(column))
        historical_value = safe_float(case.get(column))
        if current_value is None or historical_value is None:
            continue
        scale = float(config.get("scale", 1.0) or 1.0)
        similarity = max(0.0, min(1.0, 1.0 - abs(current_value - historical_value) / scale))
        components.append(similarity)
        details.append(f"{config['label']}: {similarity * 100:.0f} %")
    if not components:
        return {"similarity": None, "coverage": 0.0, "details": "Keine gemeinsamen Merkmale"}
    similarity = sum(components) / len(components)
    current_sector = str(current.get("sector") or "").strip().lower()
    historical_sector = str(case.get("sector") or "").strip().lower()
    if current_sector and historical_sector and current_sector == historical_sector:
        similarity = min(1.0, similarity + 0.05)
        details.append("Sektorbonus: +5 Punkte")
    return {
        "similarity": similarity * 100.0,
        "coverage": len(components) / max(len(LEARNING_FEATURES), 1) * 100.0,
        "details": " | ".join(details),
    }


def current_learning_comparison(current: dict[str, Any], cases: pd.DataFrame, min_coverage: float = 20.0) -> pd.DataFrame:
    prepared = prepare_learning_cases(cases)
    if prepared.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, case in prepared.iterrows():
        similarity = learning_case_similarity(current, case)
        if similarity.get("similarity") is None or float(similarity.get("coverage") or 0) < float(min_coverage):
            continue
        rows.append({
            "Fall-ID": case.get("case_id"),
            "Ticker": case.get("ticker_yahoo"),
            "Unternehmen": case.get("company_name") or case.get("ticker_yahoo"),
            "Sektor": case.get("sector") or "Unbekannt",
            "Stichtag": case.get("as_of"),
            "Ähnlichkeit": similarity.get("similarity"),
            "Merkmalsabdeckung": similarity.get("coverage"),
            "Einordnung": case.get("learning_outcome"),
            "Rendite 12M": case.get("return_12m"),
            "Überrendite 12M": case.get("excess_return_12m"),
            "Notiz": case.get("notes"),
            "Ähnlichkeitsdetails": similarity.get("details"),
        })
    return pd.DataFrame(rows).sort_values(["Ähnlichkeit", "Merkmalsabdeckung"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def weighted_learning_expectation(comparisons: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    if comparisons is None or comparisons.empty:
        return {"count": 0}
    frame = comparisons.head(int(top_n)).copy()
    frame["weight"] = (pd.to_numeric(frame["Ähnlichkeit"], errors="coerce") / 100.0).clip(lower=0) ** 2
    valid = frame.dropna(subset=["Rendite 12M", "weight"])
    weight_sum = safe_float(valid["weight"].sum()) if not valid.empty else None
    weighted_return = None
    if weight_sum not in (None, 0):
        weighted_return = safe_float((valid["Rendite 12M"] * valid["weight"]).sum() / weight_sum)
    success = frame["Einordnung"].astype(str).str.startswith("Erfolg")
    weak = frame["Einordnung"].astype(str).isin(["Value Trap", "Schwach (automatisch)"])
    return {
        "count": len(frame),
        "weighted_return": weighted_return,
        "success_rate": float(success.mean() * 100) if len(frame) else None,
        "weak_rate": float(weak.mean() * 100) if len(frame) else None,
        "average_similarity": safe_float(pd.to_numeric(frame["Ähnlichkeit"], errors="coerce").mean()),
    }


def _render_learning_metric(value: Any, kind: str) -> str:
    if kind == "percent":
        return format_percent(value, 2, signed=True)
    if kind == "multiple":
        return f"{format_number(value, 2)}x"
    return format_number(value, 2)


def render_learning_module(df: pd.DataFrame) -> None:
    st.subheader("Persönliches Lernmodul")
    st.caption(
        "Das Lernmodul wertet dein eigenes Research-Journal transparent aus. Es sucht robuste "
        "Unterschiede zwischen guten Treffern und schwachen Fällen und vergleicht heutige Aktien "
        "mit deinen gespeicherten Beispielen. Es ist bewusst kein Black-Box-Kaufsignal."
    )

    with st.expander("So arbeitet das Lernmodul", expanded=False):
        st.markdown(
            """
            1. Speichere historische Fälle im Bereich **Backtesting** und ordne sie ein.
            2. Das Modul bevorzugt deine manuelle Bewertung. Bei offenen Fällen kann eine spätere
               positive Rendite und Überrendite vorsichtig als automatischer Erfolg gelten.
            3. Merkmale werden über robuste Mediane verglichen – nicht über ein undurchsichtiges KI-Modell.
            4. Für eine heutige Aktie werden die ähnlichsten gespeicherten Fälle gesucht.

            **Wichtig:** Kleine oder selektiv gespeicherte Stichproben können stark verzerren.
            Das Journal sollte auch schlechte, verpasste und unsichere Fälle enthalten.
            """
        )

    raw_cases = load_research_cases()
    cases = prepare_learning_cases(raw_cases)
    if cases.empty:
        st.info(
            "Noch keine Lernfälle vorhanden. Öffne den Bereich Backtesting, analysiere einen "
            "historischen Stichtag und speichere den Fall im Lernjournal."
        )
        return

    evaluable = cases.dropna(subset=["binary_outcome"])
    evidence, evidence_text = learning_evidence_label(len(evaluable))
    metrics = st.columns(6)
    metrics[0].metric("Gespeicherte Fälle", len(cases))
    metrics[1].metric("Auswertbar", len(evaluable))
    metrics[2].metric("Gute Treffer", int((cases["learning_outcome"].astype(str).str.startswith("Erfolg")).sum()))
    metrics[3].metric("Schwache/Trap-Fälle", int(cases["learning_outcome"].isin(["Value Trap", "Schwach (automatisch)"]).sum()))
    metrics[4].metric("Ø 12M-Rendite", format_percent(cases["return_12m"].mean(), 2, signed=True))
    metrics[5].metric("Aussagekraft", evidence)
    if len(evaluable) < 6:
        st.warning(f"Noch geringe Lernbasis: {evidence_text}")
    else:
        st.info(evidence_text)

    st.markdown("### Journal pflegen")
    st.caption("Du kannst Einordnung und Notiz direkt ändern. Die Rohdaten des historischen Falls bleiben unverändert.")
    editable_columns = [
        "case_id", "ticker_yahoo", "company_name", "sector", "as_of", "return_12m",
        "excess_return_12m", "label", "notes",
    ]
    editor_frame = cases[[column for column in editable_columns if column in cases.columns]].copy()
    if "as_of" in editor_frame.columns:
        editor_frame["as_of"] = pd.to_datetime(editor_frame["as_of"], errors="coerce").dt.date
    edited = st.data_editor(
        editor_frame,
        use_container_width=True,
        hide_index=True,
        disabled=[column for column in editor_frame.columns if column not in {"label", "notes"}],
        key="learning_case_editor",
    )
    action_cols = st.columns([1, 1, 2])
    if action_cols[0].button("Änderungen speichern", key="save_learning_edits"):
        updated = raw_cases.copy()
        if "case_id" in edited.columns:
            edit_map = edited.set_index("case_id")
            for idx, row in updated.iterrows():
                case_id = row.get("case_id")
                if case_id in edit_map.index:
                    updated.at[idx, "label"] = edit_map.at[case_id, "label"]
                    updated.at[idx, "notes"] = edit_map.at[case_id, "notes"]
        success, message = safe_write_csv(updated[RESEARCH_CASE_COLUMNS], RESEARCH_CASES_PATH)
        if success:
            st.success("Lernjournal aktualisiert.")
            st.rerun()
        else:
            st.warning(message)
    action_cols[1].download_button(
        "Journal exportieren",
        data=raw_cases.to_csv(index=False).encode("utf-8-sig"),
        file_name="research_cases.csv",
        mime="text/csv",
        key="download_learning_journal",
    )

    st.markdown("### Was hat das Journal bisher gelernt?")
    summary = learning_feature_summary(cases)
    if summary.empty:
        st.info(
            "Für einen Merkmalsvergleich braucht das Journal mindestens einen auswertbaren guten "
            "und einen schwachen Fall mit gemeinsamen Kennzahlen."
        )
    else:
        top_summary = summary.head(8).copy()
        display_summary = top_summary[[
            "Merkmal", "Median gute Treffer", "Median schwache Fälle", "Beobachtung",
            "Trennschärfe", "Erfolg-Fälle", "Schwache Fälle", "Aussagekraft",
        ]]
        st.dataframe(
            display_summary.style.format({
                "Median gute Treffer": "{:.2f}",
                "Median schwache Fälle": "{:.2f}",
                "Trennschärfe": "{:.0f} %",
            }, na_rep="–"),
            use_container_width=True,
            hide_index=True,
        )
        best = top_summary.iloc[0]
        st.info(
            f"Stärkste bisher beobachtete Trennung: **{best['Merkmal']}** – "
            f"{best['Beobachtung']} (Trennschärfe {safe_float(best['Trennschärfe']) or 0:.0f} %). "
            "Das ist eine Beobachtung deines Journals, keine allgemein gültige Börsenregel."
        )
        chart_data = top_summary[["Merkmal", "Trennschärfe", "Beobachtung"]].copy()
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Trennschärfe:Q", title="Robuste Trennschärfe (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Merkmal:N", sort="-x", title=None),
            tooltip=["Merkmal:N", alt.Tooltip("Trennschärfe:Q", format=".0f"), "Beobachtung:N"],
        ).properties(height=max(240, len(chart_data) * 34))
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Heutige Aktie mit dem eigenen Journal vergleichen")
    if df is None or df.empty:
        st.info("Lade zuerst einen Index, um eine heutige Aktie mit dem Lernjournal zu vergleichen.")
        return
    ticker = company_selectbox("Aktie auswählen", df, key="learning_current_ticker")
    selected = df[df["ticker_yahoo"].astype(str) == ticker]
    if selected.empty:
        return
    current = _current_learning_snapshot(selected.iloc[0])
    min_coverage = st.slider(
        "Minimale gemeinsame Merkmalsabdeckung",
        10, 80, value=25, step=5, key="learning_min_similarity_coverage",
        help="Alte Lernfälle enthalten eventuell noch weniger Merkmale. Je höher der Wert, desto strenger der Vergleich.",
    )
    comparisons = current_learning_comparison(current, cases, min_coverage=float(min_coverage))
    if comparisons.empty:
        st.info("Keine gespeicherten Fälle mit genügend gemeinsamen Merkmalen gefunden.")
        return

    expectation = weighted_learning_expectation(comparisons, top_n=min(10, len(comparisons)))
    comparison_cols = st.columns(5)
    comparison_cols[0].metric("Vergleichsfälle", expectation.get("count", 0))
    comparison_cols[1].metric("Ø Ähnlichkeit", format_percent(expectation.get("average_similarity"), 1))
    comparison_cols[2].metric("Gewichtete 12M-Rendite", format_percent(expectation.get("weighted_return"), 2, signed=True))
    comparison_cols[3].metric("Erfolgsanteil", format_percent(expectation.get("success_rate"), 1))
    comparison_cols[4].metric("Schwach/Trap", format_percent(expectation.get("weak_rate"), 1))

    comparison_evidence, comparison_text = learning_evidence_label(int(expectation.get("count") or 0))
    if int(expectation.get("count") or 0) < 6:
        st.warning(f"Aussagekraft {comparison_evidence}: {comparison_text}")
    else:
        st.info(f"Aussagekraft {comparison_evidence}: {comparison_text}")

    display_comparisons = comparisons.head(15).copy()
    display_comparisons["Stichtag"] = pd.to_datetime(display_comparisons["Stichtag"], errors="coerce").dt.strftime("%d.%m.%Y")
    st.dataframe(
        display_comparisons[[
            "Unternehmen", "Ticker", "Stichtag", "Ähnlichkeit", "Merkmalsabdeckung",
            "Einordnung", "Rendite 12M", "Überrendite 12M", "Notiz",
        ]].style.format({
            "Ähnlichkeit": "{:.0f} %",
            "Merkmalsabdeckung": "{:.0f} %",
            "Rendite 12M": lambda value: format_percent(value, 2, signed=True),
            "Überrendite 12M": lambda value: format_percent(value, 2, signed=True),
        }, na_rep="–").map(colorize_change, subset=["Rendite 12M", "Überrendite 12M"]),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Warum gelten diese Fälle als ähnlich?", expanded=False):
        detail_case = st.selectbox(
            "Vergleichsfall",
            options=list(display_comparisons.index),
            format_func=lambda idx: (
                f"{display_comparisons.loc[idx, 'Unternehmen']} ({display_comparisons.loc[idx, 'Ticker']}) · "
                f"{safe_float(display_comparisons.loc[idx, 'Ähnlichkeit']) or 0:.0f} %"
            ),
            key="learning_similarity_detail",
        )
        st.write(display_comparisons.loc[detail_case, "Ähnlichkeitsdetails"])

    st.caption(
        "Die gewichtete Rendite ist eine Zusammenfassung deiner ähnlichsten gespeicherten Fälle. "
        "Sie ist keine Prognose und kann durch selektive Fallauswahl, kleine Stichproben und fehlende Daten verzerrt sein."
    )


def render_research(df: pd.DataFrame, histories: dict[str, pd.DataFrame], index_name: str) -> None:
    st.subheader("Research & historische Einordnung")
    st.caption(
        "Dieser Bereich vergleicht historische Kurs-/Adj-Close-Entwicklung mit einem Benchmark. "
        "Er ist bewusst kein fundamentaler Backtest: Historische Fundamentals, damalige Indexzusammensetzungen und Transaktionskosten fehlen dafür."
    )
    ticker = company_selectbox("Aktie für Vergleich", df, key="research_ticker")
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
            {"Serie": f"{df.loc[df["ticker_yahoo"] == ticker, "name"].iloc[0]} ({ticker})", "Rendite": company_metrics["return_pct"], "Volatilität": company_metrics["volatility_pct"], "Max. Drawdown": company_metrics["max_drawdown_pct"]},
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
            "Bewertung", "Qualität", "Dividende", "Cashflow", "Deep Value", "Fehler/Hinweis",
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



def _score_label(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "keine Daten"
    if number >= 75:
        return "stark"
    if number >= 55:
        return "solide"
    if number >= 35:
        return "schwach"
    return "sehr schwach"


def render_profile_scores(df: pd.DataFrame) -> None:
    st.subheader("Aktienprofile: Value, Growth, Momentum & Safety")
    st.caption(
        "Die vier Profile trennen unterschiedliche Eigenschaften einer Aktie. Ein hoher Value-Score bedeutet nicht automatisch "
        "hohe Sicherheit; ein starkes Momentum bedeutet nicht automatisch eine günstige Bewertung."
    )
    if df is None or df.empty:
        st.info("Keine Daten für die aktuelle Filtereinstellung.")
        return

    # Schutz vor doppelten Spaltennamen aus älteren Session-State-Ständen.
    df = deduplicate_dataframe_columns(df)

    with st.expander("So werden die Profile interpretiert", expanded=False):
        explanation = pd.DataFrame([
            {"Profil": "Value", "Frage": "Wie günstig oder ertragsstark wirkt die Aktie?", "Wichtige Daten": "Bewertung, Dividende, Drawdown, Payout, Verschuldung"},
            {"Profil": "Growth", "Frage": "Wie stark wachsen Umsatz und Ergebnis?", "Wichtige Daten": "Umsatz-/Gewinnwachstum, Marge, ROE"},
            {"Profil": "Momentum", "Frage": "Wie stabil ist der aktuelle Kurstrend?", "Wichtige Daten": "1M/3M/6M/12M-Performance, SMA 50/200"},
            {"Profil": "Safety", "Frage": "Wie widerstandsfähig wirkt die Aktie?", "Wichtige Daten": "Volatilität, Drawdown, Bilanz, Cashflow, Payout"},
        ])
        st.dataframe(explanation, hide_index=True, use_container_width=True)
        st.warning("Die Scores sind heuristische Recherchehilfen. Fehlende Daten senken die Abdeckung und dämpfen den jeweiligen Score.")

    ticker = company_selectbox("Aktie auswählen", df, key="profile_scores_ticker")
    row = df.loc[df["ticker_yahoo"] == ticker].iloc[0]
    company_name = display_text(row.get("name"), fallback=str(ticker))
    st.markdown(f"### {company_name} ({ticker})")

    score_rows = [
        ("Value", row.get("value_score"), row.get("value_coverage"), row.get("value_reason")),
        ("Growth", row.get("growth_score"), row.get("growth_coverage"), row.get("growth_components")),
        ("Momentum", row.get("momentum_score"), row.get("momentum_coverage"), row.get("momentum_components")),
        ("Safety", row.get("safety_score"), row.get("safety_coverage"), row.get("safety_components")),
        ("Deep Value", row.get("special_situation_score"), row.get("special_situation_coverage"), row.get("special_situation_reason")),
    ]

    metric_cols = st.columns(5)
    for column, (label, value, coverage, _) in zip(metric_cols, score_rows):
        column.metric(label, format_number(value, 1), help=f"Einordnung: {_score_label(value)} · Datenabdeckung: {format_percent(coverage, 0)}")

    score_frame = pd.DataFrame([
        {"Profil": label, "Score": safe_float(value), "Datenabdeckung": safe_float(coverage), "Einordnung": _score_label(value)}
        for label, value, coverage, _ in score_rows
    ])
    chart_data = score_frame.dropna(subset=["Score"])
    if not chart_data.empty:
        bars = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Score (0–100)"),
            y=alt.Y("Profil:N", sort=["Value", "Growth", "Momentum", "Safety", "Deep Value"], title=None),
            tooltip=["Profil:N", alt.Tooltip("Score:Q", format=".1f"), alt.Tooltip("Datenabdeckung:Q", format=".0f"), "Einordnung:N"],
        ).properties(height=260)
        labels = alt.Chart(chart_data).mark_text(align="left", dx=5).encode(
            x="Score:Q", y=alt.Y("Profil:N", sort=["Value", "Growth", "Momentum", "Safety", "Deep Value"]),
            text=alt.Text("Score:Q", format=".1f"),
        )
        st.altair_chart(bars + labels, use_container_width=True)

    detail = pd.DataFrame([
        {
            "Profil": label,
            "Score": safe_float(value),
            "Datenabdeckung": safe_float(coverage),
            "Einordnung": _score_label(value),
            "Bausteine / Erklärung": display_text(details),
        }
        for label, value, coverage, details in score_rows
    ])
    st.dataframe(
        detail.style.format({
            "Score": lambda value: format_number(value, 1),
            "Datenabdeckung": lambda value: format_percent(value, 0),
        }, na_rep="–").map(colorize_score, subset=["Score"]),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Vergleich aller geladenen Aktien")
    comparison_columns = [
        "name", "ticker_yahoo", "sector", "value_score", "growth_score", "momentum_score", "safety_score",
        "special_situation_score", "total_score", "score_coverage",
    ]
    available = [column for column in comparison_columns if column in df.columns]
    comparison = df[available].copy().rename(columns={
        "name": "Unternehmen", "ticker_yahoo": "Ticker", "sector": "Sektor",
        "value_score": "Value", "growth_score": "Growth", "momentum_score": "Momentum",
        "safety_score": "Safety", "special_situation_score": "Deep Value",
        "total_score": "Qualität", "score_coverage": "Datenabdeckung",
    })
    sort_profile = st.selectbox("Sortieren nach", ["Value", "Growth", "Momentum", "Safety", "Deep Value", "Qualität"], key="profile_sort")
    if sort_profile in comparison.columns:
        comparison = comparison.sort_values(sort_profile, ascending=False, na_position="last")
    score_cols = [column for column in ["Value", "Growth", "Momentum", "Safety", "Deep Value", "Qualität"] if column in comparison.columns]
    styled = comparison.style.format(
        {column: lambda value: format_number(value, 1) for column in score_cols} |
        ({"Datenabdeckung": lambda value: format_percent(value, 0)} if "Datenabdeckung" in comparison.columns else {}),
        na_rep="–",
    ).map(colorize_score, subset=score_cols)
    st.dataframe(styled, hide_index=True, use_container_width=True)


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


# -----------------------------------------------------------------------------
# Fundamentaldaten-Ampeln (Version 5.6)
# -----------------------------------------------------------------------------

FUNDAMENTAL_STATUS_META: dict[str, dict[str, str]] = {
    "good": {"icon": "🟢", "label": "unauffällig / stark"},
    "watch": {"icon": "🟡", "label": "prüfen"},
    "critical": {"icon": "🔴", "label": "kritisch"},
    "info": {"icon": "⚪", "label": "nur informativ"},
    "missing": {"icon": "⚪", "label": "keine Daten"},
}

FUNDAMENTAL_METRIC_LABELS: dict[str, str] = {
    "pe_ratio": "KGV",
    "forward_pe": "Forward KGV",
    "pb_ratio": "KBV",
    "ps_ratio": "KUV",
    "ev_ebitda": "EV/EBITDA",
    "revenue_growth": "Umsatzwachstum",
    "earnings_growth": "Ergebniswachstum",
    "operating_margin": "Operative Marge",
    "net_margin": "Nettomarge",
    "roe": "Eigenkapitalrendite",
    "roa": "Gesamtkapitalrendite",
    "dividend_yield": "Dividendenrendite",
    "payout_ratio": "Ausschüttungsquote",
    "dividend_growth_5y": "Dividendenwachstum 5J",
    "cashflow_dividend_coverage": "FCF/Dividende",
    "free_cashflow": "Free Cashflow",
    "debt_to_equity": "Debt/Equity",
    "net_debt_ebitda": "Netto-Schulden/EBITDA",
    "current_ratio": "Current Ratio",
}

FUNDAMENTAL_METRIC_GROUPS: dict[str, list[str]] = {
    "Bewertung": ["pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda"],
    "Wachstum & Profitabilität": [
        "revenue_growth", "earnings_growth", "operating_margin", "net_margin", "roe", "roa"
    ],
    "Dividende & Cashflow": [
        "dividend_yield", "payout_ratio", "dividend_growth_5y",
        "cashflow_dividend_coverage", "free_cashflow",
    ],
    "Bilanz": ["debt_to_equity", "net_debt_ebitda", "current_ratio"],
}

FUNDAMENTAL_OVERVIEW_METRICS: list[str] = [
    "pe_ratio", "ev_ebitda", "operating_margin", "roe", "revenue_growth",
    "payout_ratio", "cashflow_dividend_coverage", "net_debt_ebitda",
]


def fundamental_sector_group(sector: Any) -> str:
    """Ordnet uneinheitliche Yahoo-/Index-Sektorbezeichnungen grob ein."""
    text = str(sector or "").strip().lower()
    if is_financial_sector(text):
        return "financial"
    if any(term in text for term in ("real estate", "immobil", "reit")):
        return "real_estate"
    if any(term in text for term in ("utility", "utilities", "versorgung")):
        return "utilities"
    if any(term in text for term in ("energy", "oil", "gas", "energie")):
        return "energy"
    if any(term in text for term in ("technology", "software", "semiconductor", "information tech")):
        return "technology"
    if any(term in text for term in ("communication", "telecom", "media")):
        return "communication"
    if any(term in text for term in ("health", "pharma", "biotech")):
        return "healthcare"
    if any(term in text for term in ("consumer staples", "basic consumer", "nahr", "tobacco")):
        return "staples"
    if any(term in text for term in ("consumer discretionary", "cyclical", "retail", "automotive", "auto")):
        return "discretionary"
    if any(term in text for term in ("industrial", "maschinen", "aerospace", "transport")):
        return "industrials"
    if any(term in text for term in ("material", "chemical", "mining", "rohstoff")):
        return "materials"
    return "general"


def _fundamental_result(
    status: str,
    explanation: str,
    criterion: str,
    *,
    applicable: bool = True,
) -> dict[str, Any]:
    meta = FUNDAMENTAL_STATUS_META[status]
    return {
        "status": status,
        "icon": meta["icon"],
        "label": meta["label"],
        "explanation": explanation,
        "criterion": criterion,
        "applicable": bool(applicable),
    }


def _threshold_assessment(
    value: Optional[float],
    *,
    good_if: callable,
    watch_if: callable,
    good_text: str,
    watch_text: str,
    critical_text: str,
    explanation_good: str,
    explanation_watch: str,
    explanation_critical: str,
) -> dict[str, Any]:
    if value is None:
        return _fundamental_result("missing", "Keine verwertbaren Daten vorhanden.", f"{good_text}; {watch_text}; {critical_text}")
    if good_if(value):
        return _fundamental_result("good", explanation_good, good_text)
    if watch_if(value):
        return _fundamental_result("watch", explanation_watch, watch_text)
    return _fundamental_result("critical", explanation_critical, critical_text)


def assess_fundamental_metric(metric: str, row: pd.Series) -> dict[str, Any]:
    """Bewertet eine Kennzahl sektorabhängig mit Ampel und transparenter Regel.

    Die Ampel ist eine Recherchehilfe. Sie bewertet eine Kennzahl isoliert und
    darf nicht als Kauf-/Verkaufsempfehlung interpretiert werden.
    """
    value = safe_float(row.get(metric))
    if metric == "earnings_growth" and value is None:
        value = safe_float(row.get("earnings_quarterly_growth"))
    sector_group = fundamental_sector_group(row.get("sector"))
    dividend_yield = safe_float(row.get("dividend_yield"))

    if metric in {"pe_ratio", "forward_pe"}:
        label = "KGV" if metric == "pe_ratio" else "Forward KGV"
        if sector_group == "real_estate":
            return _fundamental_result(
                "info", f"{label} ist bei Immobilien-/REIT-Unternehmen oft durch Abschreibungen verzerrt; FFO/AFFO wäre geeigneter.",
                "Bei REITs nur eingeschränkt vergleichbar.", applicable=False,
            )
        if value is None:
            return _fundamental_result("missing", "Kein positives, verwertbares KGV verfügbar.", "Positives KGV erforderlich.")
        if value <= 0:
            return _fundamental_result(
                "critical", "Ein negatives oder nullnahes KGV bedeutet meist Verlust beziehungsweise einen nicht sinnvollen Quotienten.",
                "KGV muss positiv sein.",
            )
        if sector_group in {"technology", "communication"}:
            good, watch = 25.0, 40.0
        elif sector_group == "financial":
            good, watch = 14.0, 22.0
        else:
            good, watch = 18.0, 28.0
        return _threshold_assessment(
            value,
            good_if=lambda x: x <= good,
            watch_if=lambda x: x <= watch,
            good_text=f"0 < {label} ≤ {good:g}",
            watch_text=f"{good:g} < {label} ≤ {watch:g}",
            critical_text=f"{label} > {watch:g} oder ≤ 0",
            explanation_good="Bewertung liegt innerhalb des für den Sektor konservativ angesetzten Bereichs.",
            explanation_watch="Bewertung ist erhöht und sollte gegen Wachstum, Qualität und Historie geprüft werden.",
            explanation_critical="Bewertung ist sehr hoch oder wegen Verlusten nicht sinnvoll interpretierbar.",
        )

    if metric == "pb_ratio":
        if sector_group in {"technology", "communication"}:
            return _fundamental_result(
                "info", "Bei asset-light Geschäftsmodellen ist das KBV häufig wenig aussagekräftig.",
                "Nur zusammen mit Profitabilität und Geschäftsmodell verwenden.", applicable=False,
            )
        if sector_group == "financial":
            good, watch = 1.5, 2.5
        elif sector_group == "real_estate":
            good, watch = 1.2, 2.0
        else:
            good, watch = 3.0, 6.0
        return _threshold_assessment(
            value,
            good_if=lambda x: 0 <= x <= good,
            watch_if=lambda x: 0 <= x <= watch,
            good_text=f"0 ≤ KBV ≤ {good:g}",
            watch_text=f"{good:g} < KBV ≤ {watch:g}",
            critical_text=f"KBV > {watch:g} oder < 0",
            explanation_good="Buchwertbewertung wirkt für den Sektor nicht überzogen.",
            explanation_watch="Buchwertbewertung ist erhöht; Kapitalrendite und immaterielle Werte mitprüfen.",
            explanation_critical="Sehr hohe oder negative Buchwertbewertung erfordert eine genauere Bilanzprüfung.",
        )

    if metric == "ps_ratio":
        if sector_group in {"financial", "real_estate"}:
            return _fundamental_result(
                "info", "Das KUV ist für Finanz- und Immobilienunternehmen nur eingeschränkt vergleichbar.",
                "Für diesen Sektor keine harte Ampel.", applicable=False,
            )
        good, watch = (5.0, 10.0) if sector_group == "technology" else (2.5, 5.0)
        return _threshold_assessment(
            value,
            good_if=lambda x: 0 <= x <= good,
            watch_if=lambda x: 0 <= x <= watch,
            good_text=f"0 ≤ KUV ≤ {good:g}",
            watch_text=f"{good:g} < KUV ≤ {watch:g}",
            critical_text=f"KUV > {watch:g} oder < 0",
            explanation_good="Umsatzbewertung liegt im sektorüblich konservativen Bereich.",
            explanation_watch="Umsatzbewertung ist erhöht und setzt gute Margen beziehungsweise Wachstum voraus.",
            explanation_critical="Sehr hohe Umsatzbewertung birgt ein erhöhtes Enttäuschungsrisiko.",
        )

    if metric == "ev_ebitda":
        if sector_group in {"financial", "real_estate"}:
            return _fundamental_result(
                "info", "EV/EBITDA ist bei Finanzwerten und vielen REITs nicht die bevorzugte Kennzahl.",
                "Für diesen Sektor keine harte Ampel.", applicable=False,
            )
        if sector_group in {"technology", "communication"}:
            good, watch = 18.0, 28.0
        elif sector_group in {"utilities", "energy"}:
            good, watch = 10.0, 16.0
        else:
            good, watch = 12.0, 18.0
        return _threshold_assessment(
            value,
            good_if=lambda x: 0 <= x <= good,
            watch_if=lambda x: 0 <= x <= watch,
            good_text=f"0 ≤ EV/EBITDA ≤ {good:g}",
            watch_text=f"{good:g} < EV/EBITDA ≤ {watch:g}",
            critical_text=f"EV/EBITDA > {watch:g} oder < 0",
            explanation_good="Unternehmenswert im Verhältnis zum EBITDA wirkt nicht überzogen.",
            explanation_watch="Bewertung ist erhöht; Wachstum und Kapitalintensität sollten sie rechtfertigen.",
            explanation_critical="Sehr hohe oder negative Kennzahl weist auf teure Bewertung beziehungsweise negatives EBITDA hin.",
        )

    if metric in {"revenue_growth", "earnings_growth"}:
        name = "Umsatzwachstum" if metric == "revenue_growth" else "Ergebniswachstum"
        good_threshold = 3.0 if sector_group in {"utilities", "staples", "financial"} else 5.0
        return _threshold_assessment(
            value,
            good_if=lambda x: x >= good_threshold,
            watch_if=lambda x: x >= 0,
            good_text=f"{name} ≥ {good_threshold:g} %",
            watch_text=f"0 % ≤ {name} < {good_threshold:g} %",
            critical_text=f"{name} < 0 %",
            explanation_good=f"{name} ist positiv und liegt oberhalb der konservativen Sektorschwelle.",
            explanation_watch=f"{name} ist positiv, aber schwach; Nachhaltigkeit und Basiseffekte prüfen.",
            explanation_critical=f"{name} ist negativ und kann auf zyklischen oder strukturellen Druck hinweisen.",
        )

    if metric in {"operating_margin", "net_margin"}:
        if metric == "operating_margin" and sector_group == "financial":
            return _fundamental_result(
                "info", "Die operative Marge ist bei Banken und Versicherern nicht direkt mit Industrieunternehmen vergleichbar.",
                "Für Finanzwerte nur eingeschränkt verwenden.", applicable=False,
            )
        thresholds = {
            "technology": (20.0, 10.0),
            "communication": (15.0, 7.0),
            "healthcare": (15.0, 7.0),
            "staples": (12.0, 6.0),
            "discretionary": (10.0, 4.0),
            "industrials": (12.0, 6.0),
            "materials": (12.0, 5.0),
            "utilities": (15.0, 8.0),
            "energy": (15.0, 5.0),
            "financial": (12.0, 6.0),
            "general": (12.0, 5.0),
        }
        good, watch = thresholds.get(sector_group, thresholds["general"])
        if metric == "net_margin":
            good *= 0.8
            watch *= 0.8
        name = "Operative Marge" if metric == "operating_margin" else "Nettomarge"
        return _threshold_assessment(
            value,
            good_if=lambda x: x >= good,
            watch_if=lambda x: x >= watch,
            good_text=f"{name} ≥ {good:.1f} %",
            watch_text=f"{watch:.1f} % ≤ {name} < {good:.1f} %",
            critical_text=f"{name} < {watch:.1f} %",
            explanation_good=f"{name} ist im groben Sektorvergleich stark.",
            explanation_watch=f"{name} ist positiv, besitzt aber begrenzten Puffer.",
            explanation_critical=f"{name} ist niedrig oder negativ und sollte über mehrere Jahre geprüft werden.",
        )

    if metric == "roe":
        good, watch = (12.0, 8.0) if sector_group == "financial" else (15.0, 8.0)
        return _threshold_assessment(
            value,
            good_if=lambda x: x >= good,
            watch_if=lambda x: x >= watch,
            good_text=f"ROE ≥ {good:g} %",
            watch_text=f"{watch:g} % ≤ ROE < {good:g} %",
            critical_text=f"ROE < {watch:g} %",
            explanation_good="Die Eigenkapitalrendite ist im groben Sektorvergleich stark.",
            explanation_watch="Die Eigenkapitalrendite ist positiv, aber nur mittelmäßig.",
            explanation_critical="Niedrige oder negative Eigenkapitalrendite weist auf schwache Kapitalproduktivität hin.",
        )

    if metric == "roa":
        good, watch = (1.0, 0.5) if sector_group == "financial" else (6.0, 3.0)
        return _threshold_assessment(
            value,
            good_if=lambda x: x >= good,
            watch_if=lambda x: x >= watch,
            good_text=f"ROA ≥ {good:g} %",
            watch_text=f"{watch:g} % ≤ ROA < {good:g} %",
            critical_text=f"ROA < {watch:g} %",
            explanation_good="Die Gesamtkapitalrendite ist für den Sektor solide.",
            explanation_watch="Die Gesamtkapitalrendite ist positiv, aber mit begrenztem Puffer.",
            explanation_critical="Die Vermögensbasis wird nur schwach oder negativ verzinst.",
        )

    if metric == "dividend_yield":
        if value is None:
            return _fundamental_result("missing", "Keine Dividendenrendite verfügbar.", "Rendite allein ist kein Qualitätsmerkmal.")
        if value <= 0:
            return _fundamental_result(
                "info", "Keine oder keine verlässlich erfasste Dividende. Das ist bei Wachstumsunternehmen nicht automatisch negativ.",
                "Keine harte Ampel ohne Dividendenstrategie.", applicable=False,
            )
        if value > 10:
            return _fundamental_result(
                "critical", "Eine zweistellige Rendite ist häufig ein Warnsignal für Kursverfall oder Kürzungsrisiko.",
                "> 10 %: sehr hohe Rendite, Nachhaltigkeit zwingend prüfen.",
            )
        if value > 6.5:
            return _fundamental_result(
                "watch", "Die Rendite ist hoch. Payout, Cashflow-Deckung und Sondereffekte sollten geprüft werden.",
                "6,5–10 %: hoch, mögliche Chance oder Dividendenfalle.",
            )
        if value < 1:
            return _fundamental_result(
                "watch", "Die Rendite ist niedrig; bei Wachstumswerten kann das normal sein.",
                "0–1 %: niedrig, strategisch einordnen.",
            )
        return _fundamental_result(
            "good", "Die Rendite liegt in einem üblichen Bereich, ihre Nachhaltigkeit hängt aber von Payout und Cashflow ab.",
            "1–6,5 %: normaler Bereich; Qualität separat prüfen.",
        )

    if metric == "payout_ratio":
        if dividend_yield is None or dividend_yield <= 0:
            return _fundamental_result(
                "info", "Ohne relevante Dividende ist die Ausschüttungsquote für die Ampel nicht entscheidend.",
                "Nur bei Dividendenzahlern bewerten.", applicable=False,
            )
        if sector_group == "real_estate":
            good, watch = 100.0, 130.0
        else:
            good, watch = 70.0, 90.0
        return _threshold_assessment(
            value,
            good_if=lambda x: 0 <= x <= good,
            watch_if=lambda x: 0 <= x <= watch,
            good_text=f"0–{good:g} %",
            watch_text=f"> {good:g} bis {watch:g} %",
            critical_text=f"> {watch:g} % oder < 0 %",
            explanation_good="Die Ausschüttung besitzt nach dieser einfachen Regel einen angemessenen Puffer.",
            explanation_watch="Die Ausschüttung ist hoch; Cashflow-Deckung und Gewinnqualität prüfen.",
            explanation_critical="Die Ausschüttung wirkt nicht nachhaltig oder die Ergebnisbasis ist verzerrt.",
        )

    if metric == "dividend_growth_5y":
        if dividend_yield is None or dividend_yield <= 0:
            return _fundamental_result(
                "info", "Ohne relevante Dividende wird das Dividendenwachstum nicht gewertet.",
                "Nur bei Dividendenzahlern bewerten.", applicable=False,
            )
        return _threshold_assessment(
            value,
            good_if=lambda x: x >= 5,
            watch_if=lambda x: x >= 0,
            good_text="Wachstum ≥ 5 % p. a.",
            watch_text="0–5 % p. a.",
            critical_text="Wachstum < 0 % p. a.",
            explanation_good="Die Dividende ist über abgeschlossene Jahre deutlich gewachsen.",
            explanation_watch="Die Dividende war stabil bis moderat wachsend.",
            explanation_critical="Die Dividende ist im Vergleichszeitraum gesunken.",
        )

    if metric == "cashflow_dividend_coverage":
        if dividend_yield is None or dividend_yield <= 0:
            return _fundamental_result(
                "info", "Ohne relevante Dividende ist die Cashflow-Deckung nicht anwendbar.",
                "Nur bei Dividendenzahlern bewerten.", applicable=False,
            )
        return _threshold_assessment(
            value,
            good_if=lambda x: x >= 1.5,
            watch_if=lambda x: x >= 1.0,
            good_text="FCF/Dividende ≥ 1,5x",
            watch_text="1,0x bis < 1,5x",
            critical_text="< 1,0x",
            explanation_good="Der Free Cashflow deckt die geschätzte Dividendenzahlung mit gutem Puffer.",
            explanation_watch="Die Dividende ist gedeckt, der Puffer ist jedoch begrenzt.",
            explanation_critical="Der Free Cashflow deckt die Dividende nicht vollständig; Kürzungsrisiko prüfen.",
        )

    if metric == "free_cashflow":
        if value is None:
            return _fundamental_result("missing", "Kein Free Cashflow verfügbar.", "Positiver Free Cashflow ist grundsätzlich günstiger.")
        if value > 0:
            return _fundamental_result("good", "Free Cashflow ist positiv.", "Free Cashflow > 0")
        if value == 0:
            return _fundamental_result("watch", "Free Cashflow liegt ungefähr bei null.", "Free Cashflow ≈ 0")
        return _fundamental_result("critical", "Free Cashflow ist negativ.", "Free Cashflow < 0")

    if metric == "net_debt_ebitda":
        if sector_group == "financial":
            return _fundamental_result(
                "info", "Net Debt/EBITDA ist bei Banken und Versicherern nicht sinnvoll mit Industrieunternehmen vergleichbar.",
                "Für Finanzwerte regulatorische Kapital- und Liquiditätsquoten nutzen.", applicable=False,
            )
        if sector_group in {"utilities", "real_estate"}:
            good, watch = 3.0, 5.0
        elif sector_group == "energy":
            good, watch = 2.5, 4.0
        else:
            good, watch = 2.0, 3.5
        return _threshold_assessment(
            value,
            good_if=lambda x: x <= good,
            watch_if=lambda x: x <= watch,
            good_text=f"Net Debt/EBITDA ≤ {good:g}x",
            watch_text=f"> {good:g}x bis {watch:g}x",
            critical_text=f"> {watch:g}x",
            explanation_good="Verschuldung wirkt gemessen am EBITDA beherrschbar.",
            explanation_watch="Verschuldung ist erhöht; Zinslast und Entschuldungspfad prüfen.",
            explanation_critical="Hohe Verschuldung kann bei Ergebnisrückgang oder steigenden Zinsen problematisch werden.",
        )

    if metric == "debt_to_equity":
        if sector_group == "financial":
            return _fundamental_result(
                "info", "Debt/Equity ist bei Finanzunternehmen strukturell hoch und deshalb nicht direkt vergleichbar.",
                "Für Finanzwerte regulatorische Kapitalquoten bevorzugen.", applicable=False,
            )
        if sector_group in {"utilities", "real_estate"}:
            good, watch = 1.5, 3.0
        else:
            good, watch = 1.0, 2.0
        return _threshold_assessment(
            value,
            good_if=lambda x: 0 <= x <= good,
            watch_if=lambda x: 0 <= x <= watch,
            good_text=f"0–{good:g}x",
            watch_text=f"> {good:g}x bis {watch:g}x",
            critical_text=f"> {watch:g}x oder < 0",
            explanation_good="Fremdkapitalquote wirkt im groben Sektorvergleich moderat.",
            explanation_watch="Fremdkapitalquote ist erhöht; absolute Schulden und Cashflow mitprüfen.",
            explanation_critical="Sehr hohe oder wegen negativem Eigenkapital verzerrte Quote ist kritisch.",
        )

    if metric == "current_ratio":
        if sector_group == "financial":
            return _fundamental_result(
                "info", "Die Current Ratio ist bei Banken und Versicherern nicht die passende Liquiditätskennzahl.",
                "Für Finanzwerte nicht anwenden.", applicable=False,
            )
        good, watch = (0.8, 0.6) if sector_group == "utilities" else (1.5, 1.0)
        return _threshold_assessment(
            value,
            good_if=lambda x: x >= good,
            watch_if=lambda x: x >= watch,
            good_text=f"Current Ratio ≥ {good:g}",
            watch_text=f"{watch:g} bis < {good:g}",
            critical_text=f"< {watch:g}",
            explanation_good="Kurzfristige Vermögenswerte decken kurzfristige Verpflichtungen mit Puffer.",
            explanation_watch="Liquidität ist ausreichend bis knapp und sollte im Verlauf geprüft werden.",
            explanation_critical="Kurzfristige Liquiditätsdeckung wirkt angespannt.",
        )

    return _fundamental_result("info", "Für diese Kennzahl ist keine Ampellogik definiert.", "Nur informativ.", applicable=False)


def format_fundamental_metric_value(metric: str, value: Any) -> str:
    if metric in {
        "revenue_growth", "earnings_growth", "operating_margin", "net_margin", "roe", "roa",
        "dividend_yield", "payout_ratio", "dividend_growth_5y",
    }:
        return format_percent(value, 1)
    if metric == "free_cashflow":
        return human_market_cap(value)
    if metric in {"cashflow_dividend_coverage", "debt_to_equity", "net_debt_ebitda"}:
        number = safe_float(value)
        return "–" if number is None else f"{format_number(number, 2)}x"
    return format_number(value, 2)


def build_fundamental_assessment(row: pd.Series) -> dict[str, Any]:
    assessments: dict[str, dict[str, Any]] = {}
    for metric in FUNDAMENTAL_METRIC_LABELS:
        assessments[metric] = assess_fundamental_metric(metric, row)

    applicable = [item for item in assessments.values() if item.get("applicable", True)]
    available = [item for item in applicable if item["status"] in {"good", "watch", "critical"}]
    counts = {
        status: sum(item["status"] == status for item in available)
        for status in ("good", "watch", "critical")
    }
    applicable_count = len(applicable)
    coverage = len(available) / applicable_count * 100.0 if applicable_count else 0.0
    red_ratio = counts["critical"] / len(available) if available else 0.0
    green_ratio = counts["good"] / len(available) if available else 0.0

    if len(available) < 5:
        overall = "missing"
        overall_text = "Zu wenig Daten"
    elif counts["critical"] >= 3 or red_ratio >= 0.35:
        overall = "critical"
        overall_text = "Mehrere kritische Punkte"
    elif counts["critical"] == 0 and green_ratio >= 0.65:
        overall = "good"
        overall_text = "Überwiegend unauffällig"
    else:
        overall = "watch"
        overall_text = "Gemischtes Bild – prüfen"

    meta = FUNDAMENTAL_STATUS_META[overall]
    return {
        "assessments": assessments,
        "overall_status": overall,
        "overall_label": f"{meta['icon']} {overall_text}",
        "green_count": counts["good"],
        "watch_count": counts["watch"],
        "critical_count": counts["critical"],
        "assessment_coverage": round(coverage, 1),
        "assessed_count": len(available),
    }


def enrich_with_fundamental_assessments(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    result = deduplicate_dataframe_columns(df)
    summaries = result.apply(lambda row: pd.Series({
        key: value for key, value in build_fundamental_assessment(row).items() if key != "assessments"
    }), axis=1)
    for column in summaries.columns:
        result[column] = summaries[column]
    return result


def _render_raw_fundamentals_table(df: pd.DataFrame) -> None:
    columns = [
        "name", "ticker_yahoo", "currency", "market_cap", "pe_ratio", "forward_pe", "pb_ratio",
        "ps_ratio", "ev_ebitda", "revenue_growth", "earnings_growth", "gross_margin", "net_margin",
        "operating_margin", "roe", "roa", "dividend_yield", "dividend_per_share", "payout_ratio",
        "dividend_growth_5y", "dividend_frequency", "dividend_yield_5y_avg",
        "dividend_yield_vs_5y_avg_pct", "free_cashflow", "operating_cashflow",
        "cashflow_dividend_coverage", "debt_to_equity", "net_debt_ebitda", "current_ratio", "beta",
        "score_raw", "score_coverage", "score_confidence", "total_score",
    ]
    visible_columns = [column for column in columns if column in df.columns]
    display_names = {
        "name": "Unternehmen", "ticker_yahoo": "Ticker", "currency": "Währung",
        "market_cap": "Marktkapitalisierung", "pe_ratio": "KGV", "forward_pe": "Forward KGV",
        "pb_ratio": "KBV", "ps_ratio": "KUV", "ev_ebitda": "EV/EBITDA",
        "revenue_growth": "Umsatzwachstum", "earnings_growth": "Ergebniswachstum",
        "gross_margin": "Bruttomarge", "net_margin": "Nettomarge", "operating_margin": "Operative Marge",
        "roe": "Eigenkapitalrendite", "roa": "Gesamtkapitalrendite",
        "dividend_yield": "Dividendenrendite", "dividend_per_share": "Dividende je Aktie",
        "payout_ratio": "Ausschüttungsquote", "dividend_growth_5y": "Dividendenwachstum 5J",
        "dividend_frequency": "Ausschüttungsrhythmus", "dividend_yield_5y_avg": "Ø Div.-Rendite 5J",
        "dividend_yield_vs_5y_avg_pct": "Rendite vs. 5J-Schnitt", "free_cashflow": "Free Cashflow",
        "operating_cashflow": "Operativer Cashflow", "cashflow_dividend_coverage": "FCF/Dividende",
        "debt_to_equity": "Debt/Equity", "net_debt_ebitda": "Netto-Schulden/EBITDA",
        "current_ratio": "Current Ratio", "beta": "Beta", "score_raw": "Rohscore",
        "score_coverage": "Datenabdeckung", "score_confidence": "Datenvertrauen",
        "total_score": "Qualitäts-Score",
    }
    display_df = df[visible_columns].rename(columns=display_names)
    formats = {
        "Marktkapitalisierung": human_market_cap,
        "KGV": lambda value: format_number(value, 2), "Forward KGV": lambda value: format_number(value, 2),
        "KBV": lambda value: format_number(value, 2), "KUV": lambda value: format_number(value, 2),
        "EV/EBITDA": lambda value: format_number(value, 2),
        "Umsatzwachstum": lambda value: format_percent(value, 2), "Ergebniswachstum": lambda value: format_percent(value, 2),
        "Bruttomarge": lambda value: format_percent(value, 2), "Nettomarge": lambda value: format_percent(value, 2),
        "Operative Marge": lambda value: format_percent(value, 2),
        "Eigenkapitalrendite": lambda value: format_percent(value, 2), "Gesamtkapitalrendite": lambda value: format_percent(value, 2),
        "Dividendenrendite": lambda value: format_percent(value, 2), "Dividende je Aktie": lambda value: format_number(value, 2),
        "Ausschüttungsquote": lambda value: format_percent(value, 2), "Dividendenwachstum 5J": lambda value: format_percent(value, 2),
        "Ø Div.-Rendite 5J": lambda value: format_percent(value, 2),
        "Rendite vs. 5J-Schnitt": lambda value: format_percent(value, 0, signed=True),
        "Free Cashflow": human_market_cap, "Operativer Cashflow": human_market_cap,
        "FCF/Dividende": lambda value: "–" if safe_float(value) is None else f"{format_number(value, 2)}x",
        "Debt/Equity": lambda value: "–" if safe_float(value) is None else f"{format_number(value, 2)}x",
        "Netto-Schulden/EBITDA": lambda value: "–" if safe_float(value) is None else f"{format_number(value, 2)}x",
        "Current Ratio": lambda value: format_number(value, 2), "Beta": lambda value: format_number(value, 2),
        "Rohscore": lambda value: format_number(value, 1), "Datenabdeckung": lambda value: format_percent(value, 0),
        "Qualitäts-Score": lambda value: format_number(value, 1),
    }
    styled = display_df.style.format(formats, na_rep="–").map(colorize_score, subset=["Qualitäts-Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_fundamentals(df: pd.DataFrame) -> None:
    st.subheader("Fundamentaldaten & Ampeln")
    if df is None or df.empty:
        st.info("Keine Fundamentaldaten geladen.")
        return

    clean_df = deduplicate_dataframe_columns(df)
    assessed_df = enrich_with_fundamental_assessments(clean_df)

    with st.expander("So funktionieren die Fundamentaldaten-Ampeln", expanded=False):
        st.markdown(
            "Die Ampeln bewerten **einzelne Kennzahlen isoliert** und verwenden grobe, teilweise sektorspezifische "
            "Schwellen. 🟢 bedeutet nicht automatisch günstig oder kaufenswert, 🔴 nicht automatisch verkaufen. "
            "Bei Banken, Versicherern und Immobilienwerten werden ungeeignete Kennzahlen bewusst grau statt rot dargestellt."
        )
        st.markdown(
            "**Legende:** 🟢 unauffällig/stark · 🟡 genauer prüfen · 🔴 kritisch · ⚪ nicht anwendbar oder keine Daten. "
            "Die Gesamtampel fasst nur anwendbare, verfügbare Kennzahlen zusammen und ersetzt weder die Aktienprofile "
            "noch eine Prüfung von Geschäftsmodell, Sondereffekten und mehrjährigen Trends."
        )

    view_mode = st.radio(
        "Darstellung",
        ["Ampelübersicht", "Einzelwert erklären", "Alle Rohdaten"],
        horizontal=True,
        key="fundamentals_view_mode",
    )

    if view_mode == "Ampelübersicht":
        col_filter, col_sort = st.columns([2, 1])
        with col_filter:
            status_options = ["🟢 Überwiegend unauffällig", "🟡 Gemischtes Bild", "🔴 Kritische Punkte", "⚪ Zu wenig Daten"]
            selected_statuses = st.multiselect(
                "Gesamtampel filtern",
                status_options,
                default=status_options,
                key="fundamental_overall_filter",
            )
        with col_sort:
            sort_choice = st.selectbox(
                "Sortieren nach",
                ["Unternehmen", "Kritische Punkte", "Grüne Punkte", "Datenabdeckung", "Qualitäts-Score"],
                key="fundamental_sort_choice",
            )

        status_bucket = {
            "good": "🟢 Überwiegend unauffällig",
            "watch": "🟡 Gemischtes Bild",
            "critical": "🔴 Kritische Punkte",
            "missing": "⚪ Zu wenig Daten",
        }
        rows: list[dict[str, Any]] = []
        for _, row in assessed_df.iterrows():
            summary = build_fundamental_assessment(row)
            bucket = status_bucket.get(summary["overall_status"], "⚪ Zu wenig Daten")
            if selected_statuses and bucket not in selected_statuses:
                continue
            output: dict[str, Any] = {
                "Unternehmen": row.get("name", row.get("ticker_yahoo", "–")),
                "Ticker": row.get("ticker_yahoo", "–"),
                "Sektor": row.get("sector", "–"),
                "Gesamtampel": summary["overall_label"],
                "🟢": summary["green_count"],
                "🟡": summary["watch_count"],
                "🔴": summary["critical_count"],
                "Ampel-Abdeckung": summary["assessment_coverage"],
                "Qualitäts-Score": safe_float(row.get("total_score")),
            }
            for metric in FUNDAMENTAL_OVERVIEW_METRICS:
                assessment = summary["assessments"][metric]
                output[FUNDAMENTAL_METRIC_LABELS[metric]] = (
                    f"{assessment['icon']} {format_fundamental_metric_value(metric, row.get(metric))}"
                )
            rows.append(output)

        overview = pd.DataFrame(rows)
        if overview.empty:
            st.info("Für den gewählten Filter gibt es keine Unternehmen.")
        else:
            sort_map = {
                "Unternehmen": ("Unternehmen", True),
                "Kritische Punkte": ("🔴", False),
                "Grüne Punkte": ("🟢", False),
                "Datenabdeckung": ("Ampel-Abdeckung", False),
                "Qualitäts-Score": ("Qualitäts-Score", False),
            }
            sort_col, ascending = sort_map[sort_choice]
            overview = overview.sort_values(sort_col, ascending=ascending, na_position="last", kind="stable")
            styled = overview.style.format({
                "Ampel-Abdeckung": lambda value: format_percent(value, 0),
                "Qualitäts-Score": lambda value: format_number(value, 1),
            }, na_rep="–").map(colorize_score, subset=["Qualitäts-Score"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption(
                "Die kompakte Ansicht zeigt nur zentrale Kennzahlen. Öffne „Einzelwert erklären“, um Schwelle, "
                "Begründung und nicht anwendbare Kennzahlen vollständig zu sehen."
            )
        return

    if view_mode == "Einzelwert erklären":
        ticker = company_selectbox("Aktie auswählen", assessed_df, key="fundamental_detail_ticker")
        row = assessed_df[assessed_df["ticker_yahoo"].astype(str) == str(ticker)].iloc[0]
        summary = build_fundamental_assessment(row)

        st.caption(f"{row.get('name', ticker)} · {ticker} · {row.get('sector', '–')}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Gesamtampel", summary["overall_label"])
        col2.metric("🟢 Unauffällig", summary["green_count"])
        col3.metric("🟡 Prüfen", summary["watch_count"])
        col4.metric("🔴 Kritisch", summary["critical_count"])
        col5.metric("Ampel-Abdeckung", format_percent(summary["assessment_coverage"], 0))

        selected_detail_statuses = st.multiselect(
            "Kennzahlen anzeigen",
            ["good", "watch", "critical", "info", "missing"],
            default=["good", "watch", "critical", "info", "missing"],
            format_func=lambda status: f"{FUNDAMENTAL_STATUS_META[status]['icon']} {FUNDAMENTAL_STATUS_META[status]['label']}",
            key="fundamental_detail_status_filter",
        )

        detail_rows: list[dict[str, Any]] = []
        for group, metrics in FUNDAMENTAL_METRIC_GROUPS.items():
            for metric in metrics:
                assessment = summary["assessments"][metric]
                if selected_detail_statuses and assessment["status"] not in selected_detail_statuses:
                    continue
                detail_rows.append({
                    "Bereich": group,
                    "Kennzahl": FUNDAMENTAL_METRIC_LABELS[metric],
                    "Wert": format_fundamental_metric_value(metric, row.get(metric)),
                    "Ampel": f"{assessment['icon']} {assessment['label']}",
                    "Einordnung": assessment["explanation"],
                    "Verwendete Regel": assessment["criterion"],
                })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        red_items = [
            FUNDAMENTAL_METRIC_LABELS[metric]
            for metric, item in summary["assessments"].items() if item["status"] == "critical"
        ]
        yellow_items = [
            FUNDAMENTAL_METRIC_LABELS[metric]
            for metric, item in summary["assessments"].items() if item["status"] == "watch"
        ]
        if red_items:
            st.error("Kritisch zu prüfen: " + ", ".join(red_items))
        elif yellow_items:
            st.warning("Vertiefen: " + ", ".join(yellow_items))
        else:
            st.success("In den verfügbaren, anwendbaren Kennzahlen wurden keine roten Punkte erkannt.")
        st.caption(
            "Die Ampel verwendet aktuelle Yahoo-Daten und Momentaufnahmen. Ein einzelnes Jahr, zyklische Hochphasen, "
            "Sondereffekte oder abweichende Rechnungslegung können die Einordnung verzerren."
        )
        return

    _render_raw_fundamentals_table(clean_df)
    st.caption(
        "Die Rohdaten zeigen die unveränderten Kennzahlen. Negative KGVs werden nicht als günstig bewertet. "
        "Marktkapitalisierung und Cashflows beziehen sich auf die jeweilige Originalwährung."
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
        ticker = company_selectbox("Trigger für einen Titel prüfen", df, key="value_trigger_explanation_ticker")
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
    ticker = company_selectbox("Titel prüfen", sorted_df, key="special_situation_detail_ticker")
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
    ticker = company_selectbox("Aktie auswählen", df, key="analysis_ticker")
    row = df.loc[df["ticker_yahoo"] == ticker].iloc[0]
    company_name = str(row.get("name") or ticker)
    company_sector = str(row.get("sector") or "Unbekannt")
    st.caption(f"{company_name} · {ticker} · {company_sector}")
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
                st.success(f"{company_name} ({ticker}) wurde gespeichert.")
            else:
                st.info(f"{company_name} ({ticker}) ist bereits auf der Watchlist.")

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



def _extract_sector_from_chart_event(event: Any, selection_name: str) -> Optional[str]:
    """Liest eine Sektor-Auswahl robust aus einem Streamlit-/Altair-Event."""
    if event is None:
        return None

    try:
        selection = event.selection
    except Exception:
        try:
            selection = event.get("selection", {})
        except Exception:
            return None

    try:
        payload = selection.get(selection_name, [])
    except Exception:
        try:
            payload = selection[selection_name]
        except Exception:
            return None

    if isinstance(payload, list) and payload:
        candidate = payload[-1]
        if isinstance(candidate, dict):
            value = candidate.get("Sektor") or candidate.get("sector")
            return str(value) if value not in (None, "") else None

    if isinstance(payload, dict):
        value = payload.get("Sektor") or payload.get("sector")
        if isinstance(value, list) and value:
            value = value[-1]
        return str(value) if value not in (None, "") else None

    return None


def _navigate_to_company(page: str, ticker: str) -> None:
    """Wechselt per Widget-Callback sicher auf eine firmenspezifische Ansicht."""
    ticker = str(ticker)
    if page == "Einzelanalyse":
        st.session_state["analysis_ticker"] = ticker
    elif page == "News & Events":
        st.session_state["news_ticker"] = ticker
    st.session_state["main_navigation"] = page


def _render_sector_company_drilldown(df: pd.DataFrame, selected_sector: str) -> None:
    """Zeigt die Einzelwerte eines Sektors und bietet gezielte Navigation an."""
    sector_mask = df["sector"].fillna("Unbekannt").astype(str).eq(str(selected_sector))
    sector_df = df.loc[sector_mask].copy()
    if sector_df.empty:
        st.info("Für diesen Sektor sind aktuell keine Aktien verfügbar.")
        return

    st.markdown(f"### Aktien im Sektor: {selected_sector}")
    st.caption(
        "Der Sektor bleibt als Kontext sichtbar. Wähle anschließend eine Aktie und öffne gezielt "
        "die Einzelanalyse oder den News-Bereich."
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Unternehmen", len(sector_df))
    metric_cols[1].metric("Ø Qualitäts-Score", format_number(sector_df.get("total_score", pd.Series(dtype=float)).mean(), 1))
    metric_cols[2].metric("Ø Value-Score", format_number(sector_df.get("value_score", pd.Series(dtype=float)).mean(), 1))
    metric_cols[3].metric("Ø Kursrendite 1J", format_percent(sector_df.get("change_1y", pd.Series(dtype=float)).mean(), 1, signed=True))
    metric_cols[4].metric("Ø Dividendenrendite", format_percent(sector_df.get("dividend_yield", pd.Series(dtype=float)).mean(), 2))

    columns = [
        "name", "ticker_yahoo", "currency", "last_price", "change_1y", "total_return_1y",
        "dividend_yield", "total_score", "value_score", "value_trigger",
        "special_situation_score", "vol_1y", "max_drawdown_1y",
    ]
    available_columns = [column for column in columns if column in sector_df.columns]
    detail = sector_df[available_columns].copy()
    sort_columns = [column for column in ["value_score", "total_score"] if column in detail.columns]
    if sort_columns:
        detail = detail.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")

    display = detail.rename(columns={
        "name": "Unternehmen",
        "ticker_yahoo": "Ticker",
        "currency": "Währung",
        "last_price": "Letzter Kurs",
        "change_1y": "Kursrendite 1J",
        "total_return_1y": "Gesamtrendite 1J*",
        "dividend_yield": "Dividendenrendite",
        "total_score": "Qualitäts-Score",
        "value_score": "Value-Score",
        "value_trigger": "Value-Trigger",
        "special_situation_score": "Deep-Value-Score",
        "vol_1y": "Volatilität 1J",
        "max_drawdown_1y": "Max. Drawdown 1J",
    })

    formatters = {
        "Letzter Kurs": lambda value: format_number(value, 2),
        "Kursrendite 1J": lambda value: format_percent(value, 1, signed=True),
        "Gesamtrendite 1J*": lambda value: format_percent(value, 1, signed=True),
        "Dividendenrendite": lambda value: format_percent(value, 2),
        "Qualitäts-Score": lambda value: format_number(value, 1),
        "Value-Score": lambda value: format_number(value, 1),
        "Value-Trigger": format_value_trigger,
        "Deep-Value-Score": lambda value: format_number(value, 1),
        "Volatilität 1J": lambda value: format_percent(value, 1),
        "Max. Drawdown 1J": lambda value: format_percent(value, 1, signed=True),
    }
    styled = display.style.format(
        {key: value for key, value in formatters.items() if key in display.columns},
        na_rep="–",
    )
    if "Kursrendite 1J" in display.columns:
        styled = styled.map(colorize_change, subset=["Kursrendite 1J"])
    if "Gesamtrendite 1J*" in display.columns:
        styled = styled.map(colorize_change, subset=["Gesamtrendite 1J*"])
    score_columns = [column for column in ["Qualitäts-Score", "Deep-Value-Score"] if column in display.columns]
    if score_columns:
        styled = styled.map(colorize_score, subset=score_columns)
    if "Value-Trigger" in display.columns:
        styled = styled.map(colorize_value_trigger, subset=["Value-Trigger"])

    st.dataframe(styled, use_container_width=True, hide_index=True)

    selected_ticker = company_selectbox(
        "Aktie aus diesem Sektor auswählen",
        sector_df,
        key="sector_company_ticker",
    )
    selected_row = sector_df.loc[sector_df["ticker_yahoo"].astype(str).eq(str(selected_ticker))].iloc[0]
    selected_name = str(selected_row.get("name") or selected_ticker)

    action_cols = st.columns([1, 1, 3])
    action_cols[0].button(
        "Einzelanalyse öffnen",
        key=f"sector_open_analysis_{selected_ticker}",
        on_click=_navigate_to_company,
        args=("Einzelanalyse", selected_ticker),
        use_container_width=True,
    )
    action_cols[1].button(
        "News öffnen",
        key=f"sector_open_news_{selected_ticker}",
        on_click=_navigate_to_company,
        args=("News & Events", selected_ticker),
        use_container_width=True,
    )
    action_cols[2].caption(f"Ausgewählt: {selected_name} ({selected_ticker})")


def render_sector_view(df: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> None:
    st.subheader("Sektor-Übersicht 2.1")
    st.caption(
        "Vergleicht Sektoren über mehrere Zeiträume. Klicke im Diagramm auf einen Sektor "
        "oder wähle ihn darunter manuell aus."
    )
    if df is None or df.empty:
        st.info("Keine Daten für die Sektoransicht vorhanden.")
        return

    sector_stats = sector_timeframe_stats(df, histories)
    if sector_stats.empty:
        st.info("Für die geladenen Werte liegen keine ausreichenden Kursdaten vor.")
        return

    available_sectors = sorted(
        df["sector"].fillna("Unbekannt").astype(str).drop_duplicates().tolist(),
        key=str.casefold,
    )
    if not available_sectors:
        st.info("Keine Sektoren verfügbar.")
        return

    metric_mode = st.radio(
        "Kennzahl anzeigen",
        ["Kursrendite", "Gesamtrendite", "Bewertung & Risiko"],
        horizontal=True,
        key="sector_view_metric_mode_v2",
    )

    clicked_sector: Optional[str] = None

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
        selection_name = f"sector_bar_selection_{prefix.lower()}"
        selector = alt.selection_point(
            name=selection_name,
            fields=["Sektor"],
            on="click",
            clear="dblclick",
        )
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(f"{chart_period}:Q", title=f"{metric_mode} {chart_period} in %"),
                y=alt.Y("Sektor:N", sort="-x", title="Sektor"),
                opacity=alt.condition(selector, alt.value(1.0), alt.value(0.55)),
                tooltip=["Sektor:N", "Unternehmen:Q", alt.Tooltip(f"{chart_period}:Q", format=".2f")],
            )
            .add_params(selector)
            .properties(height=max(260, 34 * len(chart_data)))
        )
        try:
            event = st.altair_chart(
                chart,
                use_container_width=True,
                key=f"sector_bar_chart_{prefix.lower()}",
                on_select="rerun",
                selection_mode=selection_name,
            )
            clicked_sector = _extract_sector_from_chart_event(event, selection_name)
        except TypeError:
            # Kompatibilitäts-Fallback für ältere Streamlit-Versionen.
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

        scatter_data = display.dropna(subset=["Value-Score", "Volatilität 1J"]).copy()
        selection_name = "sector_scatter_selection"
        selector = alt.selection_point(
            name=selection_name,
            fields=["Sektor"],
            on="click",
            clear="dblclick",
        )
        scatter = (
            alt.Chart(scatter_data)
            .mark_circle(size=120)
            .encode(
                x=alt.X("Volatilität 1J:Q", title="Volatilität 1J in %"),
                y=alt.Y("Value-Score:Q", title="Value-Score"),
                size=alt.Size("Unternehmen:Q", title="Unternehmen"),
                opacity=alt.condition(selector, alt.value(1.0), alt.value(0.55)),
                tooltip=[
                    "Sektor:N",
                    "Unternehmen:Q",
                    alt.Tooltip("Value-Score:Q", format=".1f"),
                    alt.Tooltip("Volatilität 1J:Q", format=".1f"),
                ],
            )
            .add_params(selector)
            .properties(height=420)
        )
        try:
            event = st.altair_chart(
                scatter,
                use_container_width=True,
                key="sector_scatter_chart_interactive",
                on_select="rerun",
                selection_mode=selection_name,
            )
            clicked_sector = _extract_sector_from_chart_event(event, selection_name)
        except TypeError:
            st.altair_chart(scatter, use_container_width=True)

    if clicked_sector in available_sectors:
        st.session_state["sector_drilldown_selected"] = clicked_sector

    if st.session_state.get("sector_drilldown_selected") not in available_sectors:
        st.session_state["sector_drilldown_selected"] = available_sectors[0]

    st.divider()
    selected_sector = st.selectbox(
        "Sektor für Detailansicht",
        options=available_sectors,
        key="sector_drilldown_selected",
        help="Ein Klick im Diagramm setzt diese Auswahl automatisch. Die manuelle Auswahl dient als zuverlässiger Fallback.",
    )
    _render_sector_company_drilldown(df, selected_sector)

    with st.expander("Was sagt diese Ansicht aus?"):
        st.write(
            "Die Sektoransicht zeigt, ob ein Scanner-Kandidat nur wegen eines einzelnen Unternehmens auffällt "
            "oder ob ein kompletter Sektor unter Druck steht. Besonders nützlich ist der Vergleich aus 1J/3J/5J-Rendite, "
            "Value-Score, Dividendenrendite und Volatilität. Ein hoher Value-Score bei gleichzeitig hohem Drawdown kann "
            "eine Chance oder eine Value Trap sein – deshalb immer mit News, Cashflow und Verschuldung gegenprüfen."
        )
        st.write(
            "Die Detailansicht springt bewusst nicht sofort aus dem Sektorbereich heraus: Erst wird der Sektor aufgeklappt, "
            "danach entscheidest du ausdrücklich, ob du eine Aktie in der Einzelanalyse oder im News-Bereich öffnen möchtest."
        )


def _sentiment_badge_html(label: Any) -> str:
    normalized = str(label or "neutral").lower()
    styles = {
        "positiv": ("Positiv", "#166534", "#dcfce7", "#86efac"),
        "negativ": ("Negativ", "#991b1b", "#fee2e2", "#fca5a5"),
        "gemischt": ("Gemischt", "#854d0e", "#fef9c3", "#fde047"),
        "neutral": ("Neutral", "#475569", "#f1f5f9", "#cbd5e1"),
    }
    text_label, color, background, border = styles.get(normalized, styles["neutral"])
    return (
        f'<span style="display:inline-block;padding:0.18rem 0.55rem;border-radius:999px;'
        f'font-weight:700;font-size:0.8rem;color:{color};background:{background};'
        f'border:1px solid {border};">{text_label}</span>'
    )


def render_news_card(item: pd.Series, card_index: int) -> None:
    """Kompakte News-Karte mit klarer Sentimentfarbe und Begründung."""
    published = pd.to_datetime(item.get("published"), errors="coerce")
    date_label = published.strftime("%d.%m.%Y, %H:%M") if pd.notna(published) else "Datum unbekannt"
    event_label = EVENT_META.get(str(item.get("event_type", "news")), {}).get("label", "News")
    sentiment_label = str(item.get("sentiment_label") or "neutral").lower()
    relevance = str(item.get("relevance_label") or "–").capitalize()
    confidence = str(item.get("sentiment_confidence") or "niedrig").capitalize()

    with st.container(border=True):
        header_left, header_right = st.columns([5, 1])
        with header_left:
            st.markdown(f"**{str(item.get('title') or 'Ohne Titel')}**")
        with header_right:
            st.markdown(_sentiment_badge_html(sentiment_label), unsafe_allow_html=True)

        st.caption(
            f"{event_label} · Relevanz: {relevance} · Sentiment-Sicherheit: {confidence} · "
            f"{str(item.get('source') or 'Quelle unbekannt')} · {date_label}"
        )
        sentiment_reason = str(item.get("sentiment_reason") or "").strip()
        if sentiment_reason:
            st.caption(f"Sentiment-Begründung: {sentiment_reason}")
        relevance_reason = str(item.get("relevance_reason") or "").strip()
        if relevance_reason:
            st.caption(f"Firmenzuordnung: {relevance_reason}")
        link = str(item.get("link") or "").strip()
        if link:
            try:
                st.link_button(
                    "Artikel öffnen",
                    link,
                    key=f"news_link_{card_index}_{normalize_for_search(str(item.get('title', '')))}",
                )
            except TypeError:
                st.link_button("Artikel öffnen", link)


def _sentiment_filter_state_key(ticker: str) -> str:
    """Eindeutiger Session-State-Key für den klickbaren News-Sentimentfilter."""
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]+", "_", str(ticker))
    return f"news_sentiment_filter::{safe_ticker}"


def render_clickable_sentiment_filter(relevant_news: pd.DataFrame, ticker: str) -> str:
    """
    Zeigt klickbare Sentiment-Zähler und gibt den aktiven Filter zurück.

    ``st.metric`` ist nicht klickbar. Deshalb werden bewusst breite Buttons
    verwendet, die optisch als kompakte Filterkarten funktionieren.
    """
    labels = (
        relevant_news.get("sentiment_label", pd.Series("neutral", index=relevant_news.index))
        .fillna("neutral")
        .astype(str)
        .str.lower()
    )
    counts = {
        "alle": int(len(relevant_news)),
        "positiv": int((labels == "positiv").sum()),
        "negativ": int((labels == "negativ").sum()),
        "gemischt": int((labels == "gemischt").sum()),
        "neutral": int((labels == "neutral").sum()),
    }
    options = [
        ("alle", "Alle", "📰"),
        ("positiv", "Positiv", "🟢"),
        ("negativ", "Negativ", "🔴"),
        ("gemischt", "Gemischt", "🟡"),
        ("neutral", "Neutral", "⚪"),
    ]

    state_key = _sentiment_filter_state_key(ticker)
    if st.session_state.get(state_key) not in counts:
        st.session_state[state_key] = "alle"

    active_filter = st.session_state[state_key]
    columns = st.columns(len(options))
    for column, (value, title, icon) in zip(columns, options):
        active_marker = "✓ " if active_filter == value else ""
        label = f"{active_marker}{icon} {title} · {counts[value]}"
        if column.button(
            label,
            key=f"sentiment_filter_button::{ticker}::{value}",
            use_container_width=True,
            help=f"Nur {title.lower()}e Meldungen anzeigen" if value != "alle" else "Alle relevanten Meldungen anzeigen",
        ):
            st.session_state[state_key] = value
            st.rerun()

    active_filter = st.session_state[state_key]
    active_title = next(title for value, title, _ in options if value == active_filter)
    st.caption(
        f"Aktiver News-Filter: **{active_title}**. "
        "Der Klick wirkt auf den Tab „Aktuelle News“; der Kalender besitzt eigene Filter."
    )
    return active_filter


def render_news_summary(
    ticker: str,
    company_name: str,
    relevant_news: pd.DataFrame,
    uncertain_news: pd.DataFrame,
    events: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    """Zeigt eine sachliche Einordnung und gibt den aktiven Sentimentfilter zurück."""
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

    active_sentiment_filter = "alle"
    if not relevant_news.empty:
        active_sentiment_filter = render_clickable_sentiment_filter(relevant_news, ticker)

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

    return active_sentiment_filter



# -----------------------------------------------------------------------------
# Version 5.6 – Fundamentaldaten-Ampeln mit sektorabhängiger Einordnung
# -----------------------------------------------------------------------------

EVENT_COLUMNS_V55 = [
    "date", "ticker_yahoo", "event_type", "title", "source", "link",
    "sentiment_score", "sentiment_label", "sentiment_confidence",
    "sentiment_reason", "importance", "is_future_event",
    "verification_level", "verification_score", "event_status",
    "source_type", "source_url", "retrieved_at", "notes", "conflict_note",
]

IR_SOURCE_COLUMNS = [
    "ticker_yahoo", "source_name", "source_type", "feed_url", "source_url",
    "verification_level", "enabled", "notes",
]

MANUAL_EVENT_COLUMNS = [
    "date", "ticker_yahoo", "event_type", "title", "source", "link",
    "importance", "event_status", "verification_level", "notes",
]

EVENT_META.update({
    "capital_markets_day": {"label": "Kapitalmarkttag", "shape": "triangle-right"},
    "conference": {"label": "Investorenkonferenz", "shape": "triangle-left"},
})

VERIFICATION_META: dict[str, dict[str, Any]] = {
    "official": {"label": "🟢 Offiziell", "score": 100, "status": "bestätigt"},
    "official_ir": {"label": "🟢 Unternehmens-IR", "score": 95, "status": "bestätigt"},
    "manual": {"label": "🔵 Manuell bestätigt", "score": 85, "status": "bestätigt"},
    "provider": {"label": "🟡 Datenanbieter", "score": 60, "status": "Anbieterangabe"},
    "derived": {"label": "⚪ Aus News abgeleitet", "score": 40, "status": "abgeleitet"},
    "unknown": {"label": "⚪ Unbekannt", "score": 20, "status": "prüfen"},
}


def verification_label(value: Any) -> str:
    level = str(value or "unknown").strip().lower()
    return str(VERIFICATION_META.get(level, VERIFICATION_META["unknown"])["label"])


def verification_score(value: Any) -> int:
    level = str(value or "unknown").strip().lower()
    return int(VERIFICATION_META.get(level, VERIFICATION_META["unknown"])["score"])


def verification_cell_style(value: Any) -> str:
    normalized = str(value or "").lower()
    if "offiziell" in normalized or "unternehmens-ir" in normalized:
        return "color: #166534; background-color: #dcfce7; font-weight: 700"
    if "manuell" in normalized:
        return "color: #1e3a8a; background-color: #dbeafe; font-weight: 700"
    if "datenanbieter" in normalized:
        return "color: #854d0e; background-color: #fef3c7; font-weight: 700"
    return "color: #475569; background-color: #f1f5f9"


def event_status_cell_style(value: Any) -> str:
    normalized = str(value or "").lower()
    if "bestätigt" in normalized:
        return "color: #166534; background-color: #dcfce7; font-weight: 700"
    if "prüfen" in normalized or "konflikt" in normalized:
        return "color: #991b1b; background-color: #fee2e2; font-weight: 700"
    if "anbieter" in normalized:
        return "color: #854d0e; background-color: #fef3c7; font-weight: 700"
    return "color: #475569; background-color: #f1f5f9"


def empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS_V55)


def normalize_events_frame(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_events_frame()
    result = frame.copy()
    defaults: dict[str, Any] = {
        "date": pd.NaT,
        "ticker_yahoo": "",
        "event_type": "news",
        "title": "",
        "source": "",
        "link": "",
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "sentiment_confidence": "niedrig",
        "sentiment_reason": "",
        "importance": "mittel",
        "is_future_event": False,
        "verification_level": "unknown",
        "verification_score": 20,
        "event_status": "prüfen",
        "source_type": "unknown",
        "source_url": "",
        "retrieved_at": "",
        "notes": "",
        "conflict_note": "",
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).copy()
    if not result.empty:
        try:
            if getattr(result["date"].dt, "tz", None) is not None:
                result["date"] = result["date"].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    result["ticker_yahoo"] = result["ticker_yahoo"].map(clean_ticker)
    result["verification_level"] = result["verification_level"].fillna("unknown").astype(str).str.lower()
    result["verification_score"] = pd.to_numeric(result["verification_score"], errors="coerce")
    missing_score = result["verification_score"].isna()
    if missing_score.any():
        result.loc[missing_score, "verification_score"] = result.loc[missing_score, "verification_level"].map(verification_score)
    result["verification_score"] = result["verification_score"].fillna(20).astype(int)
    result["is_future_event"] = result["is_future_event"].astype(str).str.lower().isin({"true", "1", "yes", "ja"}) | (
        result["date"].dt.normalize() >= pd.Timestamp.now().normalize()
    )
    for column in ["title", "source", "link", "sentiment_label", "sentiment_confidence", "sentiment_reason",
                   "importance", "event_status", "source_type", "source_url", "retrieved_at", "notes", "conflict_note"]:
        result[column] = result[column].fillna("").astype(str)
    return result[EVENT_COLUMNS_V55].reset_index(drop=True)


def ensure_event_source_files() -> None:
    if not IR_SOURCES_PATH.exists():
        safe_write_csv(pd.DataFrame(columns=IR_SOURCE_COLUMNS), IR_SOURCES_PATH)
    if not MANUAL_EVENTS_PATH.exists():
        safe_write_csv(pd.DataFrame(columns=MANUAL_EVENT_COLUMNS), MANUAL_EVENTS_PATH)


def read_ir_sources() -> pd.DataFrame:
    ensure_event_source_files()
    try:
        frame = pd.read_csv(IR_SOURCES_PATH, dtype=str)
    except Exception:
        frame = pd.DataFrame(columns=IR_SOURCE_COLUMNS)
    for column in IR_SOURCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame["ticker_yahoo"] = frame["ticker_yahoo"].map(clean_ticker)
    frame["source_type"] = frame["source_type"].fillna("rss").astype(str).str.lower()
    frame["verification_level"] = frame["verification_level"].fillna("official_ir").astype(str).str.lower()
    frame["enabled"] = frame["enabled"].fillna("true").astype(str).str.lower().isin({"true", "1", "yes", "ja", "x"})
    return frame[IR_SOURCE_COLUMNS].drop_duplicates(
        ["ticker_yahoo", "source_name", "feed_url"], keep="last"
    ).reset_index(drop=True)


def save_ir_sources(frame: pd.DataFrame) -> tuple[bool, str]:
    result = frame.copy()
    for column in IR_SOURCE_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    result["ticker_yahoo"] = result["ticker_yahoo"].map(clean_ticker)
    result["enabled"] = result["enabled"].astype(bool)
    return safe_write_csv(result[IR_SOURCE_COLUMNS], IR_SOURCES_PATH)


def read_manual_events(ticker: Optional[str] = None) -> pd.DataFrame:
    ensure_event_source_files()
    try:
        frame = pd.read_csv(MANUAL_EVENTS_PATH, dtype=str)
    except Exception:
        frame = pd.DataFrame(columns=MANUAL_EVENT_COLUMNS)
    for column in MANUAL_EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame["ticker_yahoo"] = frame["ticker_yahoo"].map(clean_ticker)
    if ticker:
        frame = frame[frame["ticker_yahoo"] == clean_ticker(ticker)].copy()
    if frame.empty:
        return empty_events_frame()
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M")
    result = pd.DataFrame({
        "date": pd.to_datetime(frame["date"], errors="coerce"),
        "ticker_yahoo": frame["ticker_yahoo"],
        "event_type": frame["event_type"].replace("", "report"),
        "title": frame["title"],
        "source": frame["source"].replace("", "Manuell bestätigt"),
        "link": frame["link"],
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "sentiment_confidence": "niedrig",
        "sentiment_reason": "Manueller Kalendereintrag – keine automatische Sentimentbewertung",
        "importance": frame["importance"].replace("", "mittel"),
        "is_future_event": False,
        "verification_level": frame["verification_level"].replace("", "manual"),
        "verification_score": frame["verification_level"].replace("", "manual").map(verification_score),
        "event_status": frame["event_status"].replace("", "bestätigt"),
        "source_type": "manual",
        "source_url": frame["link"],
        "retrieved_at": now_text,
        "notes": frame["notes"],
        "conflict_note": "",
    })
    return normalize_events_frame(result)


def append_manual_event(row: dict[str, Any]) -> tuple[bool, str]:
    ensure_event_source_files()
    try:
        existing = pd.read_csv(MANUAL_EVENTS_PATH, dtype=str)
    except Exception:
        existing = pd.DataFrame(columns=MANUAL_EVENT_COLUMNS)
    addition = pd.DataFrame([{column: row.get(column, "") for column in MANUAL_EVENT_COLUMNS}])
    combined = pd.concat([existing, addition], ignore_index=True)
    combined = combined.drop_duplicates(["date", "ticker_yahoo", "event_type", "title"], keep="last")
    return safe_write_csv(combined[MANUAL_EVENT_COLUMNS], MANUAL_EVENTS_PATH)


def delete_manual_event(ticker: str, date_value: str, event_type: str, title: str) -> tuple[bool, str]:
    ensure_event_source_files()
    try:
        frame = pd.read_csv(MANUAL_EVENTS_PATH, dtype=str)
    except Exception:
        frame = pd.DataFrame(columns=MANUAL_EVENT_COLUMNS)
    if frame.empty:
        return True, ""
    mask = (
        frame.get("ticker_yahoo", "").map(clean_ticker).eq(clean_ticker(ticker))
        & frame.get("date", "").astype(str).eq(str(date_value))
        & frame.get("event_type", "").astype(str).eq(str(event_type))
        & frame.get("title", "").astype(str).eq(str(title))
    )
    return safe_write_csv(frame.loc[~mask, MANUAL_EVENT_COLUMNS], MANUAL_EVENTS_PATH)


def _parse_ics_datetime(value: str) -> pd.Timestamp | pd.NaT:
    raw = str(value or "").strip()
    if not raw:
        return pd.NaT
    formats = ["%Y%m%d", "%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S", "%Y%m%dT%H%M"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(raw, fmt)
            return pd.Timestamp(parsed)
        except ValueError:
            continue
    return pd.to_datetime(raw, errors="coerce")


def parse_ics_payload(payload: bytes) -> list[dict[str, str]]:
    try:
        text_value = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text_value = payload.decode("latin-1", errors="replace")
    raw_lines = text_value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines: list[str] = []
    for line in raw_lines:
        if line.startswith((" ", "\t")) and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)
    events: list[dict[str, str]] = []
    current: Optional[dict[str, str]] = None
    for line in lines:
        if line.strip() == "BEGIN:VEVENT":
            current = {}
            continue
        if line.strip() == "END:VEVENT":
            if current is not None:
                events.append(current)
            current = None
            continue
        if current is None or ":" not in line:
            continue
        key_part, value = line.split(":", 1)
        key = key_part.split(";", 1)[0].strip().upper()
        current[key] = value.replace("\\n", " ").replace("\\,", ",").strip()
    return events


def classify_event_from_text_v55(text: str) -> str:
    normalized = normalize_for_search(text)
    if any(term in normalized for term in ["capital markets day", "capital market day", "investor day", "kapitalmarkttag"]):
        return "capital_markets_day"
    if any(term in normalized for term in ["investor conference", "investorenkonferenz", "conference presentation"]):
        return "conference"
    return classify_event_from_text(text)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_sec_company_map() -> tuple[dict[str, dict[str, Any]], str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers=SEC_HEADERS, timeout=25)
        response.raise_for_status()
        payload = response.json()
        mapping: dict[str, dict[str, Any]] = {}
        iterable = payload.values() if isinstance(payload, dict) else payload
        for item in iterable:
            if not isinstance(item, dict):
                continue
            ticker_value = clean_ticker(item.get("ticker"))
            cik = item.get("cik_str")
            if ticker_value and cik is not None:
                mapping[ticker_value] = {
                    "cik": int(cik),
                    "title": str(item.get("title", "")),
                }
        return mapping, ""
    except Exception as error:
        return {}, f"{type(error).__name__}: {error}"


def _sec_form_meta(form: str) -> Optional[tuple[str, str, str]]:
    normalized = str(form or "").upper().strip()
    if normalized in {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}:
        return "report", f"Jahresbericht / {normalized}", "hoch"
    if normalized in {"10-Q", "10-Q/A"}:
        return "earnings", f"Quartalsbericht / {normalized}", "hoch"
    if normalized in {"8-K", "8-K/A", "6-K", "6-K/A"}:
        return "report", f"Unternehmensmeldung / {normalized}", "mittel"
    if normalized in {"DEF 14A", "DEFA14A", "PRE 14A"}:
        return "annual_meeting", f"Proxy Statement / {normalized}", "hoch"
    return None


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_sec_filings_events(
    ticker: str,
    company_name: str,
    days_back: int = 730,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_clean = clean_ticker(ticker)
    diagnostic = {
        "source": "SEC EDGAR",
        "kind": "official_regulator",
        "status": "Nicht verfügbar",
        "http_status": None,
        "entries": 0,
        "matches": 0,
        "uncertain_matches": 0,
        "duration_ms": None,
        "message": "",
        "url": "https://data.sec.gov/",
    }
    # Börsensuffixe wie .DE/.PA sind nicht direkt SEC-gemappt. US-ADRs ohne
    # Suffix können dagegen vorhanden sein.
    if "." in ticker_clean:
        diagnostic["message"] = "SEC-Quelle wird nur für SEC-gemappte US-Ticker/ADRs verwendet."
        return empty_events_frame(), pd.DataFrame([diagnostic])

    started = time.perf_counter()
    company_map, map_error = fetch_sec_company_map()
    if map_error:
        diagnostic["status"] = "Fehler"
        diagnostic["message"] = map_error
        diagnostic["duration_ms"] = round((time.perf_counter() - started) * 1000)
        return empty_events_frame(), pd.DataFrame([diagnostic])
    match = company_map.get(ticker_clean)
    if not match:
        diagnostic["message"] = f"Kein SEC-CIK für {ticker_clean} gefunden."
        diagnostic["duration_ms"] = round((time.perf_counter() - started) * 1000)
        return empty_events_frame(), pd.DataFrame([diagnostic])

    cik = int(match["cik"])
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    diagnostic["url"] = url
    try:
        response = requests.get(url, headers=SEC_HEADERS, timeout=25)
        diagnostic["http_status"] = response.status_code
        response.raise_for_status()
        payload = response.json()
        recent = payload.get("filings", {}).get("recent", {})
        forms = list(recent.get("form", []))
        dates = list(recent.get("filingDate", []))
        accessions = list(recent.get("accessionNumber", []))
        docs = list(recent.get("primaryDocument", []))
        descriptions = list(recent.get("primaryDocDescription", []))
        diagnostic["entries"] = len(forms)
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(days_back))
        rows: list[dict[str, Any]] = []
        for position, form in enumerate(forms):
            filing_date = dates[position] if position < len(dates) else None
            accession = accessions[position] if position < len(accessions) else ""
            document = docs[position] if position < len(docs) else ""
            description = descriptions[position] if position < len(descriptions) else ""
            meta = _sec_form_meta(str(form))
            if meta is None:
                continue
            parsed_date = pd.to_datetime(filing_date, errors="coerce")
            if pd.isna(parsed_date) or pd.Timestamp(parsed_date).normalize() < cutoff:
                continue
            event_type, base_title, importance = meta
            accession_compact = str(accession).replace("-", "")
            filing_link = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_compact}/{document}"
                if accession_compact and document else url
            )
            detail = str(description or "").strip()
            title = base_title if not detail else f"{base_title}: {detail}"
            rows.append({
                "date": pd.Timestamp(parsed_date).tz_localize(None),
                "ticker_yahoo": ticker_clean,
                "event_type": event_type,
                "title": title,
                "source": "SEC EDGAR",
                "link": filing_link,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "sentiment_confidence": "niedrig",
                "sentiment_reason": "Offizielles Filing – keine automatische inhaltliche Bewertung",
                "importance": importance,
                "is_future_event": False,
                "verification_level": "official",
                "verification_score": 100,
                "event_status": "bestätigt",
                "source_type": "regulator",
                "source_url": url,
                "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "notes": f"SEC-CIK {cik:010d} · {company_name}",
                "conflict_note": "",
            })
        events = normalize_events_frame(pd.DataFrame(rows))
        diagnostic["matches"] = len(events)
        diagnostic["status"] = "OK" if not events.empty else "Keine relevanten Filings"
        diagnostic["duration_ms"] = round((time.perf_counter() - started) * 1000)
        return events, pd.DataFrame([diagnostic])
    except Exception as error:
        diagnostic["status"] = "Fehler"
        diagnostic["message"] = f"{type(error).__name__}: {error}"
        diagnostic["duration_ms"] = round((time.perf_counter() - started) * 1000)
        return empty_events_frame(), pd.DataFrame([diagnostic])


def _ir_feed_to_events(
    source_row: pd.Series,
    ticker: str,
    earliest: pd.Timestamp,
    latest: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_name = str(source_row.get("source_name", "Unternehmens-IR") or "Unternehmens-IR")
    source_type = str(source_row.get("source_type", "rss") or "rss").lower()
    feed_url = str(source_row.get("feed_url", "") or "").strip()
    source_url = str(source_row.get("source_url", "") or "").strip()
    verification = str(source_row.get("verification_level", "official_ir") or "official_ir").lower()
    diagnostic: dict[str, Any] = {
        "source": source_name,
        "kind": f"ir_{source_type}",
        "status": "Nicht konfiguriert",
        "http_status": None,
        "entries": 0,
        "matches": 0,
        "uncertain_matches": 0,
        "duration_ms": None,
        "message": "",
        "url": feed_url or source_url,
    }
    if source_type == "web" or not feed_url:
        diagnostic["status"] = "Nur Link" if source_url else "Nicht konfiguriert"
        diagnostic["message"] = "Diese Quelle dient als offizieller Referenzlink und wird nicht automatisch ausgelesen."
        return empty_events_frame(), diagnostic

    payload, request_diag = request_feed(feed_url)
    diagnostic.update({
        "status": request_diag.get("status"),
        "http_status": request_diag.get("http_status"),
        "duration_ms": request_diag.get("duration_ms"),
        "message": request_diag.get("message", ""),
    })
    if payload is None:
        return empty_events_frame(), diagnostic

    rows: list[dict[str, Any]] = []
    if source_type == "ics":
        parsed = parse_ics_payload(payload)
        diagnostic["entries"] = len(parsed)
        for item in parsed:
            date_value = _parse_ics_datetime(item.get("DTSTART", ""))
            if pd.isna(date_value) or not (earliest <= pd.Timestamp(date_value) <= latest):
                continue
            title = str(item.get("SUMMARY", "Termin") or "Termin")
            description = str(item.get("DESCRIPTION", "") or "")
            link = str(item.get("URL", "") or source_url or feed_url)
            event_type = classify_event_from_text_v55(f"{title} {description}")
            rows.append({
                "date": pd.Timestamp(date_value).tz_localize(None),
                "ticker_yahoo": ticker,
                "event_type": event_type,
                "title": title,
                "source": source_name,
                "link": link,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "sentiment_confidence": "niedrig",
                "sentiment_reason": "Offizieller Kalendertermin – keine automatische inhaltliche Bewertung",
                "importance": "hoch" if event_type in {"earnings", "annual_meeting", "report"} else "mittel",
                "is_future_event": pd.Timestamp(date_value).normalize() >= pd.Timestamp.now().normalize(),
                "verification_level": verification,
                "verification_score": verification_score(verification),
                "event_status": VERIFICATION_META.get(verification, VERIFICATION_META["unknown"])["status"],
                "source_type": "company_calendar",
                "source_url": source_url or feed_url,
                "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "notes": description,
                "conflict_note": "",
            })
    else:
        parsed_entries, parser_error = parse_feed_entries(payload, source_name, "official_ir")
        diagnostic["entries"] = len(parsed_entries)
        if parser_error:
            diagnostic["status"] = "Parser-Fehler"
            diagnostic["message"] = parser_error
            return empty_events_frame(), diagnostic
        for entry in parsed_entries:
            date_value = pd.Timestamp(entry["published"])
            if not (earliest <= date_value <= latest):
                continue
            combined = f"{entry['title']} {entry['summary']}"
            event_type = classify_event_from_text_v55(combined)
            # Allgemeine IR-News bleiben im News-Teil; der Kalender übernimmt
            # nur klar klassifizierte Ereignisse.
            if event_type == "news":
                continue
            sentiment = analyze_sentiment(entry["title"], entry["summary"])
            rows.append({
                "date": date_value,
                "ticker_yahoo": ticker,
                "event_type": event_type,
                "title": entry["title"],
                "source": source_name,
                "link": entry["link"] or source_url,
                "sentiment_score": sentiment["score"],
                "sentiment_label": sentiment["label"],
                "sentiment_confidence": sentiment["confidence"],
                "sentiment_reason": sentiment["reason"],
                "importance": "hoch" if event_type in {"earnings", "annual_meeting", "report"} else "mittel",
                "is_future_event": date_value.normalize() >= pd.Timestamp.now().normalize(),
                "verification_level": verification,
                "verification_score": verification_score(verification),
                "event_status": VERIFICATION_META.get(verification, VERIFICATION_META["unknown"])["status"],
                "source_type": "company_feed",
                "source_url": source_url or feed_url,
                "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "notes": entry["summary"],
                "conflict_note": "",
            })
    events = normalize_events_frame(pd.DataFrame(rows))
    diagnostic["matches"] = len(events)
    diagnostic["status"] = "OK" if not events.empty else "Keine Kalenderereignisse"
    return events, diagnostic


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_registered_ir_events(
    ticker: str,
    days_back: int = 730,
    days_forward: int = 730,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = read_ir_sources()
    sources = sources[(sources["ticker_yahoo"] == clean_ticker(ticker)) & sources["enabled"]].copy()
    if sources.empty:
        diagnostic = pd.DataFrame([{
            "source": "Unternehmens-IR",
            "kind": "ir_registry",
            "status": "Nicht konfiguriert",
            "http_status": None,
            "entries": 0,
            "matches": 0,
            "uncertain_matches": 0,
            "duration_ms": None,
            "message": "Für diesen Ticker wurde noch kein offizieller RSS-/ICS-Link hinterlegt.",
            "url": "",
        }])
        return empty_events_frame(), diagnostic
    earliest = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(days_back))
    latest = pd.Timestamp.now().normalize() + pd.Timedelta(days=int(days_forward))
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for _, source_row in sources.iterrows():
        events, diagnostic = _ir_feed_to_events(source_row, clean_ticker(ticker), earliest, latest)
        frames.append(events)
        diagnostics.append(diagnostic)
    combined = normalize_events_frame(pd.concat(frames, ignore_index=True)) if frames else empty_events_frame()
    return combined, pd.DataFrame(diagnostics)


def detect_event_conflicts(events: pd.DataFrame, tolerance_days: int = 14) -> pd.DataFrame:
    result = normalize_events_frame(events)
    if result.empty:
        return result
    result["conflict_note"] = result["conflict_note"].fillna("").astype(str)
    future = result[result["date"].dt.normalize() >= pd.Timestamp.now().normalize()]
    for (ticker, event_type), group in future.groupby(["ticker_yahoo", "event_type"]):
        group = group.sort_values("date")
        indices = list(group.index)
        for left_pos, left_index in enumerate(indices):
            for right_index in indices[left_pos + 1:]:
                delta = abs((result.at[right_index, "date"].normalize() - result.at[left_index, "date"].normalize()).days)
                if delta > tolerance_days:
                    break
                if delta <= 1:
                    continue
                left_source = str(result.at[left_index, "source"])
                right_source = str(result.at[right_index, "source"])
                if left_source == right_source:
                    continue
                note = (
                    f"Abweichende Terminangaben innerhalb von {tolerance_days} Tagen: "
                    f"{result.at[left_index, 'date'].strftime('%d.%m.%Y')} ({left_source}) vs. "
                    f"{result.at[right_index, 'date'].strftime('%d.%m.%Y')} ({right_source})."
                )
                for index in [left_index, right_index]:
                    result.at[index, "event_status"] = "prüfen"
                    result.at[index, "conflict_note"] = note
    return result


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_yahoo_calendar_events(ticker: str, days_back: int = 365, days_forward: int = 365) -> pd.DataFrame:
    ticker = clean_ticker(ticker)
    now = datetime.now().replace(tzinfo=None)
    earliest = now - timedelta(days=int(days_back))
    latest = now + timedelta(days=int(days_forward))
    rows: list[dict[str, Any]] = []

    def append_provider_event(date_value: Any, event_type: str, title: str, importance: str = "mittel") -> None:
        parsed = pd.to_datetime(date_value, errors="coerce")
        if pd.isna(parsed):
            return
        event_date = pd.Timestamp(parsed).tz_localize(None).to_pydatetime()
        if not (earliest <= event_date <= latest):
            return
        rows.append({
            "date": event_date,
            "ticker_yahoo": ticker,
            "event_type": event_type,
            "title": title,
            "source": "Yahoo Finance",
            "link": "",
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "sentiment_confidence": "niedrig",
            "sentiment_reason": "Datenanbieter-Termin – keine automatische inhaltliche Bewertung",
            "importance": importance,
            "is_future_event": bool(event_date > now),
            "verification_level": "provider",
            "verification_score": 60,
            "event_status": "Anbieterangabe",
            "source_type": "market_data_provider",
            "source_url": "https://finance.yahoo.com/",
            "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "notes": "Termin kann sich ändern; nach Möglichkeit mit Unternehmens-IR abgleichen.",
            "conflict_note": "",
        })

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
            append_provider_event(
                dividend_row["date"],
                "dividend",
                f"Dividende: {format_number(float(dividend_row['amount']), 4)} je Aktie",
            )

    try:
        calendar = yf.Ticker(ticker).calendar
    except Exception:
        calendar = None

    def add_values(value: Any, event_type: str, title: str) -> None:
        values = list(value) if isinstance(value, (pd.Series, pd.Index, list, tuple, set)) else [value]
        for raw_date in values:
            append_provider_event(raw_date, event_type, title, "hoch" if event_type == "earnings" else "mittel")

    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
        for index_value, row in calendar.iterrows():
            lowered = str(index_value).lower()
            values = row.tolist()
            if "earning" in lowered:
                add_values(values, "earnings", "Quartalszahlen / Earnings")
            elif "ex-dividend" in lowered or "ex dividend" in lowered:
                add_values(values, "dividend", "Ex-Dividende")
    elif isinstance(calendar, dict):
        for key, value in calendar.items():
            lowered = str(key).lower()
            if "earning" in lowered:
                add_values(value, "earnings", "Quartalszahlen / Earnings")
            elif "ex-dividend" in lowered or "ex dividend" in lowered:
                add_values(value, "dividend", "Ex-Dividende")
    return normalize_events_frame(pd.DataFrame(rows))


def news_to_events(news: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if news is None or news.empty:
        return empty_events_frame()
    eligible = news.copy()
    if "is_relevant" in eligible.columns:
        eligible = eligible[eligible["is_relevant"].fillna(False)]
    if "event_type" in eligible.columns:
        eligible = eligible[eligible["event_type"].astype(str) != "news"]
    if eligible.empty:
        return empty_events_frame()
    result = pd.DataFrame({
        "date": pd.to_datetime(eligible["published"], errors="coerce"),
        "ticker_yahoo": clean_ticker(ticker),
        "event_type": eligible.get("event_type", "news"),
        "title": eligible.get("title", ""),
        "source": eligible.get("source", ""),
        "link": eligible.get("link", ""),
        "sentiment_score": eligible.get("sentiment_score", 0.0),
        "sentiment_label": eligible.get("sentiment_label", "neutral"),
        "sentiment_confidence": eligible.get("sentiment_confidence", "niedrig"),
        "sentiment_reason": eligible.get("sentiment_reason", ""),
        "importance": "mittel",
        "is_future_event": False,
        "verification_level": "derived",
        "verification_score": 40,
        "event_status": "abgeleitet",
        "source_type": "news_derived",
        "source_url": eligible.get("link", ""),
        "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes": eligible.get("relevance_reason", "Aus relevanter Nachricht abgeleitet"),
        "conflict_note": "",
    })
    return normalize_events_frame(result)


def combine_events(
    news: pd.DataFrame,
    calendar_events: pd.DataFrame,
    ticker: str,
    extra_events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frames = [news_to_events(news, ticker), normalize_events_frame(calendar_events)]
    if extra_events is not None:
        frames.append(normalize_events_frame(extra_events))
    combined = normalize_events_frame(pd.concat(frames, ignore_index=True))
    if combined.empty:
        return combined
    combined["_title_key"] = combined["title"].map(normalize_for_search)
    combined["_date_key"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined = (
        combined.sort_values(["verification_score", "retrieved_at"], ascending=[False, False])
        .drop_duplicates(["ticker_yahoo", "_date_key", "event_type", "_title_key"], keep="first")
        .drop(columns=["_title_key", "_date_key"])
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    return detect_event_conflicts(combined)


def persist_events(events: pd.DataFrame, replace_ticker: Optional[str] = None) -> tuple[bool, str]:
    events = normalize_events_frame(events)
    existing = empty_events_frame()
    if EVENTS_PATH.exists():
        try:
            existing = normalize_events_frame(pd.read_csv(EVENTS_PATH))
        except Exception:
            existing = empty_events_frame()
    if replace_ticker and not existing.empty:
        existing = existing[existing["ticker_yahoo"] != clean_ticker(replace_ticker)].copy()
    combined = normalize_events_frame(pd.concat([existing, events], ignore_index=True))
    if combined.empty:
        return safe_write_csv(pd.DataFrame(columns=EVENT_COLUMNS_V55), EVENTS_PATH)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    combined = combined.drop_duplicates(
        ["ticker_yahoo", "date", "event_type", "title", "source"], keep="last"
    ).sort_values(["ticker_yahoo", "date"], ascending=[True, False])
    return safe_write_csv(combined[EVENT_COLUMNS_V55], EVENTS_PATH)


def load_persisted_events(ticker: str, days_back: int = 1825) -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        return empty_events_frame()
    try:
        events = normalize_events_frame(pd.read_csv(EVENTS_PATH))
    except Exception:
        return empty_events_frame()
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(days_back))
    events = events[(events["ticker_yahoo"] == clean_ticker(ticker)) & (events["date"] >= cutoff)].copy()
    return detect_event_conflicts(events.sort_values("date", ascending=False).reset_index(drop=True))


def fetch_verified_event_bundle(
    ticker: str,
    company_name: str,
    days_back: int,
    days_forward: int = 730,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sec_events, sec_diagnostics = fetch_sec_filings_events(ticker, company_name, days_back=max(days_back, 730))
    ir_events, ir_diagnostics = fetch_registered_ir_events(ticker, days_back=max(days_back, 730), days_forward=days_forward)
    manual_events = read_manual_events(ticker)
    manual_diagnostic = pd.DataFrame([{
        "source": "Manuell bestätigte Termine",
        "kind": "manual",
        "status": "Lokal",
        "http_status": None,
        "entries": len(manual_events),
        "matches": len(manual_events),
        "uncertain_matches": 0,
        "duration_ms": 0,
        "message": "Termine aus data/manual_events.csv",
        "url": str(MANUAL_EVENTS_PATH),
    }])
    events = normalize_events_frame(pd.concat([sec_events, ir_events, manual_events], ignore_index=True))
    diagnostics = pd.concat([sec_diagnostics, ir_diagnostics, manual_diagnostic], ignore_index=True)
    return events, diagnostics


def _event_type_label(value: Any) -> str:
    event_type = str(value or "news")
    icon = {
        "news": "📰", "earnings": "📊", "dividend": "💶", "annual_meeting": "🗳️",
        "report": "📄", "analyst": "🎯", "capital_markets_day": "🏛️", "conference": "🎤",
    }.get(event_type, "•")
    label = EVENT_META.get(event_type, {}).get("label", event_type)
    return f"{icon} {label}"


def _render_event_table(frame: pd.DataFrame, empty_message: str) -> None:
    frame = normalize_events_frame(frame)
    if frame.empty:
        st.info(empty_message)
        return
    display = frame.copy()
    display["Datum"] = display["date"].map(_friendly_event_date)
    display["Typ"] = display["event_type"].map(_event_type_label)
    display["Titel"] = display["title"]
    display["Quelle"] = display["source"]
    display["Verifiziert"] = display["verification_level"].map(verification_label)
    display["Status"] = display["event_status"].replace("", "prüfen")
    display["Sentiment"] = display["sentiment_label"].map(sentiment_badge)
    display["Priorität"] = display["importance"].str.capitalize()
    display["Hinweis"] = display["conflict_note"].where(display["conflict_note"].ne(""), display["notes"])
    display["Quelle öffnen"] = display["link"].where(display["link"].ne(""), display["source_url"])
    visible = ["Datum", "Typ", "Titel", "Quelle", "Verifiziert", "Status", "Sentiment", "Priorität", "Hinweis", "Quelle öffnen"]
    styled = (
        display[visible].style
        .map(sentiment_cell_style, subset=["Sentiment"])
        .map(verification_cell_style, subset=["Verifiziert"])
        .map(event_status_cell_style, subset=["Status"])
    )
    try:
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Quelle öffnen": st.column_config.LinkColumn("Quelle", display_text="Öffnen"),
                "Titel": st.column_config.TextColumn("Titel", width="large"),
                "Hinweis": st.column_config.TextColumn("Hinweis", width="large"),
            },
        )
    except (TypeError, AttributeError):
        st.dataframe(styled, use_container_width=True, hide_index=True)


def render_event_calendar(events: pd.DataFrame) -> None:
    st.markdown("#### Ereigniskalender 2.0")
    display = normalize_events_frame(events)
    if display.empty:
        st.info("Noch keine Ereignisse gespeichert. Aktualisiere die Aktie oder ergänze einen bestätigten Termin.")
        return
    event_options = sorted(display["event_type"].dropna().astype(str).unique().tolist())
    sentiment_options = ["positiv", "negativ", "gemischt", "neutral"]
    verification_options = sorted(display["verification_level"].dropna().astype(str).unique().tolist())
    status_options = sorted(display["event_status"].replace("", "prüfen").dropna().astype(str).unique().tolist())
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        selected_types = st.multiselect(
            "Ereignistypen", event_options, default=event_options,
            format_func=_event_type_label, key="calendar_event_types_v55",
        )
    with filter_col2:
        selected_sentiments = st.multiselect(
            "Sentiment", sentiment_options, default=sentiment_options,
            format_func=sentiment_badge, key="calendar_sentiments_v55",
        )
    with filter_col3:
        selected_verifications = st.multiselect(
            "Quellenqualität", verification_options, default=verification_options,
            format_func=verification_label, key="calendar_verifications_v55",
        )
    with filter_col4:
        selected_statuses = st.multiselect(
            "Status", status_options, default=status_options, key="calendar_statuses_v55",
        )
    display = display[
        display["event_type"].isin(selected_types)
        & display["sentiment_label"].str.lower().isin(selected_sentiments)
        & display["verification_level"].isin(selected_verifications)
        & display["event_status"].replace("", "prüfen").isin(selected_statuses)
    ]
    today = pd.Timestamp.now().normalize()
    upcoming = display[display["date"].dt.normalize() >= today].sort_values(["date", "verification_score"], ascending=[True, False])
    past = display[display["date"].dt.normalize() < today].sort_values(["date", "verification_score"], ascending=[False, False])
    counts = st.columns(4)
    counts[0].metric("Kommende Termine", len(upcoming))
    counts[1].metric("Offiziell bestätigt", int(display["verification_level"].isin(["official", "official_ir"]).sum()))
    counts[2].metric("Datenanbieter / abgeleitet", int(display["verification_level"].isin(["provider", "derived"]).sum()))
    counts[3].metric("Bitte prüfen", int(display["event_status"].str.lower().eq("prüfen").sum()))
    conflicts = display[display["conflict_note"].str.len() > 0]
    if not conflicts.empty:
        st.warning(
            f"{len(conflicts)} Terminangabe(n) haben abweichende Quellen. "
            "Prüfe vor einer Entscheidung bevorzugt die offizielle Unternehmensquelle."
        )
    st.markdown("##### Kommende Termine")
    _render_event_table(upcoming, "Keine kommenden Termine für die gewählten Filter.")
    with st.expander(f"Vergangene Ereignisse ({len(past)})", expanded=upcoming.empty):
        _render_event_table(past, "Keine vergangenen Ereignisse für die gewählten Filter.")
    with st.expander("Quellenqualität und Status verstehen", expanded=False):
        st.markdown(
            """
- **🟢 Offiziell:** Einreichung bei einer Behörde, aktuell SEC EDGAR.
- **🟢 Unternehmens-IR:** RSS-/ICS-Quelle, die du als offizielle Investor-Relations-Quelle hinterlegt hast.
- **🔵 Manuell bestätigt:** Von dir mit Datum und Quelle eingetragener Termin.
- **🟡 Datenanbieter:** Best-Effort-Termin von Yahoo; bitte vor wichtigen Entscheidungen mit der IR-Seite abgleichen.
- **⚪ Aus News abgeleitet:** Ereignistyp wurde aus einer relevanten Überschrift/Beschreibung erkannt.
- **Status „prüfen“:** Quellen nennen nahe beieinanderliegende, aber unterschiedliche Termine.

Ein offizielles Filing bestätigt, dass ein Dokument eingereicht wurde. Es ist nicht automatisch ein positives oder negatives Signal.
            """
        )


def render_event_source_management(ticker: str, company_name: str) -> None:
    ensure_event_source_files()
    with st.expander("Verifizierte Event-Quellen & manuelle Termine", expanded=False):
        st.caption(
            "US-Ticker werden automatisch mit SEC EDGAR abgeglichen. Für andere Unternehmen kannst du "
            "offizielle RSS-/Atom-Feeds oder ICS-Kalender der Investor-Relations-Seite hinterlegen."
        )
        current = read_ir_sources()
        current = current[current["ticker_yahoo"] == clean_ticker(ticker)].copy()
        if current.empty:
            st.info("Für diese Aktie ist noch keine Unternehmens-IR-Quelle hinterlegt.")
        else:
            st.dataframe(
                current[["source_name", "source_type", "feed_url", "source_url", "verification_level", "enabled", "notes"]],
                use_container_width=True, hide_index=True,
            )

        with st.form(f"add_ir_source_{ticker}", clear_on_submit=True):
            source_cols = st.columns(2)
            source_name = source_cols[0].text_input("Quellenname", placeholder=f"{company_name} Investor Relations")
            source_type_label = source_cols[1].selectbox("Format", ["RSS / Atom", "ICS-Kalender", "Webseite (nur Link)"])
            feed_url = st.text_input("Feed-/Kalender-URL", placeholder="https://...")
            source_url = st.text_input("Offizielle Quellseite", placeholder="https://.../investor-relations")
            verification_choice = st.selectbox(
                "Quellenstatus", ["Offizielle Unternehmens-IR", "Andere/noch zu prüfende Quelle"]
            )
            notes = st.text_input("Notiz", placeholder="z. B. Finanzkalender des Unternehmens")
            submitted = st.form_submit_button("Quelle speichern")
            if submitted:
                source_type = {"RSS / Atom": "rss", "ICS-Kalender": "ics", "Webseite (nur Link)": "web"}[source_type_label]
                if not source_name.strip() or (source_type != "web" and not feed_url.strip()) or (source_type == "web" and not source_url.strip()):
                    st.error("Bitte Quellenname und eine passende URL angeben.")
                else:
                    all_sources = read_ir_sources()
                    addition = pd.DataFrame([{
                        "ticker_yahoo": clean_ticker(ticker),
                        "source_name": source_name.strip(),
                        "source_type": source_type,
                        "feed_url": feed_url.strip(),
                        "source_url": source_url.strip(),
                        "verification_level": "official_ir" if verification_choice.startswith("Offizielle") else "unknown",
                        "enabled": True,
                        "notes": notes.strip(),
                    }])
                    all_sources = pd.concat([all_sources, addition], ignore_index=True).drop_duplicates(
                        ["ticker_yahoo", "source_name", "feed_url"], keep="last"
                    )
                    ok, message = save_ir_sources(all_sources)
                    fetch_registered_ir_events.clear()
                    if ok:
                        st.success("Event-Quelle wurde gespeichert.")
                        st.rerun()
                    else:
                        st.warning(message)

        if not current.empty:
            delete_labels = {
                index: f"{row['source_name']} · {row['source_type']} · {row['feed_url'] or row['source_url']}"
                for index, row in current.iterrows()
            }
            delete_index = st.selectbox(
                "Quelle entfernen", options=list(delete_labels), format_func=lambda value: delete_labels[value],
                key=f"delete_ir_source_select_{ticker}",
            )
            if st.button("Ausgewählte Quelle entfernen", key=f"delete_ir_source_{ticker}"):
                all_sources = read_ir_sources()
                target = current.loc[delete_index]
                mask = (
                    all_sources["ticker_yahoo"].eq(clean_ticker(ticker))
                    & all_sources["source_name"].eq(str(target["source_name"]))
                    & all_sources["feed_url"].eq(str(target["feed_url"]))
                )
                ok, message = save_ir_sources(all_sources.loc[~mask].copy())
                fetch_registered_ir_events.clear()
                if ok:
                    st.success("Quelle wurde entfernt.")
                    st.rerun()
                else:
                    st.warning(message)

        st.divider()
        st.markdown("##### Bestätigten Termin manuell ergänzen")
        with st.form(f"manual_event_form_{ticker}", clear_on_submit=True):
            manual_cols = st.columns(3)
            manual_date = manual_cols[0].date_input("Datum", value=datetime.now().date())
            manual_type = manual_cols[1].selectbox(
                "Ereignistyp", options=list(EVENT_META), format_func=lambda value: _event_type_label(value)
            )
            manual_importance = manual_cols[2].selectbox("Priorität", ["hoch", "mittel", "niedrig"], index=1)
            manual_title = st.text_input("Titel", placeholder="z. B. Hauptversammlung 2027")
            manual_source = st.text_input("Quelle", value=f"{company_name} Investor Relations")
            manual_link = st.text_input("Quellenlink", placeholder="https://...")
            manual_notes = st.text_area("Notiz", height=70)
            save_manual = st.form_submit_button("Termin speichern")
            if save_manual:
                if not manual_title.strip():
                    st.error("Bitte einen Titel eintragen.")
                else:
                    ok, message = append_manual_event({
                        "date": pd.Timestamp(manual_date).strftime("%Y-%m-%d"),
                        "ticker_yahoo": clean_ticker(ticker),
                        "event_type": manual_type,
                        "title": manual_title.strip(),
                        "source": manual_source.strip() or "Manuell bestätigt",
                        "link": manual_link.strip(),
                        "importance": manual_importance,
                        "event_status": "bestätigt",
                        "verification_level": "manual",
                        "notes": manual_notes.strip(),
                    })
                    if ok:
                        st.success("Termin wurde gespeichert. Aktualisiere danach News & Ereignisse.")
                        st.rerun()
                    else:
                        st.warning(message)

        manual = read_manual_events(ticker)
        if not manual.empty:
            manual = manual.sort_values("date", ascending=False).reset_index(drop=True)
            labels = {
                index: f"{row['date'].strftime('%d.%m.%Y')} · {_event_type_label(row['event_type'])} · {row['title']}"
                for index, row in manual.iterrows()
            }
            delete_manual_index = st.selectbox(
                "Manuellen Termin entfernen", options=list(labels), format_func=lambda value: labels[value],
                key=f"delete_manual_event_select_{ticker}",
            )
            if st.button("Ausgewählten Termin entfernen", key=f"delete_manual_event_{ticker}"):
                target = manual.loc[delete_manual_index]
                ok, message = delete_manual_event(
                    ticker,
                    target["date"].strftime("%Y-%m-%d"),
                    str(target["event_type"]),
                    str(target["title"]),
                )
                if ok:
                    st.success("Termin wurde entfernt.")
                    st.rerun()
                else:
                    st.warning(message)

        st.download_button(
            "IR-Quellen-Vorlage herunterladen",
            data=pd.DataFrame(columns=IR_SOURCE_COLUMNS).to_csv(index=False).encode("utf-8-sig"),
            file_name="ir_sources_template.csv",
            mime="text/csv",
            key=f"download_ir_template_{ticker}",
        )

def render_news(df: pd.DataFrame) -> None:
    st.subheader("News & Ereignisse")
    st.caption(
        "Relevante Unternehmensnews und Kalendertermine werden nach Quellenqualität getrennt. "
        "SEC-Filings, Unternehmens-IR, manuelle Bestätigungen, Datenanbieter und abgeleitete News sind klar gekennzeichnet."
    )
    with st.expander("Sentiment-Logik kurz erklärt", expanded=False):
        st.markdown(
            """
Das Label beschreibt nur den **Ton der RSS-Überschrift und Kurzbeschreibung**, nicht die Qualität der Aktie.

- 🟢 **Positiv:** klare Verbesserungen wie Prognoseanhebung, Erwartungsübertreffen, Dividendenerhöhung oder Upgrade.
- 🔴 **Negativ:** klare Verschlechterungen wie Gewinnwarnung, Prognosesenkung, Dividendenkürzung, Verfehlung oder Downgrade.
- 🟡 **Gemischt:** positive und negative Aussagen gleichzeitig.
- ⚪ **Neutral:** Termin-/Informationsmeldung ohne klare Richtung.

Starke Phrasen werden höher gewichtet und Überschriften zählen stärker als Beschreibungen. Das Ergebnis ist eine Recherchehilfe, kein Handelssignal.
            """
        )

    ticker = company_selectbox("Aktie für News", df, key="news_ticker")
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

    render_event_source_management(ticker, company_name)

    if st.button("News & Ereignisse aktualisieren", key=f"refresh_news_{ticker}", type="primary"):
        with st.spinner("News, Yahoo-Kalender, offizielle Filings und hinterlegte IR-Quellen werden geladen …"):
            news, diagnostics = fetch_news_bundle(
                ticker=ticker,
                company_name=company_name,
                days_back=days_back,
                include_google_news=include_google,
                locale=locale_code,
            )
            calendar_events = fetch_yahoo_calendar_events(ticker, days_back=max(365, days_back), days_forward=730)
            verified_events, event_diagnostics = fetch_verified_event_bundle(
                ticker=ticker, company_name=company_name, days_back=max(730, days_back), days_forward=730
            )
            diagnostics = pd.concat([diagnostics, event_diagnostics], ignore_index=True, sort=False)
            events = combine_events(news, calendar_events, ticker, extra_events=verified_events)
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
    for column, default in {
        "sentiment_confidence": "niedrig",
        "sentiment_reason": "",
    }.items():
        if column not in news.columns:
            news[column] = default
        if column not in events.columns:
            events[column] = default
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
    if not diagnostics.empty:
        http_ok = diagnostics.get("http_status", pd.Series(index=diagnostics.index, dtype=float)).fillna(0).astype(int).between(200, 299)
        local_ok = diagnostics.get("status", pd.Series(index=diagnostics.index, dtype=str)).fillna("").astype(str).isin(
            ["OK", "Lokal", "Nur Link", "Keine relevanten Filings", "Keine Kalenderereignisse"]
        )
        reachable_sources = int((http_ok | local_ok).sum())

    status_columns = st.columns(4)
    status_columns[0].metric("Relevante News", int(len(relevant_news)))
    status_columns[1].metric("Nächstes Ereignis", pd.Timestamp(future_events.iloc[0]["date"]).strftime("%d.%m.%Y") if not future_events.empty else "–")
    status_columns[2].metric("Abrufbare Quellen", f"{reachable_sources} / {total_sources}" if total_sources else "–")
    status_columns[3].metric("Daten aktualisiert", bundle.get("loaded_at", "–"))

    active_sentiment_filter = render_news_summary(ticker, company_name, relevant_news, uncertain_news, events, diagnostics)

    # Auch die Unternavigation des News-Bereichs bleibt bei Reruns erhalten.
    news_views = ["Aktuelle News", "Kalender", "Quellen & Diagnose", "Export"]
    if st.session_state.get("news_navigation") not in news_views:
        st.session_state["news_navigation"] = news_views[0]

    active_news_view = st.radio(
        "News-Bereich auswählen",
        options=news_views,
        horizontal=True,
        key="news_navigation",
        label_visibility="collapsed",
    )
    st.divider()

    if active_news_view == "Aktuelle News":
        filtered_relevant_news = relevant_news.copy()
        if active_sentiment_filter != "alle" and not filtered_relevant_news.empty:
            filtered_relevant_news = filtered_relevant_news[
                filtered_relevant_news.get(
                    "sentiment_label",
                    pd.Series("neutral", index=filtered_relevant_news.index),
                )
                .fillna("neutral")
                .astype(str)
                .str.lower()
                .eq(active_sentiment_filter)
            ]

        filter_titles = {
            "alle": "Alle relevanten News",
            "positiv": "Positive News",
            "negativ": "Negative News",
            "gemischt": "Gemischte News",
            "neutral": "Neutrale News",
        }
        st.markdown(f"#### {filter_titles.get(active_sentiment_filter, 'Aktuelle News')}")

        if relevant_news.empty:
            st.warning(
                "Keine verlässlich passende Unternehmensmeldung gefunden. "
                "Prüfe bei Bedarf die Firmen-Aliase oder die Quellen-Diagnose."
            )
        elif filtered_relevant_news.empty:
            st.info(
                "Für den gewählten Sentimentfilter gibt es im aktuellen Zeitraum keine Meldung. "
                "Klicke oben auf „Alle“ oder wähle einen anderen Filter."
            )
        else:
            st.caption(
                f"{len(filtered_relevant_news)} von {len(relevant_news)} relevanten Meldungen werden angezeigt."
            )
            for index, (_, item) in enumerate(filtered_relevant_news.iterrows()):
                render_news_card(item, index)

        if not uncertain_news.empty:
            with st.expander(f"Unsichere Treffer anzeigen ({len(uncertain_news)})", expanded=False):
                st.caption(
                    "Diese Artikel hatten einen zu schwachen Firmenbezug und werden nicht als relevante Unternehmensnews gezählt. "
                    "Sie erscheinen nicht im Kalender und nicht im Chart."
                )
                for index, (_, item) in enumerate(uncertain_news.iterrows(), start=1000):
                    render_news_card(item, index)

    elif active_news_view == "Kalender":
        st.caption("Bevorzuge grün markierte offizielle Quellen. Yahoo-Termine und aus News abgeleitete Ereignisse dienen als Ergänzung und sollten gegengeprüft werden.")
        render_event_calendar(events)

    elif active_news_view == "Quellen & Diagnose":
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
            http_series = diagnostics.get("http_status", pd.Series(index=diagnostics.index, dtype=float))
            status_series = diagnostics.get("status", pd.Series(index=diagnostics.index, dtype=str)).fillna("").astype(str)
            attempted_http = http_series.notna()
            failed = diagnostics[attempted_http & ~http_series.fillna(0).astype(int).between(200, 299)]
            if not failed.empty:
                st.warning("Mindestens eine Onlinequelle war nicht erreichbar oder lieferte keinen brauchbaren Feed. Das ist ein Quellenproblem, kein Handelssignal.")
            not_configured = diagnostics[status_series.eq("Nicht konfiguriert")]
            if not not_configured.empty:
                st.info("Für mindestens eine optionale offizielle Quelle ist noch kein IR-Feed/Kalender hinterlegt.")

    elif active_news_view == "Export":
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
        "Sie ist eine Filterhilfe. Für Termine zeigt der Kalender deshalb zusätzlich Quellenqualität, Bestätigungsstatus und mögliche Konflikte."
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

    st.divider()
    render_portfolio_health(portfolio_view)
    st.divider()
    render_portfolio_risk(portfolio_view, all_histories)


def render_watchlist(
    metrics: pd.DataFrame,
    drawdown_trigger: float,
    payout_max: float,
    score_min: float,
    yield_min: float,
) -> None:
    """Zeigt die Watchlist unabhängig vom aktuell gewählten Index.

    Werte, die bereits im geladenen Index enthalten sind, verwenden dessen
    Marktdaten. Für alle übrigen Watchlist-Ticker können die Daten über einen
    eigenen Button separat geladen und im Session State zwischengespeichert
    werden. Dadurch bleiben beispielsweise US-Titel sichtbar, auch wenn gerade
    der DAX ausgewählt ist.
    """
    st.subheader("Eigene Watchlist")
    watchlist = read_watchlist()
    if watchlist.empty:
        st.info("Noch keine Titel gespeichert. Nutze in der Einzelanalyse den Button „Zur Watchlist hinzufügen“.")
        return

    watchlist = watchlist.copy()
    watchlist["ticker_yahoo"] = watchlist["ticker_yahoo"].map(clean_ticker)
    watchlist = watchlist[watchlist["ticker_yahoo"].astype(bool)].drop_duplicates("ticker_yahoo")

    current_metrics = metrics.copy() if metrics is not None else empty_metrics_frame()
    if not current_metrics.empty:
        current_metrics["ticker_yahoo"] = current_metrics["ticker_yahoo"].map(clean_ticker)
    current_tickers = set(current_metrics.get("ticker_yahoo", pd.Series(dtype=str)).astype(str))
    watchlist_tickers = set(watchlist["ticker_yahoo"].astype(str))
    independent_tickers = sorted(watchlist_tickers - current_tickers)

    # Bereits separat geladene Rohdaten beibehalten, aber auf die aktuelle
    # Watchlist beschränken. Scores werden bei jedem Rerun mit den aktuellen
    # Scanner-Regeln neu berechnet.
    extra_raw = st.session_state.get("watchlist_extra_raw_metrics")
    if extra_raw is None:
        extra_raw = empty_metrics_frame()
    else:
        extra_raw = extra_raw.copy()
    if not extra_raw.empty:
        extra_raw["ticker_yahoo"] = extra_raw["ticker_yahoo"].map(clean_ticker)
        extra_raw = extra_raw[extra_raw["ticker_yahoo"].isin(watchlist_tickers)].copy()

    controls = st.columns([1.35, 1, 1, 1.4])
    load_clicked = controls[0].button(
        "Watchlist-Daten laden / aktualisieren",
        type="primary",
        use_container_width=True,
        key="load_watchlist_market_data",
        help="Lädt nur Watchlist-Titel separat, die nicht bereits im aktuell geladenen Index enthalten sind.",
    )

    if load_clicked:
        if independent_tickers:
            extra_constituents = watchlist[watchlist["ticker_yahoo"].isin(independent_tickers)][
                ["ticker_yahoo", "name", "sector"]
            ].copy()
            extra_constituents["name"] = extra_constituents["name"].replace(r"^\s*$", pd.NA, regex=True)
            extra_constituents["name"] = extra_constituents["name"].fillna(extra_constituents["ticker_yahoo"])
            extra_constituents["sector"] = extra_constituents["sector"].replace(r"^\s*$", pd.NA, regex=True).fillna("Unbekannt")

            with st.spinner(f"Lade Marktdaten für {len(extra_constituents)} Watchlist-Titel …"):
                loaded_raw, loaded_histories, load_errors = collect_metrics(extra_constituents)
            st.session_state["watchlist_extra_raw_metrics"] = loaded_raw
            st.session_state["watchlist_extra_histories"] = loaded_histories
            st.session_state["watchlist_load_errors"] = load_errors
        else:
            # Alle Titel sind bereits Bestandteil des aktuell geladenen Index.
            st.session_state["watchlist_extra_raw_metrics"] = empty_metrics_frame()
            st.session_state["watchlist_extra_histories"] = {}
            st.session_state["watchlist_load_errors"] = []
        st.session_state["watchlist_last_refresh"] = datetime.now().strftime("%d.%m.%Y %H:%M")
        st.rerun()

    # Nach einem eventuellen Rerun die gespeicherten Daten erneut lesen.
    extra_raw = st.session_state.get("watchlist_extra_raw_metrics", empty_metrics_frame())
    if extra_raw is None:
        extra_raw = empty_metrics_frame()
    extra_raw = extra_raw.copy()
    if not extra_raw.empty:
        extra_raw["ticker_yahoo"] = extra_raw["ticker_yahoo"].map(clean_ticker)
        extra_raw = extra_raw[extra_raw["ticker_yahoo"].isin(watchlist_tickers)].copy()
        extra_scored = enrich_with_scores(
            extra_raw,
            drawdown_trigger=float(drawdown_trigger),
            payout_max=float(payout_max),
            score_min=float(score_min),
            yield_min=float(yield_min),
        )
        extra_scored = enrich_with_special_situations(extra_scored)
    else:
        extra_scored = empty_metrics_frame()

    # Separat geladene Daten zuerst, aktuell geladene Indexdaten zuletzt: Bei
    # Duplikaten haben die frisch sichtbaren Indexdaten Vorrang.
    combined_metrics = pd.concat([extra_scored, current_metrics], ignore_index=True, sort=False)
    if not combined_metrics.empty:
        combined_metrics["ticker_yahoo"] = combined_metrics["ticker_yahoo"].map(clean_ticker)
        combined_metrics = combined_metrics.drop_duplicates("ticker_yahoo", keep="last")

    merged = watchlist.merge(combined_metrics, on="ticker_yahoo", how="left", suffixes=("", "_market"))
    loaded_mask = pd.to_numeric(merged.get("last_price"), errors="coerce").notna()
    loaded_count = int(loaded_mask.sum())
    total_count = int(len(merged))
    open_count = max(total_count - loaded_count, 0)

    controls[1].metric("Watchlist-Titel", total_count)
    controls[2].metric("Daten geladen", loaded_count)
    controls[3].metric("Letzte separate Abfrage", st.session_state.get("watchlist_last_refresh", "Noch nicht geladen"))

    if open_count:
        missing_names = merged.loc[~loaded_mask, "name"].fillna(merged.loc[~loaded_mask, "ticker_yahoo"]).astype(str).tolist()
        st.info(
            f"Für {open_count} Watchlist-Titel fehlen noch Marktdaten. Klicke auf „Watchlist-Daten laden / aktualisieren“. "
            f"Offen: {', '.join(missing_names[:8])}" + (" …" if len(missing_names) > 8 else "")
        )
    else:
        st.success("Alle Watchlist-Titel verfügen über Marktdaten – unabhängig vom aktuell ausgewählten Index.")

    load_errors = st.session_state.get("watchlist_load_errors", [])
    if load_errors:
        with st.expander(f"Hinweise zu {len(load_errors)} Watchlist-Datenabruf(en)"):
            for message in load_errors:
                st.write(f"- {message}")

    visible_columns = [
        "ticker_yahoo", "name", "sector", "added_at", "note", "last_price", "currency", "change_1y",
        "dividend_yield", "total_score", "value_score", "value_trigger",
    ]
    visible_columns = [column for column in visible_columns if column in merged.columns]
    display = merged[visible_columns].rename(columns={
        "ticker_yahoo": "Ticker",
        "name": "Unternehmen",
        "sector": "Sektor",
        "added_at": "Hinzugefügt",
        "note": "Notiz",
        "last_price": "Letzter Kurs",
        "currency": "Währung",
        "change_1y": "Kursrendite 1J",
        "dividend_yield": "Dividendenrendite",
        "total_score": "Qualitäts-Score",
        "value_score": "Value-Score",
        "value_trigger": "Value-Trigger",
    })

    display_formats = {
        "Letzter Kurs": lambda value: format_number(value, 2),
        "Kursrendite 1J": lambda value: format_percent(value, 2, signed=True),
        "Dividendenrendite": lambda value: format_percent(value, 2),
        "Qualitäts-Score": lambda value: format_number(value, 1),
        "Value-Score": lambda value: format_number(value, 1),
    }
    styled = display.style.format(
        {key: formatter for key, formatter in display_formats.items() if key in display.columns},
        na_rep="–",
    )
    if "Kursrendite 1J" in display.columns:
        styled = styled.map(colorize_change, subset=["Kursrendite 1J"])
    if "Qualitäts-Score" in display.columns:
        styled = styled.map(colorize_score, subset=["Qualitäts-Score"])
    if "Value-Trigger" in display.columns:
        styled = styled.map(colorize_value_trigger, subset=["Value-Trigger"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    ticker_to_remove = company_selectbox("Titel von Watchlist entfernen", merged, key="remove_watchlist_ticker")
    if st.button("Aus Watchlist entfernen", key="remove_watchlist_button"):
        remove_from_watchlist(ticker_to_remove)
        # Gespeicherte Zusatzdaten beim Entfernen ebenfalls bereinigen.
        stored = st.session_state.get("watchlist_extra_raw_metrics")
        if isinstance(stored, pd.DataFrame) and not stored.empty:
            st.session_state["watchlist_extra_raw_metrics"] = stored[
                stored["ticker_yahoo"].map(clean_ticker) != clean_ticker(ticker_to_remove)
            ].copy()
        st.success(f"{ticker_to_remove} wurde entfernt.")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    show_header()

    # ----- Sidebar: Daten, Filter und Scanner-Parameter -----
    with st.sidebar:
        st.header("Analyse-Einstellungen")
        index_name = st.selectbox("Index", INDEX_OPTIONS, key="index_name")
        st.caption(
            "DAX, MDAX und SDAX werden offline aus integrierten Listen geladen. "
            "Eigene CSVs unter data/indices/ überschreiben die Vorlage."
        )

        try:
            constituents = load_index_constituents(index_name)
            st.success(
                f"{index_name}: {len(constituents)} Unternehmen geladen",
                icon="✅",
            )
            st.caption(f"Indexquelle: {index_source_description(index_name)}")
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
        drawdown_trigger = st.slider(
            "Min. Drawdown vom 52W-Hoch",
            min_value=10,
            max_value=60,
            step=5,
            key="scanner_drawdown",
        )
        payout_max = st.slider(
            "Max. Payout Ratio",
            min_value=40,
            max_value=120,
            step=5,
            key="scanner_payout",
        )
        score_min = st.slider(
            "Min. Qualitäts-Score",
            min_value=0,
            max_value=100,
            step=5,
            key="scanner_score",
        )
        yield_min = st.slider(
            "Min. Dividendenrendite",
            min_value=1.0,
            max_value=10.0,
            step=0.5,
            key="scanner_yield",
        )

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
            fetch_sec_company_map.clear()
            fetch_sec_filings_events.clear()
            fetch_registered_ir_events.clear()
            fetch_benchmark_history.clear()
            fetch_long_history.clear()
            fetch_historical_financials.clear()
            fx_to_eur.clear()
            for state_key in [
                "metrics_raw", "histories", "loaded_tickers", "portfolio_extra_metrics",
                "portfolio_extra_histories", "watchlist_extra_raw_metrics",
                "watchlist_extra_histories", "watchlist_load_errors", "watchlist_last_refresh",
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

    # Persistente Hauptnavigation: Anders als st.tabs bleibt diese Auswahl bei jedem
    # Streamlit-Rerun erhalten, zum Beispiel nach dem Aktualisieren der News.
    main_pages = [
        "Überblick", "Datenstatus", "Fundamentaldaten", "Aktienprofile", "Einzelanalyse", "Sektoren",
        "News & Events", "Portfolio", "Watchlist", "Value-Scanner", "Deep Value", "Backtesting", "Mustervergleich", "Lernmodul", "Research",
    ]
    if st.session_state.get("main_navigation") not in main_pages:
        st.session_state["main_navigation"] = main_pages[0]

    active_page = st.radio(
        "Bereich auswählen",
        options=main_pages,
        horizontal=True,
        key="main_navigation",
        label_visibility="collapsed",
    )
    st.divider()

    if active_page == "Überblick":
        render_overview(data)
    elif active_page == "Datenstatus":
        render_data_status(status_summary, status_detail, data)
    elif active_page == "Fundamentaldaten":
        render_fundamentals(data)
    elif active_page == "Aktienprofile":
        render_profile_scores(data)
    elif active_page == "Einzelanalyse":
        render_risk_and_chart(data, histories)
    elif active_page == "Sektoren":
        render_sector_view(data, histories)
    elif active_page == "News & Events":
        render_news(data)
    elif active_page == "Portfolio":
        render_portfolio(data, histories)
    elif active_page == "Watchlist":
        render_watchlist(
            data,
            drawdown_trigger=float(drawdown_trigger),
            payout_max=float(payout_max),
            score_min=float(score_min),
            yield_min=float(yield_min),
        )
    elif active_page == "Value-Scanner":
        render_value_watchlist(
            data,
            drawdown_trigger=float(drawdown_trigger),
            payout_max=float(payout_max),
            score_min=float(score_min),
            yield_min=float(yield_min),
            profile_name=profile_name,
        )
    elif active_page == "Deep Value":
        render_special_situation_scanner(data)
    elif active_page == "Backtesting":
        render_bat_backtesting(data, index_name)
    elif active_page == "Mustervergleich":
        render_universe_pattern_comparison(data, index_name)
    elif active_page == "Lernmodul":
        render_learning_module(data)
    elif active_page == "Research":
        render_research(data, histories, index_name)



if __name__ == "__main__":
    main()