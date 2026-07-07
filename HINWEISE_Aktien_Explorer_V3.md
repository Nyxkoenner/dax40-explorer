# Aktien Explorer 3.0 – Hinweise

1. Sichere zuerst deine bisherige `app.py`.
2. Benenne `app_aktien_explorer_v3.py` in `app.py` um oder ersetze den Inhalt deiner vorhandenen Datei.
3. Installiere die Pakete:
   `python -m pip install -r requirements.txt`
4. Starte:
   `python -m streamlit run app.py`

## Neue automatisch angelegte Dateien
- `data/company_aliases.csv` – Suchbegriffe pro Unternehmen für News
- `data/events.csv` – gespeicherte News-/Dividend-/Earnings-Ereignisse
- `data/transactions.csv` – optionales Transaktionsbuch
- `data/news_snapshots/` – zuletzt gespeicherte News je Ticker

## Portfolio
Die bisherige `portfolio.csv` bleibt kompatibel.
Wenn `data/transactions.csv` gefüllt ist, hat sie Vorrang:
`date,ticker_yahoo,type,shares,price,currency,fees,comment`
Beispiel:
`2025-01-15,ALV.DE,buy,10,245.50,EUR,4.90,Erstkauf`

## Wichtig
- News-Kalenderdaten und Kennzahlen hängen von Yahoo/RSS-Quellen ab und können fehlen.
- „Research“ ist ein historischer Kursvergleich, kein fundamentaler Backtest.
- Alle Scores sind Recherchehilfen, keine Kauf- oder Verkaufssignale.
