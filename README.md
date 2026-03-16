# RAG Chat Assistant

Ein Streamlit-basierter Chat-Assistent mit erweiterten RAG (Retrieval-Augmented Generation) Funktionen.

## Features

- **Intelligente Dokumentensuche** mit deutscher Lemmatisierung
- **Multi-Query Expansion** für bessere Suchergebnisse
- **Chat-Interface** mit Verlauf
- **Debug-Funktionen** zur Analyse der Suchergebnisse
- **Strukturierte Antworten** mit Quellenangaben

## Installation

1. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

2. Deutsches spaCy-Modell herunterladen:
```bash
python -m spacy download de_core_news_sm
```

3. Umgebungsvariablen einrichten:
```bash
# .env Datei erstellen (von .env.example kopieren)
cp .env.example .env
# Dann .env bearbeiten und echte Werte eintragen
```

Benötigte Umgebungsvariablen:
- `ANTHROPIC_API_KEY`: Dein Anthropic API Key
- `APP_PASSWORD`: Passwort für den App-Zugang (Standard: "DefaultPassword123!")
- `DOCS_DIR`: (Optional) Standard-Dokumentenverzeichnis

## Streamlit Cloud Deployment

### Secrets Konfiguration
**Nach dem Deployment** müssen API Keys und Passwort als Secrets hinzugefügt werden:

**Option 1: Über Streamlit Cloud Interface (Empfohlen)**
1. Gehe zu deiner deployed App
2. Unten rechts: "Manage app" → Settings → Secrets
3. Edit Secrets und folgendes eintragen:
```toml
ANTHROPIC_API_KEY = "sk-ant-api03-your-real-key-here"
APP_PASSWORD = "dein-sicheres-passwort-hier"
```

**Option 2: Via lokale secrets.toml**
```bash
# Erstelle lokale Secrets-Datei (nur für lokale Entwicklung)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Dann echte API Keys in secrets.toml eintragen
```

### Deployment Steps
1. Fork/Clone dieses Repository
2. Gehe zu https://share.streamlit.io
3. "New app" → Repository auswählen
4. Main file: `app.py`
5. Nach Deployment: Secrets konfigurieren (siehe oben)
6. App sollte funktionieren!

## Verwendung

1. App starten:
```bash
streamlit run app.py
```

2. In der Sidebar:
   - Dokumenten-Verzeichnis angeben
   - "Dokumente laden und indexieren" klicken

3. Chat verwenden:
   - Fragen in das Chat-Eingabefeld eingeben
   - Antworten werden basierend auf den indexierten Dokumenten generiert

## Funktionalitäten

### Dokument-Verarbeitung
- Lädt Markdown-Dateien aus einem Verzeichnis
- Teilt Dokumente in Chunks mit 1500 Zeichen und 400 Zeichen Überlappung
- Wendet deutsche Lemmatisierung an
- Erstellt Vektor-Embeddings mit HuggingFace

### Verbesserte Suche
- Query Expansion mit Lemmatisierung
- Keyword-Extraktion
- Multi-Query Retrieval
- Score-basierte Sortierung

### Chat-Interface
- Persistenter Chat-Verlauf
- Strukturierte Antworten
- Quellenangaben
- Debug-Funktionen

## Konfiguration

Die App kann über die Sidebar konfiguriert werden:
- Dokumenten-Verzeichnis festlegen
- Index neu laden
- Chat-Verlauf löschen

## Debug-Features

Im erweiterbaren Debug-Bereich können Sie:
- Test-Suchen durchführen
- Query-Varianten anzeigen
- Score-Werte der Suchergebnisse einsehen