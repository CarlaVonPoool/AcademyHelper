# 🔒 Sicherheitshinweise für RAG Chat Assistant

## 🛡️ Implementierte Sicherheitsmaßnahmen

### 1. **Passwort-Authentifizierung**
- App ist durch Passwort geschützt (ENV: `APP_PASSWORD`)
- Session-basierte Authentifizierung
- Logout-Funktionalität verfügbar

### 2. **Embedding-Store Sicherheit**
- **Dateigröße-Limits**: Max. 500MB für Cache-Dateien
- **Integritätsprüfungen**: Validierung aller geladenen Daten
- **Directory Traversal Schutz**: Verhindert Zugriff außerhalb App-Verzeichnis
- **Automatische Korruptions-Erkennung**: Defekte Caches werden automatisch gelöscht

### 3. **Daten-Isolation**
- Embedding-Store in `./embeddings_store/` (nicht in Git)
- Dokument-Hashes für Änderungserkennung
- Keine Speicherung sensibler API-Keys im Cache

## ⚠️ Sicherheitsrisiken

### 1. **Pickle-Deserialisierung**
- **Risk**: Pickle kann beliebigen Code ausführen
- **Mitigation**: Cache-Validierung und Größen-Limits
- **Empfehlung**: Nur vertrauenswürdige Cache-Dateien verwenden

### 2. **Dokumenten-Inhalte**
- **Risk**: Sensitive Daten werden im Embedding-Store gespeichert
- **Mitigation**: Store ist in `.gitignore` und lokal
- **Empfehlung**: Regelmäßig Cache löschen bei sensiblen Dokumenten

### 3. **API-Key Management**
- **Risk**: Anthropic API-Key im Environment
- **Mitigation**: `.env` in `.gitignore`
- **Empfehlung**: Rotation der API-Keys

## 🔧 Empfohlene Sicherheitspraxis

### Für Entwicklung:
```bash
# 1. Sichere .env Datei erstellen
cp .env.example .env
# 2. Starke Passwörter setzen
# 3. API-Keys regelmäßig rotieren
```

### Für Produktion:
```bash
# 1. Cache-Verzeichnis sichern
chmod 700 embeddings_store/

# 2. Regelmäßige Cache-Bereinigung
# Schedule: rm -rf embeddings_store/* (bei Bedarf)

# 3. Environment Variables sichern
export APP_PASSWORD="sehr-sicheres-passwort"
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Cache-Management:
```python
# Sicherer Cache-Reset (in App)
1. Sidebar → Index-Verwaltung → "Index löschen" 
2. Oder manuell: rm -rf embeddings_store/
```

## 🚨 Bei Sicherheitsvorfällen

1. **Cache kompromittiert**: Sofort löschen mit "Index löschen"
2. **API-Key leaked**: Neuen Key generieren und rotieren
3. **Passwort kompromittiert**: `APP_PASSWORD` in `.env` ändern

## 📋 Security Checklist

- [ ] `.env` Datei ist in `.gitignore`
- [ ] `APP_PASSWORD` ist stark (min. 12 Zeichen)
- [ ] `ANTHROPIC_API_KEY` ist aktuell und sicher
- [ ] Embedding-Store wird nicht in Git committed
- [ ] Regelmäßige Cache-Bereinigung bei sensiblen Daten
- [ ] Nur vertrauenswürdige Dokumente indexieren

## 🛠️ Technische Details

### Cache-Validierung:
- Prüfung auf erforderliche Felder
- Datentyp-Validierung
- Embeddings-Konsistenz
- Größen-Limits

### Directory Traversal Schutz:
```python
docs_dir = os.path.abspath(docs_dir)
if not docs_dir.startswith(os.path.abspath(".")):
    raise SecurityError("Path outside app directory")
```

---
*Letzte Aktualisierung: 2025-01-15*