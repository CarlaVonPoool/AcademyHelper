# 📡 N8N Setup für Streamlit Feedback Logging

## 🚀 Quick Start

### 1. N8N installieren

**Option A: Docker (Empfohlen)**
```bash
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

**Option B: npm**
```bash
npm install n8n -g
n8n start
```

N8N läuft dann auf: http://localhost:5678

### 2. Workflow importieren

1. Öffne N8N in deinem Browser
2. Gehe zu "Workflows" → "Import from file" 
3. Importiere `n8n_workflow_feedback_logger.json`
4. Workflow aktivieren

### 3. Streamlit konfigurieren

**Lokale Entwicklung (.env Datei):**
```bash
N8N_WEBHOOK_URL=http://localhost:5678/webhook/feedback
N8N_API_KEY=optional-secret-key
```

**Streamlit Cloud (secrets.toml):**
```toml
N8N_WEBHOOK_URL = "https://your-n8n.com/webhook/feedback"
N8N_API_KEY = "your-secret-key"
```

## 📁 Datei-Struktur

N8N erstellt automatisch diese Dateien:

```
~/.n8n/
├── 2026-03-19_feedback.jsonl    # Alle Session-Daten
├── critical_feedback.jsonl      # Nur kritisches Feedback  
└── daily_stats.jsonl           # Aggregierte Statistiken
```

## 📊 Beispiel Dateiinhalte

### feedback.jsonl (Hauptdatei)
```jsonl
{"session_id":"abc123","session_start":"2026-03-19T10:00:00","interactions":[{"id":"int1","question":"Wie erstelle ich eine Rechnung?","answer":"Um eine Rechnung zu erstellen...","feedback":{"is_helpful":true}}],"statistics":{"total_interactions":1,"helpful_feedback":1}}
{"session_id":"def456","session_start":"2026-03-19T11:30:00","interactions":[{"id":"int2","question":"Wo finde ich die Einstellungen?","answer":"Die Einstellungen findest du...","feedback":{"is_helpful":false,"error_type":"Falsche Information"}}],"statistics":{"total_interactions":1,"unhelpful_feedback":1}}
```

### critical_feedback.jsonl (Nur Probleme)
```jsonl
{"timestamp":"2026-03-19T11:35:00","session_id":"def456","critical_interactions":[{"id":"int2","feedback":{"is_helpful":false,"error_type":"Falsche Information","user_comment":"Die Antwort war komplett falsch"}}]}
```

### daily_stats.jsonl (Aggregiert)
```jsonl
{"date":"2026-03-19","hour":"10","session_id":"abc123","interactions":1,"avg_confidence":0.95,"feedback_rate":100,"helpful_rate":100}
{"date":"2026-03-19","hour":"11","session_id":"def456","interactions":1,"avg_confidence":0.75,"feedback_rate":100,"helpful_rate":0}
```

## 🔧 Erweiterte N8N Workflows

### Email-Benachrichtigung bei kritischem Feedback
```json
{
  "name": "Email Alert on Critical Feedback",
  "trigger": "File Watch: critical_feedback.jsonl", 
  "action": "Send Email to admin@company.com"
}
```

### Slack-Integration
```json
{
  "name": "Slack Notification",
  "trigger": "Webhook: /feedback",
  "condition": "if helpful_rate < 70%",
  "action": "Post to #feedback channel"
}
```

### Automatische Berichte
```json
{
  "name": "Daily Report",
  "trigger": "Cron: 0 8 * * *",
  "action": "Generate PDF report from daily_stats.jsonl"
}
```

## 🔍 Daten auswerten

### Mit jq (Command Line)
```bash
# Heute's Statistiken
cat daily_stats.jsonl | jq 'select(.date == "2026-03-19")'

# Durchschnittliche Hilfreich-Rate
cat daily_stats.jsonl | jq '.helpful_rate' | awk '{sum+=$1} END {print sum/NR}'

# Alle kritischen Feedbacks
cat critical_feedback.jsonl | jq '.critical_interactions[].feedback.user_comment'
```

### Mit Python
```python
import json
import pandas as pd

# Lade alle Feedback-Daten
with open('2026-03-19_feedback.jsonl') as f:
    sessions = [json.loads(line) for line in f]

# Analysiere
total_interactions = sum(s['statistics']['total_interactions'] for s in sessions)
avg_helpful_rate = sum(s['statistics'].get('helpful_rate', 0) for s in sessions) / len(sessions)

print(f"Heute: {total_interactions} Interaktionen, {avg_helpful_rate:.1f}% hilfreich")
```

### Mit SQL (wenn du N8N → PostgreSQL erweiterst)
```sql
SELECT 
  DATE(session_start) as date,
  COUNT(*) as sessions,
  AVG(statistics->>'helpful_rate') as avg_helpful_rate
FROM feedback_sessions 
WHERE session_start > NOW() - INTERVAL '7 days'
GROUP BY DATE(session_start);
```

## 🛡️ Sicherheit & Datenschutz

### API-Key für Webhook
```javascript
// In N8N Webhook Node:
if (headers['x-api-key'] !== 'your-secret-key') {
  return []; // Blockiere Request
}
```

### Daten-Anonymisierung
```javascript
// N8N Code Node für Anonymisierung:
const anonymized = items[0].json.session_data;

// Entferne/hash sensible Daten
anonymized.interactions.forEach(i => {
  i.question = i.question.replace(/\b[A-Z][a-z]+ [A-Z][a-z]+\b/g, '[NAME]');
  i.question = i.question.replace(/\S+@\S+/g, '[EMAIL]');
});

return [{json: anonymized}];
```

### Automatische Bereinigung
```javascript
// N8N Cron Job - täglich um 2 Uhr
// Lösche Dateien älter als 30 Tage
const fs = require('fs');
const path = require('path');
const maxAge = 30 * 24 * 60 * 60 * 1000; // 30 Tage

fs.readdirSync('.').forEach(file => {
  if (file.endsWith('_feedback.jsonl')) {
    const stats = fs.statSync(file);
    if (Date.now() - stats.mtime.getTime() > maxAge) {
      fs.unlinkSync(file);
    }
  }
});
```

## 🚀 Production Setup

### N8N Cloud (Kostenpflichtig)
- Webhook URL: `https://yourname.app.n8n.cloud/webhook/feedback`
- Automatische Backups
- Höhere Performance

### N8N Self-Hosted mit Docker Compose
```yaml
version: '3.7'
services:
  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    volumes:
      - n8n_data:/home/node/.n8n
      - ./feedback_files:/home/node/feedback_files
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=your-password
volumes:
  n8n_data:
```

## 📈 Monitoring

### Webhook Health Check
```python
# Teste ob N8N Webhook funktioniert
import requests

try:
    response = requests.post('http://localhost:5678/webhook/feedback', 
                           json={'test': True}, timeout=5)
    print(f"✅ N8N Status: {response.status_code}")
except:
    print("❌ N8N nicht erreichbar")
```

### Log-Datei Monitoring
```bash
# Überwache neue Einträge
tail -f ~/.n8n/2026-03-19_feedback.jsonl

# Zähle tägliche Einträge  
wc -l ~/.n8n/*_feedback.jsonl
```

## 🔄 Migration & Backup

### Backup erstellen
```bash
# Alle N8N Daten sichern
tar -czf n8n_backup_$(date +%Y-%m-%d).tar.gz ~/.n8n/

# Nur Feedback-Daten
tar -czf feedback_backup.tar.gz ~/.n8n/*_feedback.jsonl ~/.n8n/critical_feedback.jsonl
```

### Migration zu anderer Datenbank
```python
# N8N → PostgreSQL Migration
import json
import psycopg2

conn = psycopg2.connect("dbname=feedback user=postgres")
cur = conn.cursor()

with open('2026-03-19_feedback.jsonl') as f:
    for line in f:
        session = json.loads(line)
        cur.execute("""
            INSERT INTO sessions (session_id, session_data) 
            VALUES (%s, %s)
        """, (session['session_id'], json.dumps(session)))

conn.commit()
```

## 🎯 Fazit

Mit N8N bekommst du:
- ✅ **Vollständige Kontrolle** über deine Feedback-Daten
- ✅ **Flexible Verarbeitung** (Email-Alerts, Slack, etc.)
- ✅ **Kostenlos** bei Self-Hosting
- ✅ **Einfache JSON-Dateien** für Auswertungen
- ✅ **Skalierbar** für Millionen von Events

Die Daten sind sicher auf deinem Server und du kannst sie beliebig weiterverarbeiten!