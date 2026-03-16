# Häufig gestellte Fragen (FAQ)

## Wie funktioniert die Dokumentensuche?

Der RAG (Retrieval Augmented Generation) Assistant durchsucht Ihre Markdown-Dokumente und findet relevante Abschnitte zu Ihrer Frage.

### Schritte der Suche:
1. **Text-Verarbeitung**: Dokumente werden in Chunks aufgeteilt
2. **Embedding-Erstellung**: Jeder Chunk wird in einen Vektor umgewandelt
3. **Similarity Search**: Ähnlichste Chunks zur Frage werden gefunden
4. **Antwort-Generierung**: Claude erstellt eine Antwort basierend auf den Chunks

## Welche Dateiformate werden unterstützt?

Derzeit werden nur **Markdown (.md) Dateien** unterstützt.

## Wie kann ich die Suche verbessern?

- Verwenden Sie **spezifische Fragen**
- **Schlüsselwörter** aus den Dokumenten verwenden
- Fragen in **vollständigen Sätzen** stellen

## Gibt es Limits?

- Maximale Chunk-Größe: 1500 Zeichen
- Overlap zwischen Chunks: 400 Zeichen
- Top Suchergebnisse: 40 Chunks
- Claude Token-Limit: 4096 Token für Antworten

## Kann ich eigene Dokumente verwenden?

Ja! Einfach:
1. Ihre .md Dateien in einen Ordner legen
2. Ordner-Pfad in der Sidebar eingeben
3. "Dokumente laden und indexieren" klicken