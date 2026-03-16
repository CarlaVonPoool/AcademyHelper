# Angebots - Export

Article Description: Export aller Angebote als Excel-Tabelle oder ZIP mit PDF-Dokumenten.
Last Updated: 7. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: Yes

## 📊 Artikel-Metadaten

- **Artikel-Typ:** Feature-Überblick
- **Verfügbarkeit:** Alle Pläne
- **Bereich:** Buchhaltung → Export → Angebote
- **Letztes Update:** Januar 2026

## 🎯 Was ist der Angebots-Export?

Der Angebots-Export ermöglicht den Download aller Angebote eines bestimmten Zeitraums – entweder als strukturierte Excel-Tabelle (XLSX) für Auswertungen oder als ZIP-Archiv mit den Original-PDF-Dokumenten für die Ablage.

## ✨ Hauptvorteile

- **Flexibles Format:** Wählen Sie zwischen Datentabelle (XLSX) oder Originaldokumenten (PDF-ZIP)
- **Zeitraumfilter:** Exportieren Sie gezielt Angebote nach Angebotsdatum
- **Vollständige Daten:** Alle Kundendaten, Projektbezüge und Beträge inkl. Steueraufschlüsselung

## 🔧 Export-Varianten

| Variante | Format | Inhalt | Typischer Anwendungsfall |
| --- | --- | --- | --- |
| Angebote | XLSX | Strukturierte Datentabelle mit allen Feldern | Auswertungen, Reporting, Weiterverarbeitung |
| Dokumente | ZIP | Original-PDF-Dateien der Angebote | Archivierung, Weitergabe an Steuerberater |

## 🚀 So erstellen Sie den Export

1. Navigieren Sie zu **Buchhaltung → Export**
2. Wählen Sie den gewünschten **Zeitraum** (Filterung nach Angebotsdatum)
3. Klicken Sie auf **Angebote** für die XLSX-Datei oder auf **Dokumente** in der Angebots-Zeile für das PDF-Archiv

![image.png](Angebots%20-%20Export/image.png)

<aside>
⚠️

## Wichtige Hinweise

- Der Datumsfilter bezieht sich auf das **Angebotsdatum**, nicht auf das Erstellungsdatum im System.
- Der Angebots-Export enthält nur die **Gesamtsummen** je Angebot. Benötigen Sie eine detaillierte Positionsübersicht mit allen Einzelpositionen eines Angebots, nutzen Sie stattdessen den **Kalkulationspositions-Export**.
- Die Umsatzsteuer-Spalten werden dynamisch basierend auf den im System konfigurierten Steuersätzen generiert.
</aside>

## 💼 Anwendungsfälle

### Szenario 1: Quartalsauswertung erstellen

Sie möchten analysieren, welche Angebote in Q1 erstellt wurden und wie hoch das Angebotsvolumen war. Exportieren Sie alle Angebote des Zeitraums als XLSX und werten Sie Summe netto nach Kunde oder Projekt aus.

### Szenario 2: Unterlagen für Steuerberater

Ihr Steuerberater benötigt alle Angebotsbelege eines Jahres. Exportieren Sie die Dokumente als ZIP und übergeben Sie die gesammelten PDFs.

## 📋 Enthaltene Felder (XLSX)

### Kundenbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Kundennummer | Eindeutige Kennung des Kunden |
| Kunde | Name des Kunden |
| Kundenverantwortlich | Zuständige Person für den Kunden |

### Projektbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Projekttyp | Kategorisierung des Projekts |
| Projektnummer | Eindeutige Projektkennung |
| Projekt | Projektbezeichnung |
| Team | Zugeordnetes Team |
| Projektverantwortlich | Zuständige Projektleitung |
| Phase | Aktuelle Projektphase |

### Angebotsbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Angebotstitel | Bezeichnung des Angebots |
| Angebotsnummer | Eindeutige Angebotsnummer |
| Angebotsdatum | Datum der Angebotserstellung (Filterkriterium) |
| Status | Aktueller Angebotsstatus (z.B. Beauftragt) |
| Tags | Zugewiesene Schlagwörter |
| Angebotsverantwortlich | Zuständige Person |

### Beträge

| Feld | Beschreibung |
| --- | --- |
| Summe netto | Gesamtbetrag ohne Umsatzsteuer |
| Summe brutto | Gesamtbetrag inkl. Umsatzsteuer |
| Ust. [Steuersatz] | Aufschlüsselung nach konfigurierten Steuersätzen |

## 📚 Verwandte Artikel

### Weitere Buchhaltungs-Exporte

[Auftrags - Export](Auftrags%20-%20Export%202e1015f5d93f81a28c68d570d758bdb1.md)

- Ausgangsrechnungs-Export
- Eingangsrechnungs-Export
- Kalkulationspositions-Export (für detaillierte Positionsübersicht)