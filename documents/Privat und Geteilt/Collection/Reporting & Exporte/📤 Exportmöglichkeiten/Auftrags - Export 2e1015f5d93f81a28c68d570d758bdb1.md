# Auftrags - Export

Article Description: Export aller Aufträge als Excel-Tabelle oder ZIP mit PDF-Dokumenten.
Last Updated: 7. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: Yes

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## 🎯 Was ist der Auftrags-Export?

Der Auftrags-Export ermöglicht den Download aller Aufträge eines bestimmten Zeitraums – entweder als strukturierte Excel-Tabelle (XLSX) für Auswertungen oder als ZIP-Archiv mit den Original-PDF-Dokumenten für die Ablage. Im Vergleich zum Angebots-Export enthält er zusätzlich Planwerte und Abrechnungsstände.

## ✨ Vorteile

- **Flexibles Format:** Wähle zwischen Datentabelle (XLSX) oder Originaldokumenten (PDF-ZIP)
- **Zeitraumfilter:** Exportiere gezielt Aufträge nach Auftragsdatum
- **Plan-Ist-Vergleich:** Enthält Eigen- und Fremdleistungen mit Soll-, Ist- und Differenzwerten
- **Abrechnungsstand:** Zeigt bereits abgerechnete Beträge pro Auftrag

## 🔧 Export-Varianten

| Variante | Format | Inhalt | Typischer Anwendungsfall |
| --- | --- | --- | --- |
| Aufträge | XLSX | Strukturierte Datentabelle mit allen Feldern inkl. Plan-Ist-Vergleich | Auswertungen, Controlling, Reporting |
| Dokumente | ZIP | Original-PDF-Dateien der Aufträge | Archivierung, Weitergabe an Steuerberater |

## 🚀 So erstellst du den Export

1. Navigiere zu **Buchhaltung → Export**
2. Wähle den gewünschten **Zeitraum** (Filterung nach Auftragsdatum)
3. Klicke auf **Aufträge** für die XLSX-Datei oder auf **Dokumente** in der Auftrags-Zeile für das PDF-Archiv

![image.png](Auftrags%20-%20Export/image.png)

<aside>
⚠️

## Wichtige Hinweise

- Der Datumsfilter bezieht sich auf das **Auftragsdatum**, nicht auf das Erstellungsdatum im System.
- Der Auftrags-Export enthält nur die **Gesamtsummen** je Auftrag. Benötigst du eine detaillierte Positionsübersicht mit allen Einzelpositionen eines Auftrags, nutze stattdessen den **Kalkulationspositions-Export**.
- Die **Planwerte** (Soll) stammen aus der Auftragsplanung und werden mit den tatsächlich gebuchten Werten (Ist) verglichen.
- Die Umsatzsteuer-Spalten werden dynamisch basierend auf den im System konfigurierten Steuersätzen generiert.
</aside>

## 💼 Anwendungsfälle

### Szenario 1: Auftragsvolumen analysieren

Du möchtest wissen, welche Aufträge in Q1 beauftragt wurden und wie hoch das Auftragsvolumen ist. Exportiere alle Aufträge des Zeitraums als XLSX und werte Summe netto nach Kunde oder Projekt aus.

### Szenario 2: Plan-Ist-Abgleich für Controlling

Du möchtest prüfen, welche Aufträge über oder unter Plan liegen. Nutze die Felder Eigenleistungen Diff und Fremdleistungen Diff, um Abweichungen zwischen Planung und Realität zu identifizieren.

### Szenario 3: Offene Abrechnungen identifizieren

Vergleiche Summe netto mit Summe abgerechnet netto, um Aufträge zu finden, bei denen noch Beträge zur Abrechnung ausstehen.

### Szenario 4: Unterlagen für Steuerberater

Dein Steuerberater benötigt alle Auftragsbelege eines Jahres. Exportiere die Dokumente als ZIP und übergib die gesammelten PDFs.

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

### Auftragsbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Auftragstitel | Bezeichnung des Auftrags |
| Auftragsnummer | Eindeutige Auftragsnummer |
| Bestellnummer | Kundenbestellnummer (falls vorhanden) |
| Auftragsdatum | Datum der Beauftragung (Filterkriterium) |
| Status | Aktueller Auftragsstatus (z.B. Offen, Abgeschlossen) |
| Tags | Zugewiesene Schlagwörter |
| Auftragsverantwortlich | Zuständige Person |

### Planwerte und Leistungen

| Feld | Beschreibung |
| --- | --- |
| Eigenleistungen Soll | Geplante interne Leistungen laut Auftragsplanung |
| Eigenleistungen Ist | Tatsächlich gebuchte interne Leistungen |
| Eigenleistungen Diff | Differenz zwischen Soll und Ist (positiv = unter Plan) |
| Fremdleistungen Soll | Geplante externe Leistungen laut Auftragsplanung |
| Fremdleistungen Ist | Tatsächlich gebuchte externe Leistungen |
| Fremdleistungen Diff | Differenz zwischen Soll und Ist (positiv = unter Plan) |

### Beträge

| Feld | Beschreibung |
| --- | --- |
| Summe Auftragsplanung netto | Geplanter Gesamtbetrag laut Auftragsplanung |
| Summe abgerechnet netto | Bereits in Rechnung gestellter Betrag |
| Summe netto | Gesamtbetrag des Auftrags ohne Umsatzsteuer |
| Summe brutto | Gesamtbetrag inkl. Umsatzsteuer |
| Ust. [Steuersatz] | Aufschlüsselung nach konfigurierten Steuersätzen |

## 📚 Verwandte Artikel

### Weitere Buchhaltungs-Exporte

[Angebots - Export](Angebots%20-%20Export%202e1015f5d93f817a9133d62e6cbe6514.md)

[Auftragsplanungen - Export](Auftragsplanungen%20-%20Export%202e1015f5d93f817c82e6f55c169223e3.md)