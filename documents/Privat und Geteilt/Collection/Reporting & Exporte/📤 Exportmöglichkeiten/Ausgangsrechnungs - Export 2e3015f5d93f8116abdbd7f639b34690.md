# Ausgangsrechnungs - Export

Article Description: Export aller Ausgangsrechnungen als Excel-Tabelle oder ZIP mit PDF-Dokumenten.
Last Updated: 14. Januar 2026
Published: No
Suggested: No
Neues UI 2026: Yes

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## 🎯 Was ist der Ausgangsrechnungs-Export?

Der Ausgangsrechnungs-Export ermöglicht den Download aller Ausgangsrechnungen eines bestimmten Zeitraums – entweder als strukturierte Excel-Tabelle (XLSX) für Auswertungen oder als ZIP-Archiv mit den Original-PDF-Dokumenten für die Ablage. Er liefert alle relevanten Rechnungsdaten inklusive Zahlungsstatus und Fälligkeiten.

## ✨ Vorteile

- **Flexibles Format:** Wähle zwischen Datentabelle (XLSX) oder Originaldokumenten (PDF-ZIP)
- **Zeitraumfilter:** Exportiere gezielt Rechnungen nach Rechnungsdatum
- **Zahlungsstatus:** Zeigt offene, bezahlte und überfällige Rechnungen
- **Buchhaltungsintegration:** Ideal für DATEV-Vorbereitung und Steuerberater

## 🔧 Export-Varianten

| Variante | Format | Inhalt | Typischer Anwendungsfall |
| --- | --- | --- | --- |
| Ausgangsrechnungen | XLSX | Strukturierte Datentabelle mit allen Rechnungsfeldern | Auswertungen, Buchhaltung, Reporting |
| Dokumente | ZIP | Original-PDF-Dateien der Rechnungen | Archivierung, Weitergabe an Steuerberater |

## 🚀 So erstellst du den Export

1. Navigiere zu **Buchhaltung → Export**
2. Wähle den gewünschten **Zeitraum** (Filterung nach Rechnungsdatum)
3. Klicke auf **Ausgangsrechnungen** für die XLSX-Datei oder auf **Dokumente** in der Ausgangsrechnungs-Zeile für das PDF-Archiv

![image.png](Ausgangsrechnungs%20-%20Export/image.png)

<aside>
⚠️

## Wichtiger Hinweis zum Datumsfilter

Der Zeitraumfilter bezieht sich ausschließlich auf das **Rechnungsdatum** – nicht auf den Leistungszeitraum oder das Fälligkeitsdatum!

Beispiel: Eine Rechnung vom 15.01.2026 mit Leistungszeitraum Dezember 2025 erscheint nur im Export für Januar 2026, nicht im Dezember-Export.

</aside>

<aside>
💡

## Weitere Hinweise

- Der Export enthält nur die **Gesamtsummen** je Rechnung. Für eine detaillierte Positionsübersicht nutze den **Kalkulationspositions-Export**.
- Die Umsatzsteuer-Spalten werden dynamisch basierend auf den im System konfigurierten Steuersätzen generiert.
</aside>

## 💼 Anwendungsfälle

### Szenario 1: Umsatzauswertung nach Zeitraum

Du möchtest wissen, welcher Umsatz im letzten Quartal fakturiert wurde. Exportiere alle Rechnungen des Zeitraums als XLSX und summiere die Netto-Beträge – gruppiert nach Kunde, Projekt oder Monat.

### Szenario 2: Offene Posten prüfen

Du möchtest alle offenen Rechnungen identifizieren. Filtere im Export nach Status "Offen" und prüfe die Fälligkeitsdaten für dein Mahnwesen.

### Szenario 3: Unterlagen für Steuerberater

Dein Steuerberater benötigt alle Rechnungsbelege eines Monats oder Jahres. Exportiere die Dokumente als ZIP und übergib die gesammelten PDFs zusammen mit der XLSX-Übersicht.

### Szenario 4: Umsatzsteuer-Voranmeldung vorbereiten

Für die UStVA benötigst du alle Rechnungen eines Monats. Exportiere den relevanten Zeitraum und nutze die aufgeschlüsselten USt-Spalten für deine Meldung.

## 📋 Enthaltene Felder (XLSX)

### Kundenbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Kundennummer | Eindeutige Kennung des Kunden |
| Kunde Buchhaltung | Kundenname für Buchhaltungszwecke |
| Kunde | Name des Kunden |
| Kunde offiziell | Offizieller/rechtlicher Kundenname |
| Kunden UID | Umsatzsteuer-Identifikationsnummer des Kunden |
| Kundenverantwortlich | Zuständige Person für den Kunden |

### Projektbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Projekttyp | Kategorisierung des Projekts |
| Projektnummer | Eindeutige Projektkennung |
| Projekt | Projektbezeichnung |
| Team | Zugeordnetes Team |
| Projektverantwortlich | Zuständige Projektleitung |
| Phase | Zugehörige Projektphase |

### Rechnungsbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Rechnungstitel | Bezeichnung/Betreff der Rechnung |
| Rechnungsnummer | Eindeutige Rechnungsnummer |
| Leistungszeitraum | Zeitraum der abgerechneten Leistung |
| Rechnungsdatum | Datum der Rechnung (Filterkriterium!) |
| Fälligkeit | Zahlungsziel der Rechnung |
| Rechnungsstatus zum Exporttag | Aktueller Status zum Zeitpunkt des Exports (Offen, Bezahlt, etc.) |
| Tags | Zugewiesene Schlagwörter |
| Rechnungsverantwortlich | Zuständige Person für diese Rechnung |
| Bestellnummer | Kundenbestellnummer (falls vorhanden) |

### Leistungen

| Feld | Beschreibung |
| --- | --- |
| Eigenleistungen | Interne Leistungen auf dieser Rechnung |
| Fremdleistungen | Externe/eingekaufte Leistungen auf dieser Rechnung |

### Beträge

| Feld | Beschreibung |
| --- | --- |
| Summe netto | Rechnungsbetrag ohne Umsatzsteuer |
| Summe brutto | Rechnungsbetrag inkl. Umsatzsteuer |
| Summe bezahlt | Bereits eingegangene Zahlungen |
| Summe Abzüge | Abzüge (z.B. Skonto, Gutschriften) |
| Summe offen | Noch offener Restbetrag |
| Ust. [Steuersatz] | Aufschlüsselung nach konfigurierten Steuersätzen (dynamisch) |

## 📚 Verwandte Artikel

### Weitere Buchhaltungs-Exporte

- Angebots-Export
- Auftrags-Export
- Auftragsplanungen-Export
- Eingangsrechnungs-Export
- Kalkulationspositions-Export (für detaillierte Positionsübersicht)

### Grundlagen

- Rechnungen erstellen und verwalten
- Offene Posten und Mahnwesen