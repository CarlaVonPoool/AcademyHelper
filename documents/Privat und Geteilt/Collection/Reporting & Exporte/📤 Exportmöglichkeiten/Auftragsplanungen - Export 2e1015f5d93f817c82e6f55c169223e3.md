# Auftragsplanungen - Export

Article Description: Export aller Aufträge mit Zahlungsmeilensteinen, Soll-Ist-Vergleich und Abrechnungsstatus.
Last Updated: 7. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: Yes

## 📊 Artikel-Metadaten

- **Artikel-Typ:** Feature-Überblick
- **Verfügbarkeit:** Benutzer mit Export-Berechtigung im Bereich Buchhaltung
- **Bereich:** Buchhaltung → Export
- **Letztes Update:** Januar 2026

## 🎯 Was ist der Auftragsplanungen-Export?

Der Auftragsplanungen-Export zeigt Ihnen alle Aufträge mit ihren geplanten Zahlungsmeilensteinen und dem aktuellen Abrechnungsstand. Jede Zeile repräsentiert einen Auftragsplan-Eintrag – so sehen Sie auf einen Blick, welche Zahlungen wann fällig sind und welche bereits abgerechnet wurden.

**Wichtig:** Der Export ist **auftragsplan-zentriert** – ein Auftrag mit mehreren Zahlungsmeilensteinen erscheint in mehreren Zeilen.

## ✨ Hauptvorteile

- **Liquiditätsplanung:** Übersicht aller geplanten Zahlungseingänge mit Fälligkeitsdaten
- **Soll-Ist-Vergleich:** Eigenleistungen und Fremdleistungen im direkten Vergleich (Soll/Ist/Differenz)
- **Abrechnungsstatus:** Sofort sehen, welche Aufträge offen oder bereits abgerechnet sind
- **Rechnungszuordnung:** Verknüpfung zwischen Auftragsplan und tatsächlicher Ausgangsrechnung
- **Umsatz-Forecast:** Gleiche Datenbasis wie die Quartalsauswertung Umsatz in Poool

## 🔧 Grundlegende Bedienung

### So erstellen Sie den Export:

1. Navigieren Sie zu **Buchhaltung → Export**
2. Wählen Sie den gewünschten **Zeitraum** (Datumsfilter oben)
3. Klicken Sie auf **Auftragsplanungen**
4. Die Excel-Datei wird heruntergeladen

![image.png](Auftragsplanungen%20-%20Export/image.png)

<aside>
⚠️

**Achtung:** Der Zeitraum-Filter bezieht sich auf das **Beauftragungsdatum** – nicht auf das geplante Abrechnungsdatum. Ein Auftrag vom März 2025 mit Zahlungsmeilensteinen bis Dezember 2025 erscheint nur, wenn März 2025 im Filterzeitraum liegt.

</aside>

## 💼 Anwendungsfälle

### Szenario 1: Liquiditätsvorschau erstellen

Sie möchten wissen, welche Zahlungseingänge in den nächsten Monaten zu erwarten sind.

**Vorgehen:** Export erstellen und nach "Auftragsplan Fälligkeit" sortieren. So sehen Sie alle geplanten Zahlungen chronologisch.

### Szenario 2: Offene Aufträge prüfen

Sie möchten alle Aufträge identifizieren, die noch nicht vollständig abgerechnet wurden.

**Vorgehen:** Export nach Status = "Offen" filtern und "Eigenleistungen Diff" prüfen – positive Werte zeigen noch abzurechnende Leistungen.

### Szenario 3: Rechnungsstellung vorbereiten

Sie möchten sehen, welche Auftragsplan-Positionen zur Abrechnung anstehen.

**Vorgehen:** Nach "Auftragsplan Fälligkeit" filtern (z.B. aktueller Monat) und Positionen ohne zugeordnete Ausgangsrechnung identifizieren.

### Szenario 4: Umsatz-Forecast erstellen

Sie möchten eine Umsatzplanung auf Basis bereits beauftragter Projekte erstellen.

**Vorgehen:** Der Export liefert die gleichen Zahlen wie die Quartalsauswertung Umsatz in Poool. Damit können Sie einen Forecast für die bereits beauftragte Umsatzplanung erstellen – ideal für Controlling und Geschäftsplanung.

<aside>
⚠️

## Wichtige Hinweise

- **Nur Aufträge:** Der Export enthält ausschließlich Aufträge – keine Skizzen oder Leads. Für eine vollständige Pipeline-Betrachtung nutzen Sie zusätzlich den Angebote-Export.
- **Gleiche Datenbasis wie Quartalsauswertung:** Die Zahlen entsprechen der Quartalsauswertung Umsatz in Poool – ideal für Umsatz-Forecasts.
- **Mehrere Zeilen pro Auftrag:** Aufträge mit mehreren Auftragsplan-Positionen erscheinen in mehreren Zeilen.
- **Leere Rechnungsfelder:** Wenn keine Ausgangsrechnung zugeordnet ist, bleiben die Rechnungsfelder leer.
- **Berechtigung erforderlich:** Sie benötigen die entsprechende Export-Berechtigung im Bereich Buchhaltung.
</aside>

## 📋 Exportierte Felder (36 Felder)

### Kundendaten

| Feld | Beschreibung | Beispielwert |
| --- | --- | --- |
| Kundennummer | Eindeutige Kundennummer | KD-0021 |
| Kunde | Name des Kunden | Gotham City Police Department |
| Kundenverantwortlich | Zuständiger Mitarbeiter für den Kunden | Unser Alfred |

### Projektdaten

| Feld | Beschreibung | Beispielwert |
| --- | --- | --- |
| Projektnummer | Eindeutige Projektnummer | GCP-21 |
| Projekt | Projektname | Image Kampagne |
| Projekttyp | Typ/Kategorie des Projekts | Scope 2024, Digital, Analog |
| Team | Zugeordnetes Team | Transformation, Project Management |
| Phase | Projektphase (falls vorhanden) | – |
| Projektverantwortlich | Zuständiger Mitarbeiter für das Projekt | Unser Alfred |

### Auftragsdaten

| Feld | Beschreibung | Beispielwert |
| --- | --- | --- |
| Auftragstitel | Bezeichnung des Auftrags | Image Kampagne |
| Auftragsnummer | Eindeutige Auftragsnummer | AF-0106 |
| Bestellnummer | Bestellnummer des Kunden (falls vorhanden) | – |
| Auftragsdatum | Datum der Auftragserteilung | 03.04.2025 |
| Status | Auftragsstatus | Offen, Abgerechnet |
| Tags | Zugewiesene Tags | – |
| Auftragsverantwortlich | Zuständiger Mitarbeiter für den Auftrag | Unser Alfred |

### Leistungen (Soll-Ist-Vergleich)

| Feld | Beschreibung | Beispielwert |
| --- | --- | --- |
| Eigenleistungen Soll | Geplante Eigenleistungen (netto) | 7.160,00 € |
| Eigenleistungen Ist | Tatsächlich erbrachte Eigenleistungen | 1.320,00 € |
| Eigenleistungen Diff | Differenz (Soll - Ist) | 5.840,00 € |
| Fremdleistungen Soll | Geplante Fremdleistungen (netto) | – |
| Fremdleistungen Ist | Tatsächlich angefallene Fremdleistungen | – |
| Fremdleistungen Diff | Differenz (Soll - Ist) | – |

### Summen

| Feld | Beschreibung | Beispielwert |
| --- | --- | --- |
| Summe Auftragsplanung netto | Gesamtsumme der Auftragsplanung | 7.160,00 € |
| Summe abgerechnet netto | Bereits abgerechneter Betrag | – |
| Summe netto | Gesamtsumme des Auftrags | 7.160,00 € |

### Auftragsplan-Details

| Feld | Beschreibung | Beispielwert |
| --- | --- | --- |
| Auftragsplan Titel | Bezeichnung des Zahlungsmeilensteins | Schiff 2, Gesamtzahlung |
| Auftragsplan Fälligkeit | Geplantes Fälligkeitsdatum | 02.06.2025 |
| Auftragsplan Eigenleistung | Eigenleistungsanteil dieses Meilensteins | 3.580,00 € |
| Auftragsplan Fremdleistung | Fremdleistungsanteil dieses Meilensteins | – |

### Ausgangsrechnung (falls vorhanden)

| Feld | Beschreibung | Beispielwert |
| --- | --- | --- |
| Ausgangsrechnung Projektnummer | Projektnummer der zugeordneten Rechnung | – |
| Ausgangsrechnung Nummer | Rechnungsnummer | – |
| Ausgangsrechnung Titel | Rechnungstitel | – |
| Ausgangsrechnung Datum | Rechnungsdatum | – |
| Ausgangsrechnung Status | Zahlungsstatus der Rechnung | Offen, Bezahlt |
| Ausgangsrechnung Summe netto anteilig | Anteiliger Rechnungsbetrag | – |
| Ausgangsrechnung Summe netto gesamt | Gesamter Rechnungsbetrag | – |

## 📚 Verwandte Artikel

### Weitere Buchhaltungs-Exporte

[Angebots - Export](Angebots%20-%20Export%202e1015f5d93f817a9133d62e6cbe6514.md)