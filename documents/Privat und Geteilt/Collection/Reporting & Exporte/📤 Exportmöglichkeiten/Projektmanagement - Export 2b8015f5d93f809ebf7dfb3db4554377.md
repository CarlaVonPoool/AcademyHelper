# Projektmanagement - Export

Article Description: Exporte aller Projekte in Poool.
Last Updated: 27. November 2025
Published: Yes
Suggested: No
Neues UI 2026: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## 🎯 Was ist der Projektmanagement-Export?

Der Projektmanagement-Export ermöglicht Ihnen, alle Projektstrukturelemente als Excel-Datei (.xlsx) herunterzuladen. Die Liste enthält Ihre **Projekte, Phasen, Tickets und Aufgaben** mit allen relevanten Daten – ideal für Reporting, Analysen oder die Weiterverarbeitung in externen Tools.

**Wichtig:** Alle Werte in diesem Export basieren auf der **Kundensicht** (Verkaufspreise/Kundenstundensätze). Eine interne Kostenbetrachtung ist hier nicht enthalten.

## ✨ Vorteile

- **Komplette Projektübersicht:** Alle Hierarchieebenen (Projekt → Phase → Ticket → Aufgabe) in einer Datei
- **Flexible Filterung:** Export basiert auf Ihren aktuellen Filtereinstellungen
- **Detailgrad wählbar:** Von der reinen Projektübersicht bis zur vollständigen Ticket-Analyse
- **Sofort einsatzbereit:** Excel-Format für direkte Weiterverarbeitung

## 🔧 Grundlegende Bedienung

### So erstellen Sie einen Export:

1. Navigieren Sie zu **Projekte → Projektmanagement**
2. Setzen Sie bei Bedarf Filter (z.B. Status, Kunde, Zeitraum)
3. Klicken Sie auf **→ Export (xlsx)**
4. Optional: Klicken Sie auf das **Zahnrad-Symbol** für erweiterte Optionen
5. Der Download startet automatisch

<aside>
💡

**Tipp:** Der Export enthält nur die Projekte, die Ihren aktuellen Filtereinstellungen entsprechen. Prüfen Sie vor dem Export, ob die richtigen Filter gesetzt sind.

</aside>

![image.png](Projektmanagement%20-%20Export/image.png)

## ⚙️ Export-Optionen

Über das Zahnrad-Symbol können Sie den Detailgrad des Exports steuern:

| Option | Beschreibung | Wann verwenden? |
| --- | --- | --- |
| Nur Projektebene exportieren | Export enthält nur eine Zeile pro Projekt – ohne Phasen, Tickets und Aufgaben | Für schnelle Projektübersichten und Management-Reports |
| Abgeschlossene Tickets exportieren | Nimmt auch abgeschlossene Tickets in den Export auf | Für vollständige Projekthistorie und Nachkalkulation |
| Abgeschlossene Phasen exportieren | Nimmt auch abgeschlossene Phasen in den Export auf | Für vollständige Projekthistorie und Nachkalkulation |

**Standard-Verhalten:** Ohne aktivierte Optionen erhalten Sie alle Hierarchieebenen, aber abgeschlossene Tickets und Phasen werden ausgeblendet.

## 💼 Anwendungsfälle

### Szenario 1: Projekt-Statusbericht für die Geschäftsleitung

Sie brauchen einen schnellen Überblick über alle laufenden Projekte.

**Vorgehen:** Filter auf "In Arbeit" setzen → Option "Nur Projektebene exportieren" aktivieren → Export erstellen

### Szenario 2: Detaillierte Projektanalyse

Sie wollen sehen, wo ein Projekt zeitlich steht und welche Tickets das Budget überschreiten.

**Vorgehen:** Auf ein Projekt filtern → Alle Optionen aktivieren → Export erstellen → In Excel nach "Diff Std/Ktg" sortieren

### Szenario 3: Nachkalkulation abgeschlossener Projekte

Sie wollen prüfen, ob abgerechnete Leistungen mit dem tatsächlichen Aufwand übereinstimmen.

**Vorgehen:** Filter auf abgeschlossene Projekte → Beide Optionen für abgeschlossene Elemente aktivieren → Export erstellen → Spalte "Diff Ist/Soll" analysieren

## 📋 Exportierte Felder

| Feld | Beschreibung |
| --- | --- |
| Kunde | Name des zugeordneten Kunden |
| Kontakt | Ansprechpartner beim Kunden |
| Art | Hierarchische Ebene (Projekt, Phase, Ticket oder Aufgabe) |
| Nummer | Eindeutige Nummer des Elements (z.B. ACEC-01, T-0235) |
| Projekt - Phase / Ticket / Aufgabe | Bezeichnung des Elements |
| Team | Zugeordnetes Team |
| Status | Aktueller Status des Elements |
| Projekttyp | Art des Projekts |
| Projektstart | Geplantes Startdatum |
| Termin | Geplanter Endtermin |
| Verantwortlich | Zuständige Person |
| Gesamtstunden | Alle gebuchten Stunden inkl. aller Unterebenen |
| Stunden | Gebuchte Stunden nur auf dieser Ebene (ohne Unterebenen) |
| Kontingent | Vom Projektmanager veranschlagte Stunden |
| Diff Std/Ktg | Differenz zwischen gebuchten Stunden und Kontingent (positiv = Überschreitung) |
| Ticketkosten | Gebuchte Stunden × Kundenstundensatz |
| Ticketwert | Beauftragter Wert des Tickets |
| Diff Ticketwert/Kosten | Ticketwert minus Ticketkosten (positiv = Budget noch verfügbar) |
| Angebot | Angebotssumme in € |
| Beauftragt | Vom Kunden beauftragte Summe in € |
| Abgerechnet | Bereits an den Kunden abgerechnete Summe in € |
| Kosten | Gebuchte Eingangsrechnungen (Fremdkosten) |
| Diff Ist/Soll | Abgerechnet minus (Gesamtstunden × Kundenstundensatz) minus Eingangsrechnungen |
| Tags | Zugeordnete Schlagwörter |
| Beschreibung | Beschreibungstext des Elements |
| Kommentar | Kommentar zum Element |

![image.png](Projektmanagement%20-%20Export/image%201.png)

<aside>
⚠️

## Wichtige Hinweise

- **Berechtigung erforderlich:** Sie benötigen die entsprechende Berechtigung, um Exporte erstellen zu können.
- **Kundensicht:** Alle finanziellen Werte basieren auf Kundenstundensätzen – nicht auf internen Kosten.
- **Keine DB2-Betrachtung:** Der Export enthält keine internen Kostensätze oder Deckungsbeitragsanalyse. Sollte eine DB2-Betrachtung gewünscht sein, nutzen Sie den [Export Projektauswertung].
- **Filter beachten:** Der Export enthält nur Daten, die Ihren aktuellen Filtereinstellungen entsprechen.
</aside>