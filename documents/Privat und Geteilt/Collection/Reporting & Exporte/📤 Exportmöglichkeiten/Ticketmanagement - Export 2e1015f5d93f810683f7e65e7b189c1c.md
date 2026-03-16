# Ticketmanagement - Export

Article Description: Export aller Tickets mit Zeitkontingenten, Verantwortlichkeiten und Kosten.
Last Updated: 14. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## 🎯 Was ist der Ticketmanagement-Export?

Der Ticketmanagement-Export ermöglicht Ihnen, alle Tickets als Excel-Datei (.xlsx) herunterzuladen. Jedes Ticket wird dabei als einzelne Zeile exportiert – mit allen relevanten Informationen zu Kunde, Projekt, Zeitkontingenten und Kosten.

**Wichtig:** Der Export zeigt eine **Ticket-zentrierte Sicht** – ideal für die Übersicht über Arbeitspakete, Zeitkontingente und deren Auslastung.

## ✨ Vorteile

- **Ticket-Überblick:** Alle Tickets mit Status, Verantwortlichkeiten und Zeitkontingenten auf einen Blick
- **Auslastungsanalyse:** Vergleich von Zeitkontingent und tatsächlich gebuchten Stunden pro Ticket
- **Flexible Filterung:** Export basiert auf Ihren aktuellen Filtereinstellungen (Kunde, Projekt, Status, etc.)
- **Vollständige Zuordnung:** Von Ticket bis Kunde – alle Hierarchieebenen in einer Zeile

## 🔧 Grundlegende Bedienung

### So erstellen Sie einen Export:

1. Navigieren Sie zu **Ticketmanagement**
2. Setzen Sie bei Bedarf Filter (Kunde, Projekt, Status, etc.)
3. Klicken Sie auf **Export (xlsx)**
4. Der Download startet automatisch

<aside>
💡

**Tipp:** Der Export enthält nur die Tickets, die Ihren aktuellen Filtereinstellungen entsprechen. Prüfen Sie vor dem Export, ob die gewünschten Filter gesetzt sind.

</aside>

## 💼 Anwendungsfälle

### Szenario 1: Ticket-Auslastung analysieren

Sie möchten sehen, welche Tickets ihr Zeitkontingent überschritten haben.

**Vorgehen:** Filter nach Bedarf setzen → Export (xlsx) → In Excel die Spalten "Zeitkontingent" und "Stunden" vergleichen

### Szenario 2: Offene Tickets pro Kunde

Sie wollen eine Übersicht aller offenen Tickets eines bestimmten Kunden erstellen.

**Vorgehen:** Filter auf Kunde und Status "Offen" setzen → Export (xlsx) → Ggf. nach Projektphase sortieren

### Szenario 3: Team-Verantwortlichkeiten prüfen

Sie möchten sehen, welche Tickets welchem Team oder welchen Verantwortlichen zugeordnet sind.

**Vorgehen:** Export (xlsx) → In Excel Pivot-Tabelle nach "Verantwortlich" oder "Ticket Team" erstellen

## 📋 Exportierte Felder (36 Felder)

### Kundeninformationen

| Feld | Beschreibung |
| --- | --- |
| Kundennummer | Eindeutige Nummer des Kunden |
| Kundenkürzel | Kurzbezeichnung des Kunden |
| Kunde | Vollständiger Kundenname |
| Kundentags | Dem Kunden zugewiesene Tags |
| Kontakt | Ansprechpartner beim Kunden |

### Projektinformationen

| Feld | Beschreibung |
| --- | --- |
| Projektnummer | Eindeutige Projektnummer |
| Projekt | Projektbezeichnung |
| Team (Projekt) | Dem Projekt zugeordnetes Team |
| Projektphase | Phase innerhalb des Projekts |
| Projektstatus | Aktueller Status des Projekts |
| Projekttags | Dem Projekt zugewiesene Tags |
| Projektstart | Startdatum des Projekts |
| Projekttermin | Zieltermin des Projekts |

### Ticketinformationen

| Feld | Beschreibung |
| --- | --- |
| Typ | Art des Tickets |
| Ticketnummer | Eindeutige Ticketnummer |
| Titel | Bezeichnung des Tickets |
| Ticket Team | Dem Ticket zugeordnetes Team |
| Beschreibung | Beschreibungstext des Tickets |
| Offen / Abgeschlossen | Workflow-Status des Tickets (System-Status) |
| Benutzerdefinierter Status | Individuell definierter Status |
| Tags | Dem Ticket zugewiesene Tags |

### Verantwortlichkeiten

| Feld | Beschreibung |
| --- | --- |
| Verantwortlich | Hauptverantwortliche Person für das Ticket |
| Team (Mitarbeiter) | Team des verantwortlichen Mitarbeiters |
| Mitverantwortlich | Weitere verantwortliche Personen |

### Termine

| Feld | Beschreibung |
| --- | --- |
| Beginn | Startdatum des Tickets |
| Termin | Fälligkeitsdatum des Tickets |

### Zeit und Kosten

| Feld | Beschreibung |
| --- | --- |
| Zeitkontingent gesamt | Geplantes Zeitkontingent inkl. Unterebenen (z.B. Aufgaben) |
| Stunden gesamt | Gebuchte Stunden inkl. Unterebenen |
| Zeitkontingent | Geplantes Zeitkontingent auf dieser Ebene |
| Stunden | Gebuchte Stunden auf dieser Ebene |
| Wert | Monetärer Wert des Tickets (Kundensicht) |
| Kosten | Kosten des Tickets |

### Auftrag und Abrechnung

| Feld | Beschreibung |
| --- | --- |
| Auftrag | Zugeordneter Auftrag |
| Abrechnung | Zugeordnete Abrechnung |

### Erstellung

| Feld | Beschreibung |
| --- | --- |
| Ersteller | Person, die das Ticket erstellt hat |
| Erstellt | Erstellungsdatum des Tickets |

<aside>
⚠️

## Wichtige Hinweise

- **Berechtigung erforderlich:** Sie benötigen die entsprechende Export-Berechtigung im Ticketmanagement.
- **Filter wirken:** Der Export enthält nur Daten entsprechend Ihrer aktuellen Filtereinstellungen.
- **Zeitkontingent vs. Stunden:** "gesamt"-Felder inkludieren Unterebenen (z.B. Aufgaben), Felder ohne "gesamt" zeigen nur die aktuelle Ebene.
</aside>