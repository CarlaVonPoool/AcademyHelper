# Stundenmanagement - Export

Article Description: Export aller gebuchten Stunden und Mitarbeiter- und Projektdetails.
Last Updated: 14. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## 🎯 Was ist der Stundenmanagement-Export?

Der Stundenmanagement-Export ermöglicht Ihnen, alle gebuchten Projektstunden als Excel-Datei (.xlsx) herunterzuladen. Jede Zeitbuchung wird dabei als einzelne Zeile exportiert – mit allen relevanten Informationen zu Projekt, Ticket, Mitarbeiter und Kosten.

**Besonderheit:** Sie haben die Wahl zwischen zwei Export-Varianten:

- **Kunde (xlsx):** Für Kunden-Reportings – ohne interne Kommentare, ohne interne Kosten
- **Intern (xlsx):** Für internes Controlling – mit allen Details inkl. interner Stundensätze und Abrechnungsinformationen

## ✨ Vorteile

- **Jede Buchung einzeln:** Detaillierte Auswertung auf Buchungsebene statt nur Summen
- **Zwei Perspektiven:** Kunden-Export für externe Kommunikation, Intern-Export für Controlling
- **Flexible Filterung:** Export basiert auf Ihren aktuellen Filtereinstellungen (Zeitraum, Team, Projekt, etc.)
- **Vollständige Zuordnung:** Von der Buchung bis zum Kunden – alle Hierarchieebenen in einer Zeile

## 🔧 Grundlegende Bedienung

### So erstellen Sie einen Export:

1. Navigieren Sie zu **Buchhaltung → Stundenmanagement**
2. Setzen Sie bei Bedarf Filter (Zeitraum, Team, Projekt, etc.)
3. **Wichtig:** Stellen Sie die Gruppierung auf **"Keine Gruppierung"**
4. Wählen Sie **→ Kunde (xlsx)** oder **→ Intern (xlsx)**
5. Der Download startet automatisch

<aside>
⚠️

**Wichtig:** Die Export-Buttons erscheinen nur, wenn **"Keine Gruppierung"** ausgewählt ist. Bei aktivierter Gruppierung sind die Buttons nicht sichtbar.

</aside>

<aside>
💡

**Tipp:** Der Export enthält nur die Zeitbuchungen, die Ihren aktuellen Filtereinstellungen entsprechen. Prüfen Sie vor dem Export, ob der richtige Zeitraum und die gewünschten Filter gesetzt sind.

</aside>

![image.png](Stundenmanagement%20-%20Export/image.png)

## ⚙️ Export-Varianten im Vergleich

| Variante | Enthält | Nicht enthalten | Typischer Einsatz |
| --- | --- | --- | --- |
| **Kunde (xlsx)** | Projektdaten, Stunden, Kosten (Kundensicht) | Interne Kommentare, interne Stundensätze, Abrechnungsinfos | Kunden-Reporting, Leistungsnachweise |
| **Intern (xlsx)** | Alle Daten inkl. interner Stundensätze und Abrechnungsstatus | – | Controlling, Nachkalkulation, interne Analyse |

## 💼 Anwendungsfälle

### Szenario 1: Leistungsnachweis für Kunden

Sie möchten einem Kunden zeigen, welche Stunden auf sein Projekt gebucht wurden.

**Vorgehen:** Filter auf Kunde und Zeitraum setzen → **Kunde (xlsx)** exportieren → Bei Bedarf in Excel formatieren und versenden

### Szenario 2: Monatliche Stundenauswertung fürs Controlling

Sie wollen analysieren, wie profitabel die gebuchten Stunden im letzten Monat waren.

**Vorgehen:** Zeitraum auf letzten Monat setzen → **Intern (xlsx)** exportieren → Spalten "Interne Stundenkosten" und "Soll Honorar" vergleichen

### Szenario 3: Team-Auslastung analysieren

Sie wollen sehen, wie viele Stunden ein bestimmtes Team auf welche Projekte gebucht hat.

**Vorgehen:** Filter auf Team setzen → **Intern (xlsx)** exportieren → In Excel Pivot-Tabelle nach Projekt und Mitarbeiter erstellen

### Szenario 4: Noch nicht abgerechnete Stunden identifizieren

Sie wollen prüfen, welche freigegebenen Stunden noch nicht abgerechnet wurden.

**Vorgehen:** Filter auf "Freigegeben" setzen → **Intern (xlsx)** exportieren → Nach Spalte "Abgerechnet" filtern (Wert = leer oder 0)

## 📋 Exportierte Felder – Kunden-Export (19 Felder)

| Feld | Beschreibung |
| --- | --- |
| Kunde | Name des zugeordneten Kunden |
| Projektnummer | Eindeutige Projektnummer |
| Projekt | Projektbezeichnung |
| Team (Projekt) | Dem Projekt zugeordnetes Team |
| Phase | Projektphase der Buchung |
| Auftragsnummer | Nummer des zugehörigen Auftrags |
| Auftrag | Bezeichnung des Auftrags |
| Ticketnummer | Nummer des Tickets |
| Ticketstatus | Aktueller Status des Tickets |
| Ticket | Bezeichnung des Tickets |
| Ticket Team | Dem Ticket zugeordnetes Team |
| Aufgabe | Aufgabe innerhalb des Tickets (falls vorhanden) |
| Mitarbeiter | Name der buchenden Person |
| Team (Mitarbeiter) | Team des Mitarbeiters |
| Funktion | Leistungsart/Funktion der Buchung |
| Kommentar | Kommentar zur Zeitbuchung (nur Kunden-Kommentar) |
| Datum | Datum der Zeitbuchung |
| Stunden | Gebuchte Stundenzahl |
| Kosten | Stunden × Kundenstundensatz |

## 📋 Exportierte Felder – Intern-Export (36 Felder)

Der Intern-Export enthält alle Felder des Kunden-Exports plus folgende Zusatzinformationen:

### Zusätzliche Organisationsfelder

| Feld | Beschreibung |
| --- | --- |
| Mandant | Zugeordneter Mandant |
| Mandant Kurz | Kurzbezeichnung des Mandanten |
| Projekttyp | Art des Projekts |
| Auftragsstatus | Status des Auftrags |

### Kommentare

| Feld | Beschreibung |
| --- | --- |
| Kommentar Kunde | Für den Kunden sichtbarer Kommentar |
| Kommentar intern | Nur intern sichtbarer Kommentar |

### Interne Kostensätze

| Feld | Beschreibung |
| --- | --- |
| Interne Stundenkosten | Stunden × interner Stundenkostensatz (echte Kosten) |
| Interner Stundenkostensatz | Der beim Mitarbeiter hinterlegte interne Kostensatz |
| Externe Stundenkosten | Stunden × externer Stundenkostensatz |
| Externer Stundenkostensatz | Der externe Kostensatz |

### Honorar und Abrechnung

| Feld | Beschreibung |
| --- | --- |
| Soll Honorar | Stunden × Kundenstundensatz (was der Kunde zahlen sollte) |
| Soll Honorar Stundensatz | Der verwendete Kundenstundensatz |
| Ticketwert | Beauftragter Wert des Tickets |
| Abrechnung | Zugeordnete Abrechnungsnummer |
| Abgerechnet | Bereits abgerechneter Betrag |

### Status und Freigabe

| Feld | Beschreibung |
| --- | --- |
| Archiviert | Ist die Buchung archiviert? (Ja/Nein) |
| Archiv | Name des Archivs (falls archiviert) |
| Überstunde | Handelt es sich um eine Überstunde? |
| Freigabestatus | Status der Stundenfreigabe |
| Freigabedatum | Datum der Freigabe |

<aside>
💡

**Tipp für Controlling:** Die Differenz zwischen "Soll Honorar" (was der Kunde zahlen sollte) und "Interne Stundenkosten" (echte Kosten) zeigt Ihnen den Deckungsbeitrag pro Buchung.

</aside>

<aside>
⚠️

## Wichtige Hinweise

- **Berechtigung erforderlich:** Sie benötigen die entsprechende Export-Berechtigung im Stundenmanagement.
- **Gruppierung beachten:** Export-Buttons sind nur bei "Keine Gruppierung" sichtbar.
- **Filter wirken:** Der Export enthält nur Daten entsprechend Ihrer aktuellen Filtereinstellungen.
- **Datenschutz:** Der Intern-Export enthält sensible Kostendaten – nur für berechtigte Personen freigeben.
</aside>

## 📚 Verwandte Artikel

### Grundlagen

[Stundenmanagement](../../Zeiterfassung/Zeiterfassung/Stundenmanagement%20f601b175a546410998d5867f3d9cd0d4.md)

### Weitere Exporte

[Projektauswertung - Export](Projektauswertung%20-%20Export%202c3015f5d93f81bfbf9dee8e38ba31f4.md)

[Exporte im Überblick](Exporte%20im%20%C3%9Cberblick%202b7015f5d93f801092a6ddc9d5488ab1.md)