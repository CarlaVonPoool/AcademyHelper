# Prism Tabellen Übersicht

Last Updated: 17. Dezember 2025
Published: Yes
Suggested: No

## 📊 Alle PRISM-Tabellen im Überblick

<aside>
ℹ️

**Hinweis zur Aktualität:** Die PRISM-Datenbank entwickelt sich kontinuierlich mit der Poool-Software weiter. Neue Felder und Tabellen können hinzugefügt werden, bevor sie in dieser Dokumentation erfasst sind. Diese Dokumentation wird regelmäßig aktualisiert und bildet den aktuellen Stand der Kernfelder ab.

</aside>

## 📌 Grundlegende Informationen

Diese Konzepte sind für mehrere PRISM-Tabellen relevant und werden hier zentral erklärt.

### Mitarbeiter-Team (Stammteam)

Das Feld `data_staff_team` (bzw. `data_time_staff_team` in Zeiterfassungstabellen) zeigt immer das **Stammteam** aus Poool an.

**Warum ist das wichtig?**

Ein:e Mitarbeiter:in kann in Poool mehreren Teams zugeordnet sein (z.B. Design, Marketing, Development). In PRISM erscheint jedoch nur das Team, das in Poool als **Stammteam** hinterlegt ist.

**Wo wird das eingestellt?**

Kontakte → Mitarbeiter → [Mitarbeiter auswählen] → Team → Stammteam aktivieren

<aside>
💡

**Tipp:** Falls ein:e Mitarbeiter:in kein Stammteam hat, bleibt das Feld leer.

</aside>

---

✅ **Stand: 17. Dezember 2025.**
Alle 10 PRISM-Tabellen sind dokumentiert!

| Nr. | Tabelle | Beschreibung | Spalten |
| --- | --- | --- | --- |
| 1 | **contacts** | Alle Kontakte (Unternehmen, Personen, Mitarbeiter:innen, Benutzer:innen) | ~60 |
| 2 | **projects** | Alle Projekte und Projektphasen mit Status und Teams | ~70 |
| 3 | **tickets** | Alle Tickets und Jobs aus dem Aufgabensystem | ~80 |
| 4 | **timetrack_times** | Einzelne Zeitbuchungen aus der Zeiterfassung | ~75 |
| 5 | **timetrack_days** | Tagesaggregierte Zeiterfassungsdaten | ~65 |
| 6 | **timetrack_working_time_accounts** | Arbeitszeitkonten der Mitarbeiter:innen | ~45 |
| 7 | **meta_dates** | Kalenderdaten für Zeitanalysen | ~15 |
| 8 | **calculation_positions** | Alle Kalkulationspositionen (Angebote, Aufträge, Rechnungen) | 95 |
| 9 | **accounts_payable** | Eingangsrechnungspositionen mit Projektzuordnung | 101 |
| 10 | **calculation_report** | Umsatz- und Kostenplanung für Quartalsauswertungen | 71 |

📚 **Dokumentationsstandard:** Jede Tabelle enthält System-Felder, Meta-Informationen, thematisch gruppierte Felder mit Datentypen, Beschreibungen, SQL-Beispiele und Analyse-Ideen.

---

# **Contacts Tabelle**

Die Contacts-Tabelle ist das zentrale CRM-Herzstück von Poool und enthält **alle Kontaktdaten** aus den folgenden Bereichen:

- **Unternehmen** (Kunden und Lieferanten) mit Firmendaten, Kunden-/Lieferantennummern und DATEV-Konten
- **Personen** (Kontaktpersonen bei Kunden/Lieferanten) mit Namen, Positionen und Abteilungen
- **Mitarbeiter:innen** mit Teamzuordnungen und Stundensätzen (intern, extern, Kunde)
- **Benutzer:innen** mit E-Mail-Adressen und Aktiv-Status

[Contacts Tabelle](../Tabellen/Contacts%20Tabelle%202a8015f5d93f804fbafecbb830edc6b7.md)

# Projects Tabelle

Die Projects-Tabelle enthält alle **Projektstammdaten** und **Projektphasen** aus Poool. Hier findest du:

- **Projekte** mit Projektnummern, Titel, Status, Team-Zuordnungen und Deadlines
- **Projektphasen** mit eigenen Status, Titeln und Terminen
- **Projekt-Teams** und Verantwortlichkeiten
- **Zeiterfassungs-Berechtigungen** und Sperren pro Projekt/Phase

<aside>
⚠️

**Wichtig:** Die Tabelle enthält sowohl Projekte als auch Projektphasen. Unterscheidung über das Feld `data_type` = `'project'` (Projekt, Hauptprojekt, Teilprojekt) oder `'projectPhase'` (Projektphase).

</aside>

[Projects Tabelle](../Tabellen/Projects%20Tabelle%202a8015f5d93f80be8dbdfba075cc417b.md)

# **Tickets Tabelle**

Die Tickets-Tabelle enthält alle **Tickets und Jobs** aus dem Poool-Aufgabensystem. Hier findest du:

- **Tickets** mit Nummern, Titeln, Status, Prioritäten und Deadlines
- **Ticket-Jobs** (Aufgaben im Ticket) mit eigenen Status und Terminen
- **Personen-Zuordnungen** (Ersteller, Zugewiesene, Kundenverantwortliche)
- **Zeitkontingente** und Wertangaben pro Ticket
- **Projekt- und Phasen-Verknüpfungen**

<aside>
⚠️

**Wichtig:** Die Tabelle enthält sowohl Tickets als auch Aufgaben. Unterscheidung über das Feld `data_type` = `'ticket'` (Ticket) oder `'ticketJob'` (Aufgabe im Ticket).

</aside>

[Tickets Tabelle](../Tabellen/Tickets%20Tabelle%202a9015f5d93f81569822c33538ee85cb.md)

# **Timetrack Times**

Die **Timetrack Times** Tabelle enthält alle **einzelnen Zeitbuchungen** aus der Poool-Zeiterfassung. Hier findest du:

- **Detaillierte Zeitbuchungen** mit Datum, Dauer (in Minuten), Überstunden-Kennzeichnung
- **Projekt-, Ticket- und Auftrags-Zuordnungen** jeder Buchung
- **Aktivitäten und Kostensätze** (intern und extern)
- **Kosten-Informationen** pro Buchung (intern, Kunde, Ticket)
- **Freigabe- und Abrechnungs-Status** für Controlling und Rechnungsstellung
- **Kommentare** (intern und kundenorientiert)
- **Archivierungs- und Sperr-Status**

<aside>
⚠️

**Wichtig:**

- Zeitwerte sind in **Minuten** gespeichert (nicht Stunden!)
- Eine Zeitbuchung kann Projekt, Ticket oder beides zugeordnet sein
- Archivierte oder gesperrte Buchungen können nicht mehr bearbeitet werden
</aside>

[Timetrack Times Tabelle](../Tabellen/Timetrack%20Times%20Tabelle%202a9015f5d93f812cb0e7fa2dfe805a8d.md)

# **Timetrack Days**

Die **Timetrack Days** Tabelle enthält **tagesweise aggregierte Zeiterfassung** für alle Mitarbeiter:innen. Hier findest du:

- **Tages-Übersichten** mit Soll- und Ist-Arbeitszeiten
- **Gleitzeit-Änderungen** pro Tag (Plus-/Minusstunden)
- **Zeittypen-Summen** (Regulär, Urlaub, Krank, Freizeitausgleich, Intern)
- **Arbeitszeit-Soll** (Target: von, bis, Pause, Netto)
- **Arbeitszeit-Ist** (tatsächliche Zeiten)
- **Feiertage und Standort-Informationen**
- **Arbeitszeitkonto-Modifikationen** (Überstunden, Urlaub, etc.)
- **Tag-Sperrungen** und Validierung

<aside>
⚠️

**Wichtig:**

- Alle Zeitwerte sind in **Minuten** gespeichert (nicht Stunden!)
- Ein Datensatz = Ein Tag für eine:n Mitarbeiter:in
- Gesperrte Tage (`data_timetrack_locked = true`) können nicht mehr bearbeitet werden
</aside>

[Timetrack Days Tabelle](../Tabellen/Timetrack%20Days%20Tabelle%2029c3c9f5d4ed4871825d7955f5f5c53d.md)

# **Timetrack Working Time Accounts**

Die **Timetrack Working Time Accounts** Tabelle enthält alle **Arbeitszeitkonten** der Mitarbeiter:innen pro Jahr. Hier findest du:

- **Jahresbezogene Arbeitszeitkonten** mit Urlaubstagen, Überstunden und Gleitzeilständen
- **Urlaubsverwaltung**: Initial-Anspruch, Übertrag, Verbrauch und Modifikationen
- **Überstunden-Tracking**: Aufbau, Übertrag und Abbau von Überstunden
- **Gleitzeit-Konten**: Aktuelle Stände und Flextime-Verwaltung
- **Freizeitausgleich**: Verbrauchte Ausgleichstage und -stunden
- **Sonderurlaub**: Spezielle Urlaubsformen

⚠️ **Wichtig:** Pro Mitarbeiter:in und Jahr gibt es genau einen Datensatz. Die Tabelle wird automatisch aktualisiert basierend auf Zeiterfassungen und manuellen Modifikationen.

[Timetrack Working Time Accounts](../Tabellen/Timetrack%20Working%20Time%20Accounts%202a9015f5d93f81deb295cf677964debe.md)

# Meta Dates Tabelle

Die **Meta Dates** Tabelle ist eine zentrale Datumsdimensionstabelle für zeitbasierte Analysen und Reports. Sie enthält für jeden Tag umfassende Kalenderinformationen (Jahr, Monat, Quartal, Kalenderwoche, Wochentag, Schaltjahr, etc.) und ermöglicht flexible Zeitauswertungen in Power BI und Excel.

**Typische Anwendungsfälle:**

- Zeiterfassungs-Auswertungen nach Perioden (Woche, Monat, Quartal)
- Projektplanung mit Arbeitstagen vs. Wochenenden
- Quartalsbezogene Finanzreports
- Jahresvergleiche und Trendanalysen

<aside>
⚠️

**Wichtig:** Die Tabelle sollte einen ausreichenden Datumsbereich abdecken (mindestens aktuelles Jahr + nächstes Jahr). Bei Bedarf muss der Datenbereich erweitert werden.

</aside>

[Meta Dates](../Tabellen/Meta%20Dates%202a9015f5d93f8119be03ea3c90e6eaf5.md)

# Calculation Positions Tabelle

Die **Calculation Positions** Tabelle enthält **alle Kalkulationspositionen** aus Poool. Hier findest du:

- **Angebotspositionen** mit Mengen, Preisen und Rabatten
- **Auftragspositionen** mit Bestellnummern und Status
- **Rechnungspositionen** mit Verkaufspreisen und Nebenkostenzuschlägen
- **Kunden- und Projektdaten** pro Position
- **Kalkulationsgruppen** (Eigen-/Fremdleistungen, optional/fest)
- **Leistungen und Materialien** aus Preislisten

<aside>
⚠️

**Wichtig:** Jede Zeile = Eine Kalkulationsposition. Eine Kalkulation (Angebot/Auftrag/Rechnung) besteht aus mehreren Positionen. Die Tabelle ist ähnlich aufgebaut wie der Export unter `Buchhaltung > Export > Kalkulationspositionen`.

</aside>

[Calculation Positions Tabelle](../Tabellen/Calculation%20Positions%20Tabelle%202a9015f5d93f81039eebf0dbfcd11028.md)

# Accounts Payable Tabelle

Die **Accounts Payable** Tabelle enthält alle **Eingangsrechnungspositionen** mit vollständigem Kontext. Hier findest du:

- **Fremdkosten** über Eingangsrechnungen erfasst
- **Projektzuordnung** für jede Position
- **Auftragsbezug** (Verknüpfung zu Bestellungen)
- **Budgetkontrolle** mit Budget- und Kostenstellen-Zuordnung
- **Abrechnungsstatus** (Clearable/Cleared) für Weiterverrechnung
- **Lieferantendaten** und Zahlungsinformationen

<aside>
⚠️

**Wichtig:** Jede Zeile = Eine Eingangsrechnungsposition. Rechnungsdaten (wie Rechnungssummen) wiederholen sich bei mehreren Positionen derselben Rechnung. Beim Aggregieren nicht summieren, sondern MAX() oder FIRST() verwenden!

</aside>

[accounts_payable](../Tabellen/accounts_payable%207d4811fcd9ce472d8a555ae83b9af6ca.md)

# Calculation Report Tabelle

Die **Calculation Report** Tabelle bildet die Grundlage für die **Quartalsauswertung** und enthält Umsatz- und Kostenplanungsdaten. Hier findest du:

- **Offene Auftragsplanungspositionen** (Orders)
- **Ausgangsrechnungen** (Invoices)
- **Lead-Planungen** mit Gewichtung
- **Wiederkehrende Ausgangsrechnungen** (zukünftige Umsätze)
- **Kosten** (Eingangsrechnungen)
- **Wiederkehrende Eingangsrechnungen**
- **Gehaltsdaten** (Personalkosten)

<aside>
🔥

**Item Types:** `invoice`, `invoice_recurring`, `lead`, `order`, `salary`, `salary_extra`

**Account Types:** `receivable` (Umsätze) vs. `payable` (Kosten)

**Lead-Gewichtung:** Werte werden mit Gewichtungsprozentsatz multipliziert für realistische Planung

</aside>

[Calculation Report Tabelle](../Tabellen/Calculation%20Report%20Tabelle%20146015f5d93f8022a872ff1f4c81496f.md)

📚 **Dokumentationsstandard:** Jede Tabelle enthält System-Felder, Meta-Informationen, thematisch gruppierte Felder mit Datentypen, Beschreibungen, SQL-Beispiele und Analyse-Ideen.

---