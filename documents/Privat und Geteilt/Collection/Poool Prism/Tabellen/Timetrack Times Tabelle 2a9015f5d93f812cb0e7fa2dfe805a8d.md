# Timetrack Times Tabelle

Article Description: Einzelne Zeitbuchungen mit Projekten, Aktivitäten, Kosten und Freigabe-Status
Last Updated: 12. November 2025
Published: Yes
Suggested: No

Die **Timetrack Times** Tabelle enthält alle **einzelnen Zeitbuchungen** aus der Zeiterfassung. Hier findest du:

- **Detaillierte Zeitbuchungen** mit Datum, Dauer und Überstunden-Kennzeichnung
- **Projekt- und Phasen-Zuordnungen** inkl. Status und Verantwortlichen
- **Ticket- und Job-Verknüpfungen** mit Werten und Status
- **Aktivitäten und Kostensätze** (intern, extern, Ticket)
- **Freigabe-Workflow** (genehmigt, offen, abgelehnt)
- **Abrechnungsstatus** (vollständig, teilweise, offen, gesperrt)
- **Kunden- und interne Kommentare** zu jeder Buchung
- **Team- und Verantwortlichkeits-Informationen**

⚠️ **Wichtig:** Jede Zeile = eine einzelne Zeitbuchung. Zeitwerte sind in **Minuten** gespeichert (nicht Stunden!).

---

## 🔑 Identifikation & Basis-Informationen

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **prism_uid** | varchar | Eindeutige PRISM-ID für diese Zeitbuchung | `time_2025_123456` |
| **prism_access_group** | varchar | Zugriffsgruppe in PRISM | Firmen-/Organisations-ID |
| **prism_access_user** | varchar | Zugriffsberechtigter Benutzer | User-ID |
| **prism_source_system** | varchar | Quellsystem der Daten | `poool` |
| **prism_source_reference** | jsonb | JSON-Referenz zum Quellsystem mit allen IDs | `{"timetrack_time_id": "123", "person_id": "456", ...}` |

---

## 👤 Person/Mitarbeiter-Daten

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_person_poool_uid** | varchar | Eindeutige Poool-ID der Person/des Mitarbeiters | `person_123456` |
| **data_person_label** | varchar | Vollständiger Name der Person | `Max Mustermann`, `Anna Schmidt` |
| **data_person_firstname** | varchar | Vorname | `Max`, `Anna` |
| **data_person_lastname** | varchar | Nachname | `Mustermann`, `Schmidt` |
| **data_person_email** | varchar | E-Mail-Adresse | [`max.mustermann@firma.de`](mailto:max.mustermann@firma.de) |
| **data_person_team_label** | varchar | Team-Bezeichnung | `Development`, `Design`, `Consulting` |
| **data_person_team_token** | varchar | Team-Kürzel/Token | `DEV`, `DES`, `CON` |
| **data_person_planner_team** | varchar | Planer-Team | Team für Ressourcenplanung |

---

## 📅 Datums-Informationen

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_day_cw** | varchar | Kalenderwoche | `2025-W11`, `2024-W52` |
| **data_day_date** | date | Datum des Tages | `2025-03-15`, `2024-12-31` |
| **data_day_weekday** | varchar | Wochentagsname | `Montag`, `Dienstag`, `Mittwoch` |
| **data_day_weekday_index** | int4 | Wochentags-Index | `1` (Montag) bis `7` (Sonntag) |

💡 **Tipp:** Nutze `meta_dates` Tabelle für erweiterte Datums-Analysen!

---

## 🏢 Kunden-Daten

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_client_company_poool_uid** | varchar | Eindeutige Poool-ID des Kunden | `company_789012` |
| **data_client_number** | varchar | Kundennummer | `K-12345`, `10001` |
| **data_client_label** | varchar | Kundenname | `ACME GmbH`, `Musterfirma AG` |
| **data_client_label_legal** | varchar | Rechtlich korrekter Firmenname | `ACME Gesellschaft mit beschränkter Haftung` |
| **data_client_label_token** | varchar | Kunden-Kürzel | `ACME`, `MUSTER` |
| **data_client_zip** | varchar | Postleitzahl des Kunden | `10115`, `80331` |
| **data_client_country** | varchar | Land des Kunden | `DE`, `AT`, `CH` |

---

## 📊 Projekt-Daten

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_project_poool_uid** | varchar | Eindeutige Poool-ID des Projekts | `project_456789` |
| **data_project_number** | varchar | Projektnummer | `P-2025-001`, `PRJ-123` |
| **data_project_label** | varchar | Projektbezeichnung | `Website Relaunch`, `App Entwicklung` |
| **data_project_start_at** | date | Projekt-Startdatum | `2025-01-15` |
| **data_project_due_at** | date | Projekt-Deadline | `2025-12-31` |
| **data_project_is_internal** | bool | Internes Projekt-Flag | `true` = internes Projekt
`false` = Kundenprojekt |
| **data_project_type** | varchar | Projekttyp | `fixedPrice`, `timeAndMaterial`, `retainer` |
| **data_project_state** | varchar | Projekt-Status | `active`, `onHold`, `finished`, `archived` |
| **data_project_state_is_finished** | bool | Projekt abgeschlossen? | `true` / `false` |
| **data_project_state_is_archived** | bool | Projekt archiviert? | `true` / `false` |
| **data_project_team_label** | varchar | Projekt-Team | `Team Alpha`, `Development` |
| **data_project_team_token** | varchar | Projekt-Team Token | `ALPHA`, `DEV` |
| **data_project_person_responsible_label** | varchar | Name des Projektverantwortlichen | `Max Mustermann` |
| **data_project_person_responsible_email** | varchar | E-Mail des Projektverantwortlichen | [`max@firma.de`](mailto:max@firma.de) |
| **data_project_person_responsible_additional** | jsonb | Weitere Verantwortliche als JSON | JSON-Array mit Personen |

---

## 📋 Projektphasen-Daten

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_project_phase_poool_uid** | varchar | Eindeutige Poool-ID der Projektphase | `phase_234567` |
| **data_project_phase_label** | varchar | Phasenbezeichnung | `Konzeption`, `Design`, `Entwicklung` |
| **data_project_phase_state** | varchar | Phasen-Status | `active`, `finished`, `archived` |
| **data_project_phase_due_at** | date | Phasen-Deadline | `2025-06-30` |
| **data_project_phase_state_is_archived** | bool | Phase archiviert? | `true` / `false` |
| **data_project_phase_state_is_finished** | bool | Phase abgeschlossen? | `true` / `false` |

---

## 📦 Auftrags-Daten

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_order_number** | varchar | Auftragsnummer | `A-2025-001`, `ORD-456` |
| **data_order_label** | varchar | Auftragsbezeichnung | `Webdesign Paket`, `Entwicklung Phase 1` |
| **data_order_date** | date | Auftragsdatum | `2025-02-15` |
| **data_order_state** | varchar | Auftragsstatus | `open`, `inProgress`, `completed`, `cancelled` |
| **data_order_value_planned_internal** | float8 | Geplanter interner Wert | `15000.00` |
| **data_order_value_planned_external** | float8 | Geplanter externer Wert (Kunde) | `25000.00` |
| **data_invoices** | jsonb | Verknüpfte Rechnungen als JSON | JSON-Array mit Rechnungsinformationen |

---

## 🎫 Ticket-Daten

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_ticket_poool_uid** | varchar | Eindeutige Poool-ID des Tickets | `ticket_345678` |
| **data_ticket_number** | varchar | Ticketnummer | `T-2025-001`, `#123` |
| **data_ticket_label** | varchar | Ticket-Titel | `Bug Fix Header`, `Feature: Dark Mode` |
| **data_ticket_is_open** | bool | Ticket offen? | `true` / `false` |
| **data_ticket_state** | varchar | Ticket-Status | `open`, `inProgress`, `done`, `closed` |
| **data_ticket_value** | float8 | Ticket-Wert/Budget | `2400.00` (= 30h à 80€) |
| **data_ticket_job_poool_uid** | varchar | Eindeutige Poool-ID der Aufgabe im Ticket | `job_567890` |
| **data_ticket_job_label** | varchar | Aufgaben-Titel | `Backend API anpassen`, `Frontend testen` |
| **data_ticket_job_is_open** | bool | Aufgabe offen? | `true` / `false` |

---

## 🎯 Aktivitäts-Daten

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_activity_group_label** | varchar | Funktionsgruppe | `Entwicklung`, `Design`, `Beratung`, `Projektmanagement` |
| **data_activity_label** | varchar | Funktionstitel | `Programmierung`, `UI Design`, `Meeting` |
| **data_activity_cost_set_label** | varchar | Preisliste | `Standard Entwickler`, `Senior Designer` |
| **data_activity_cost_title_internal** | varchar | Funktionstitel intern | `Programmierung Backend` |
| **data_activity_cost_title_external** | varchar | Funktionstitel extern | `Entwicklung` |
| **data_activity_cost_description_external** | text | Funktions Beschreibung extern | `Entwicklung der Backend-API` |

---

## ⏱️ Zeiterfassungs-Daten

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_timetrack_date** | date | Datum der Zeitbuchung | `2025-03-15` |
| **data_timetrack_time** | int4 | Gebuchte Zeit in **Minuten** | `480` (= 8h)
`120` (= 2h)
`30` (= 0.5h) |
| **data_timetrack_time_is_overtime** | bool | Überstunden-Flag | `true` = Überstunde
`false` = Normalzeit |
| **data_timetrack_time_is_client_request** | bool | Auf Kundenwunsch? | `true` / `false` |
| **data_timetrack_comment_client** | text | Kommentar Kunden (extern sichtbar) | `Entwicklung der Nutzer-Login-Funktionalität` |
| **data_timetrack_comment_internal** | text | Kommentar intern (nur intern sichtbar) | `Komplizierter als erwartet, Legacy-Code` |

💡 **Wichtig:** Zeit ist in **Minuten** gespeichert! Für Stunden: `minuten / 60.0`

---

## 💰 Kosten-Daten

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_timetrack_cost_internal** | numeric | Interne Kosten (Selbstkosten) | `640.00` (= 8h à 80€) |
| **data_timetrack_cost_client** | numeric | Kundenkosten (Verkaufspreis) | `1200.00` (= 8h à 150€) |
| **data_timetrack_cost_ticket** | numeric | Ticketbasierte Kosten | `800.00` |

💡 **Info:**

- `cost_internal` = Was die Stunde intern kostet (Mitarbeiterkosten)
- `cost_client` = Was dem Kunden berechnet wird
- `cost_ticket` = Bei ticketbasierten Projekten

---

## ✅ Freigabe & Abrechnung

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_timetrack_clearable** | varchar | Abrechenbarkeit | `blocked` = Gesperrt
`direct` = Direkt abrechenbar
`order` = Über Auftrag abrechenbar
`ticketValue` = Über Ticket-Wert abrechenbar |
| **data_timetrack_cleared** | varchar | Abrechnungsstatus | `open` = Noch nicht abgerechnet
`partial` = Teilweise abgerechnet
`full` = Vollständig abgerechnet
`fullDraft` = Vollständig (Entwurf) |
| **data_timetrack_approval_state** | varchar | Freigabestatus | `open` = Wartet auf Freigabe
`approved` = Freigegeben
`rejected` = Abgelehnt |
| **data_timetrack_approval_datetime** | timestamp | Zeitstempel der Freigabe | `2025-03-16 14:30:00` |

⚠️ **Wichtig für Abrechnungen:**

- Nur Zeit mit `approval_state = 'approved'` sollte berechnet werden
- `cleared = 'open'` zeigt noch nicht abgerechnete Zeit
- `clearable = 'blocked'` kann nicht abgerechnet werden

---

## 🔒 Archivierung & Sperrung

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_timetrack_is_archived** | bool | Archiviert? | `true` = Archiviert
`false` = Aktiv |
| **data_timetrack_archived_by** | varchar | Archivierungsgrund | `manually` = Manuell archiviert
`project` = Durch Projekt archiviert
`projectPhase` = Durch Phase archiviert |
| **data_timetrack_day_locked** | bool | Tag gesperrt? | `true` = Gesperrt (keine Änderungen mehr)
`false` = Offen |

💡 **Info:** Gesperrte Tage können nicht mehr bearbeitet werden (z.B. nach Monatsabschluss)

---

## 📦 Dynamische Attribute

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_dynamic_attributes** | jsonb | Zusätzliche dynamische Felder als JSON | Kundenspezifische Zusatzfelder |

---

## 🏢 Meta-Informationen

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **meta_domain_label** | varchar | Domain-/Mandanten-Bezeichnung | `company-x`, `acme-corp` |
| **meta_instance_label** | varchar | Instanz-Bezeichnung | `production`, `staging` |
| **meta_instance_country** | varchar | Land der Instanz | `DE`, `AT`, `CH` |

---

## 🔑 System-Felder (Prism-Interna)

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **prism_created_at** | timestamp | Zeitstempel der Erstellung des Datensatzes | `2025-03-15 09:00:00` |
| **prism_updated_at** | timestamp | Zeitstempel der letzten Aktualisierung | `2025-03-15 14:30:00` |

---

## 💡 Häufige Verwendungen

### Projektzeit-Auswertung

```sql
SELECT 
    data_project_label,
    data_activity_label,
    COUNT(*) as anzahl_buchungen,
    ROUND(SUM(data_timetrack_time) / 60.0, 2) as stunden_gesamt,
    ROUND(SUM(data_timetrack_cost_internal), 2) as kosten_intern,
    ROUND(SUM(data_timetrack_cost_client), 2) as kosten_kunde
FROM timetrack_times
WHERE data_timetrack_date >= '2025-01-01'
  AND data_timetrack_date <= '2025-03-31'
GROUP BY data_project_label, data_activity_label
ORDER BY stunden_gesamt DESC;
```

### Mitarbeiter-Produktivität

```sql
SELECT 
    data_person_label,
    data_person_team_label,
    COUNT(DISTINCT data_timetrack_date) as arbeitstage,
    COUNT(*) as anzahl_buchungen,
    ROUND(SUM(data_timetrack_time) / 60.0, 2) as stunden_gesamt,
    ROUND(AVG(data_timetrack_time) / 60.0, 2) as durchschnitt_pro_buchung
FROM timetrack_times
WHERE data_timetrack_date >= CURRENT_DATE - INTERVAL '30 days'
  AND data_timetrack_time > 0
GROUP BY data_person_label, data_person_team_label
ORDER BY stunden_gesamt DESC;
```

### Überstunden-Analyse

```sql
SELECT 
    data_person_label,
    COUNT(*) FILTER (WHERE data_timetrack_time_is_overtime = true) as anzahl_ueberstunden,
    ROUND(SUM(data_timetrack_time) FILTER (WHERE data_timetrack_time_is_overtime = true) / 60.0, 2) as ueberstunden_gesamt,
    ROUND(SUM(data_timetrack_time) / 60.0, 2) as stunden_gesamt
FROM timetrack_times
WHERE data_timetrack_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY data_person_label
HAVING SUM(data_timetrack_time) FILTER (WHERE data_timetrack_time_is_overtime = true) > 0
ORDER BY ueberstunden_gesamt DESC;
```

### Freigabe-Status Übersicht

```sql
SELECT 
    data_timetrack_approval_state,
    data_timetrack_cleared,
    COUNT(*) as anzahl_buchungen,
    ROUND(SUM(data_timetrack_time) / 60.0, 2) as stunden,
    ROUND(SUM(data_timetrack_cost_client), 2) as potentieller_umsatz
FROM timetrack_times
WHERE data_timetrack_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY data_timetrack_approval_state, data_timetrack_cleared
ORDER BY stunden DESC;
```

### Noch nicht abgerechnete Zeit

```sql
SELECT 
    data_project_label,
    data_client_label,
    COUNT(*) as buchungen,
    ROUND(SUM(data_timetrack_time) / 60.0, 2) as offene_stunden,
    ROUND(SUM(data_timetrack_cost_client), 2) as offener_umsatz
FROM timetrack_times
WHERE data_timetrack_cleared = 'open'
  AND data_timetrack_approval_state = 'approved'
  AND data_timetrack_clearable != 'blocked'
GROUP BY data_project_label, data_client_label
ORDER BY offener_umsatz DESC;
```

### Ticket-Zeit-Tracking

```sql
SELECT 
    data_ticket_number,
    data_ticket_label,
    data_ticket_value,
    COUNT(DISTINCT data_person_poool_uid) as beteiligte_personen,
    ROUND(SUM(data_timetrack_time) / 60.0, 2) as stunden_gebucht,
    ROUND(SUM(data_timetrack_cost_client), 2) as kosten_gebucht,
    ROUND(data_ticket_value - SUM(data_timetrack_cost_client), 2) as restbudget
FROM timetrack_times
WHERE data_ticket_poool_uid IS NOT NULL
  AND data_timetrack_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY data_ticket_number, data_ticket_label, data_ticket_value
ORDER BY restbudget ASC;
```

---

## ⚠️ Wichtige Hinweise

### Zeiteinheiten beachten

- ❌ **Alle Zeitwerte sind in MINUTEN gespeichert**, nicht in Stunden!
- ✅ **Umrechnung in Stunden**: `data_timetrack_time / 60.0`
- ✅ **Umrechnung in Tage**: `data_timetrack_time / 480.0` (bei 8h Arbeitstag)

### Freigabe-Workflow

1. Zeit wird gebucht → `approval_state = 'open'`
2. Freigabe durch Projektleiter → `approval_state = 'approved'`
3. Abrechnung erstellt → `cleared = 'partial'` oder `'full'`

### Archivierung

- Archivierte Zeiten (`is_archived = true`) sind historisch
- Gesperrte Tage (`day_locked = true`) können nicht mehr geändert werden
- Typisch: Monatsabschluss sperrt alle Tage des Monats

### Abrechnungslogik

- `clearable = 'blocked'` → Kann nicht abgerechnet werden
- `clearable = 'direct'` → Direkt ohne Auftrag abrechenbar
- `clearable = 'order'` → Nur mit verknüpftem Auftrag abrechenbar
- `clearable = 'ticketValue'` → Über Ticket-Budget abrechenbar

---

## 🔗 Verwandte Tabellen

- **Contacts**: Mitarbeiter- und Kundendaten (verknüpft über `data_person_poool_uid`, `data_client_company_poool_uid`)
- **Projects**: Projektinformationen (verknüpft über `data_project_poool_uid`, `data_project_phase_poool_uid`)
- **Tickets**: Ticket-Details (verknüpft über `data_ticket_poool_uid`, `data_ticket_job_poool_uid`)
- **Timetrack Days**: Aggregierte Tagesansicht der Zeiterfassung
- **Timetrack Working Time Accounts**: Jahresübersicht Urlaub/Überstunden
- **Meta Dates**: Erweiterte Datums-Dimensionen für Analysen