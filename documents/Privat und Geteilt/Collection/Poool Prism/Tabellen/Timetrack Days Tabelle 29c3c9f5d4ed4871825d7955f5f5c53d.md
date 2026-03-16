# Timetrack Days Tabelle

Article Description: Daten der Mitarbeiterzeiterfassung
Last Updated: 22. November 2024
Published: Yes
Suggested: No

## 🔑 Identifikation & Basis-Informationen

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **prism_uid** | varchar | Eindeutige PRISM-ID für diesen Tages-Datensatz | `timetrack_day_person123_20251112` |
| **prism_access_group** | varchar | Zugriffsgruppe in PRISM | Firmen-/Organisations-ID |
| **prism_access_user** | varchar | Zugriffsberechtigter Benutzer | User-ID |
| **prism_source_system** | varchar | Quellsystem der Daten | `poool` |
| **prism_source_reference** | jsonb | JSON-Referenz zum Quellsystem | `{"person_id": "123", "staff_id": "456"}` |

---

## 🏢 Meta-Informationen

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **meta_domain_label** | varchar | Domain-/Mandanten-Bezeichnung | `company-x`, `acme-corp` |
| **meta_instance_label** | varchar | Instanz-Bezeichnung | `production`, `staging` |
| **meta_instance_country** | varchar | Land der Instanz | `DE`, `AT`, `CH` |

---

## 👤 Person/Mitarbeiter:in

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_person_poool_uid** | varchar | Eindeutige Poool-ID der Person | `person_123456` |
| **data_person_label** | varchar | Vollständiger Name der Person | `Max Mustermann`, `Anna Schmidt` |
| **data_person_firstname** | varchar | Vorname | `Max`, `Anna` |
| **data_person_lastname** | varchar | Nachname | `Mustermann`, `Schmidt` |
| **data_person_email** | varchar | E-Mail-Adresse | [`max.mustermann@firma.de`](mailto:max.mustermann@firma.de) |
| **data_person_team_label** | varchar | Team-Name | `Entwicklung`, `Design` |
| **data_person_team_token** | varchar | Team-Kürzel | `dev`, `design` |
| **data_person_planner_team** | varchar | Planer-Team | Team-Bezeichnung für Ressourcenplanung |

---

## 📅 Datum & Tag-Informationen

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_day_date** | date | Das Datum des Tages | `2025-11-12`, `2025-03-15` |
| **data_day_cw** | varchar | Kalenderwoche | `2025-W46`, `2024-W52` |
| **data_day_weekday** | varchar | Wochentag (ausgeschrieben) | `Montag`, `Dienstag`, `Mittwoch` |
| **data_day_weekday_index** | int4 | Wochentag als Index | `1` = Montag
`2` = Dienstag
...
`7` = Sonntag |
| **data_day_is_holiday** | bool | Feiertags-Flag | `true` = Feiertag
`false` = regulärer Tag |
| **data_day_type_description** | varchar | Tag-Typ Beschreibung | `Arbeitstag`, `Feiertag`, `Urlaub`, `Wochenende` |
| **data_day_comment** | text | Kommentar zum Tag | Freitext-Kommentare, z.B. "Homeoffice", "Außendienst" |

💡 **Tipp:** Verwende `data_day_date` für Joins mit der `meta_dates` Tabelle für erweiterte Zeitauswertungen!

---

## 🎯 Arbeitszeit Soll (Target)

💡 **Hinweis:** Diese Felder enthalten die geplante/vertraglich vereinbarte Arbeitszeit

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_working_time_target_from** | int4 | Soll-Arbeitsbeginn in Minuten seit Mitternacht | `480` (= 08:00 Uhr)
`540` (= 09:00 Uhr) |
| **data_working_time_target_to** | int4 | Soll-Arbeitsende in Minuten seit Mitternacht | `1020` (= 17:00 Uhr)
`1080` (= 18:00 Uhr) |
| **data_working_time_target_break** | int4 | Soll-Pausenzeit in Minuten | `60` (= 1h)
`45` (= 45min) |
| **data_working_time_target_netto** | int4 | Soll-Nettoarbeitszeit in Minuten (ohne Pausen) | `480` (= 8h)
`420` (= 7h) |

💡 **Berechnung:** `target_netto = (target_to - target_from) - target_break`

---

## ⏱️ Arbeitszeit Ist (tatsächlich erfasst)

💡 **Hinweis:** Diese Felder enthalten die tatsächlich erfasste Arbeitszeit

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_working_time_from** | int4 | Ist-Arbeitsbeginn in Minuten seit Mitternacht | `495` (= 08:15 Uhr)
`0` (= nicht erfasst) |
| **data_working_time_to** | int4 | Ist-Arbeitsende in Minuten seit Mitternacht | `1035` (= 17:15 Uhr)
`0` (= nicht erfasst) |
| **data_working_time_break** | int4 | Ist-Pausenzeit in Minuten | `60` (= 1h)
`45` (= 45min) |
| **data_working_time_netto** | int4 | Ist-Nettoarbeitszeit in Minuten (ohne Pausen) | `480` (= 8h)
`540` (= 9h) |
| **data_working_time_unused** | int4 | Nicht genutzte Arbeitszeit in Minuten | Differenz zwischen Soll und gebuchter Zeit |

💡 **Berechnung:** `netto = (to - from) - break`

---

## ⚡ Gleitzeit

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_flextime_change** | int4 | Tägliche Gleitzeit-Änderung in Minuten | `60` (= +1h Guthaben)
`-30` (= -0.5h Minus)
`0` (= ausgeglichen) |

💡 **Berechnung:** `flextime_change = working_time_netto - working_time_target_netto`

**Beispiel:**

- Soll: 480 Min (8h)
- Ist: 540 Min (9h)
- Gleitzeit-Änderung: +60 Min (+1h)

---

## 📊 Zeittypen-Summen

💡 **Hinweis:** Diese Felder zeigen, wie die Arbeitszeit des Tages aufgeteilt wurde

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_timetrack_regular_sum** | int4 | Reguläre Projektzeit in Minuten | `480` (= 8h auf Projekten gebucht) |
| **data_timetrack_internal_sum** | int4 | Interne Zeit in Minuten (Meetings, Admin, etc.) | `120` (= 2h intern) |
| **data_timetrack_vacation_sum** | int4 | Urlaubszeit in Minuten | `480` (= 8h Urlaub = 1 Tag) |
| **data_timetrack_sick_sum** | int4 | Krankheitszeit in Minuten | `480` (= 8h krank = 1 Tag) |
| **data_timetrack_comp_time_sum** | int4 | Freizeitausgleich in Minuten | `480` (= 8h Überstundenabbau) |
| **data_timetrack_total_sum** | int4 | Gesamte gebuchte Zeit in Minuten | Summe aller Zeitbuchungen des Tages |

💡 **Idealerweise gilt:** `total_sum = regular_sum + internal_sum + vacation_sum + sick_sum + comp_time_sum`

---

## 📈 Arbeitszeitkonto-Modifikationen

💡 **Hinweis:** Diese Felder zeigen, wie dieser Tag die Arbeitszeitkonten beeinflusst

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_working_time_account_modification_flextime** | float8 | Änderung des Gleitzeit-Kontos in Stunden | `1.5` (= +1.5h)
`-0.5` (= -0.5h) |
| **data_working_time_account_modification_overtime** | float8 | Änderung des Überstunden-Kontos in Stunden | `2.0` (= +2h)
`-1.0` (= -1h Abbau) |
| **data_working_time_account_modification_vacation_days** | float8 | Änderung Urlaubstage-Konto | `-1.0` (= 1 Urlaubstag genommen)
`0.5` (= halber Tag) |
| **data_working_time_account_modification_vacation_time** | float8 | Änderung Urlaubszeit-Konto in Stunden | `-8.0` (= 8h Urlaub genommen) |
| **data_working_time_account_modification_details** | jsonb | Detaillierte Modifikations-Informationen (JSON) | Zusätzliche Details zu den Kontoänderungen |

💡 **Wichtig:** Diese Werte summieren sich über das Jahr zum Jahres-Arbeitszeitkonto (siehe `Timetrack Working Time Accounts`)

---

## 🔒 Zeiterfassungs-Kontrolle & Validierung

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **data_timetrack_locked** | bool | Tag gesperrt für Bearbeitung? | `true` = gesperrt
`false` = editierbar |
| **data_timetrack_location** | varchar | Arbeitsort | `office` = Büro
`homeoffice` = Homeoffice
`client` = Beim Kunden |
| **data_timetrack_is_invalid** | bool | Zeiterfassung ungültig/fehlerhaft? | `true` = ungültig
`false` = gültig |
| **data_timetrack_is_invalid_reason** | text | Grund für Ungültigkeit | `Keine Zeitbuchungen`, `Soll-Zeit nicht erreicht`, `Überschneidungen` |

⚠️ **Wichtig:**

- Gesperrte Tage können nicht mehr bearbeitet werden
- Ungültige Tage sollten korrigiert werden
- `is_invalid = true` weist auf Datenkonsistenz-Probleme hin

---

## 🔑 System-Felder (Prism-Interna)

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **prism_created_at** | timestamp | Zeitstempel der Erstellung | `2025-11-12 00:00:00` |
| **prism_updated_at** | timestamp | Zeitstempel der letzten Aktualisierung | `2025-11-12 18:30:00` |

---

## ⚠️ Wichtige Hinweise

### Zeiteinheiten beachten

- ⚠️ **Alle Zeitwerte sind in Minuten gespeichert** (nicht Stunden!)
- ⚠️ **Ausnahme:** `data_working_time_account_modification_*` Felder sind in **Stunden** (float8)
- ✅ **Umrechnung in Stunden:** `minuten / 60.0`
- ✅ **Umrechnung Minuten seit Mitternacht in Uhrzeit:** `08:00 Uhr = 480 Minuten`

### Soll vs. Ist verstehen

- **Target-Felder** (`*_target_*`) = Vertraglich vereinbarte Arbeitszeit
- **Ist-Felder** (`working_time_*` ohne target) = Tatsächlich erfasste Zeiten
- **Gleitzeit** entsteht aus der Differenz zwischen Soll und Ist

### Zeittypen-Summen

- `regular_sum` = Produktive Projektzeit
- `internal_sum` = Meetings, Admin, Schulungen
- `vacation_sum` = Urlaub
- `sick_sum` = Krankheit
- `comp_time_sum` = Überstundenabbau

### Datenkonsistenz

- Gesperrte Tage (`locked = true`) können nicht bearbeitet werden
- Ungültige Tage (`is_invalid = true`) weisen auf Probleme hin
- Filter in Reports: `WHERE data_timetrack_is_invalid = false`