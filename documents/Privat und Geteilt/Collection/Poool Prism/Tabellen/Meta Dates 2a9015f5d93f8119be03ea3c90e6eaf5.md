# Meta Dates

Article Description: Datumsdimensionstabelle mit Kalenderinformationen für zeitbasierte Analysen
Last Updated: 12. November 2025
Published: Yes
Suggested: No

---

## 📅 Datum-Basis

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **date** | date | Das Datum im Format YYYY-MM-DD (Primary Key) | `2025-03-15`, `2024-12-31` |

---

## 📆 Jahr & Monat

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **year** | int2 | Kalenderjahr (4-stellig) | `2025`, `2024` |
| **year_is_leap** | bool | Schaltjahr-Flag | `true` = Schaltjahr
`false` = kein Schaltjahr |
| **month** | int2 | Monat als Zahl | `1` (Januar) bis `12` (Dezember) |
| **month_name** | varchar | Monatsname (vollständig) | `Januar`, `Februar`, `März`, etc. |
| **month_days** | int2 | Anzahl der Tage im jeweiligen Monat | `28`, `29`, `30`, `31` |

---

## 🗓️ Tag & Wochentag

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **day** | int2 | Tag im Monat | `1` bis `31` |
| **day_name** | varchar | Wochentagsname (vollständig) | `Montag`, `Dienstag`, `Mittwoch`, etc. |
| **day_is_weekend** | bool | Wochenend-Flag | `true` = Samstag oder Sonntag
`false` = Montag bis Freitag |
| **day_of_week** | int2 | Wochentag (US-Standard) | `0` = Sonntag
`1` = Montag
`2` = Dienstag
...
`6` = Samstag |
| **day_of_week_iso** | int2 | Wochentag nach ISO 8601 (empfohlen für Europa!) | `1` = Montag
`2` = Dienstag
...
`7` = Sonntag |
| **day_of_year** | int2 | Tag im Jahr | `1` bis `366` (bei Schaltjahr) |

💡 **Tipp:** Für europäische Anwendungen immer `day_of_week_iso` verwenden statt `day_of_week`!

---

## 📊 ISO-Kalenderwoche

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **week_iso** | int2 | ISO-Kalenderwoche | `1` bis `53` |
| **week_iso_year** | int2 | ISO-Jahr der Kalenderwoche | `2025`, `2024`

⚠️ **Wichtig:** Kann vom `year` abweichen!
Beispiel: `2024-12-30` (Montag) liegt in ISO-Woche 1 von **2025** |

⚠️ **Wichtig:** ISO-Kalenderwochen können über Jahresgrenzen gehen. Immer `week_iso_year` statt `year` für Wochen-Gruppierungen verwenden!

**Beispiel:**

```sql
-- Korrekt:
GROUP BY week_iso_year, week_iso

-- Falsch:
GROUP BY year, week_iso
```

---

## 🎯 Quartal

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **quarter** | int2 | Quartal des Jahres | `1` = Q1 (Jan-März)
`2` = Q2 (Apr-Jun)
`3` = Q3 (Jul-Sep)
`4` = Q4 (Okt-Dez) |

---

## 🔑 System-Felder (Prism-Interna)

| Spalte | Typ | Beschreibung | Mögliche Werte / Beispiele |
| --- | --- | --- | --- |
| **prism_created_at** | timestamp | Zeitstempel der Erstellung des Datensatzes | `2025-01-01 00:00:00` |
| **prism_updated_at** | timestamp | Zeitstempel der letzten Aktualisierung | `2025-01-01 00:00:00` |