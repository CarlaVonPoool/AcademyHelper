# Timetrack Working Time Accounts

Article Description: Arbeitszeitkonten mit Urlaub, Überstunden und Gleitzeit pro Mitarbeiter:in und Jahr
Last Updated: 17. Dezember 2025
Published: Yes
Suggested: No

## 📊 Überblick

Die Tabelle `timetrack_working_time_accounts` enthält alle **Arbeitszeitkonten** der Mitarbeiter:innen pro Jahr. Jeder Datensatz repräsentiert das vollständige Arbeitszeitkonto einer Person für ein Kalenderjahr.

### Wichtige Konzepte

💡 **Ein Datensatz = Ein Jahr pro Person**: Pro Mitarbeiter:in und Jahr gibt es genau einen Datensatz. Die Tabelle wird automatisch aktualisiert basierend auf Zeiterfassungen und manuellen Modifikationen.

🔄 **Automatische Berechnung**: Die meisten Werte werden automatisch aus den täglichen Zeiterfassungen (`timetrack_days`) berechnet und aggregiert.

### Inhalt der Tabelle

- **Urlaubsverwaltung**: Initial-Anspruch, Übertrag, Verbrauch und Modifikationen
- **Überstunden-Tracking**: Aufbau, Übertrag und Abbau von Überstunden
- **Gleitzeit-Konten**: Aktuelle Stände und Flextime-Verwaltung
- **Freizeitausgleich**: Verbrauchte Ausgleichstage und -stunden
- **Sonderurlaub**: Spezielle Urlaubsformen

## 🎯 Analyse-Ideen

- Urlaubsplanung und -auswertung pro Team/Abteilung
- Überstunden-Monitoring und Abbau-Planung
- Gleitzeit-Entwicklung über Quartale
- Personalkosten-Planung basierend auf Arbeitszeit-Salden
- Team-Übersichten zu Resturlaub und Überstunden
- Compliance-Reports für Arbeitszeitgesetze

## 📋 Feldstruktur

### 1. System-Felder

Technische Felder für Datenherkunft und Zugriffskontrolle.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `prism_uid` | Prism UID | varchar | Eindeutige ID des Datensatzes in Prism |
| `prism_access_group` | Zugriffsgruppe | varchar | Zugriffsgruppe für Berechtigungen |
| `prism_access_user` | Zugriffsbenutzer | varchar | Benutzer-ID für Zugriffskontrolle |
| `prism_source_system` | Quellsystem | varchar | System aus dem die Daten stammen (z.B. "poool") |
| `prism_source_reference` | Quellreferenz | jsonb | JSON-Objekt mit Referenz-IDs zum Quellsystem |
| `prism_created_at` | Erstellt am | timestamp | Zeitpunkt der Erstellung in Prism |
| `prism_updated_at` | Aktualisiert am | timestamp | Zeitpunkt der letzten Aktualisierung |

### Unter-Felder von prism_source_reference:

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `prism_source_reference → entity_model` | Entitätstyp | text | Art des Datensatzes ("timetrackWorkingTimeAccount") |
| `prism_source_reference → timetrackWorkingTimeAccount_id` | Arbeitszeitkonto-ID | text | ID des Arbeitszeitkontos im Quellsystem |

### 2. Meta-Informationen

Informationen zur Instanz und Umgebung.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `meta_domain_label` | Domain | varchar | Domain-Bezeichnung der Instanz |
| `meta_instance_label` | Instanz | varchar | Name der Poool-Instanz |
| `meta_instance_country` | Instanz-Land | varchar | Ländercode der Instanz (z.B. "DE", "AT") |

### 3. Personendaten

Zuordnung zum Mitarbeiter:in.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_person_poool_uid` | Personen-UID | varchar | Eindeutige Poool-ID der Person |
| `data_person_label` | Personenname | varchar | Name der Person (Vor- und Nachname) |

### 4. Jahresinformationen

Basisdaten zum Arbeitsjahr.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_twta_year` | Jahr | int4 | 2024, 2025 | Kalenderjahr des Arbeitszeitkontos |
| `data_twta_year_days` | Tage im Jahr | int4 | 365 oder 366 | Anzahl der Tage im Jahr (Schaltjahr berücksichtigt) |
| `data_twta_timetrack_setup_days` | Zeiterfassungs-Setup-Tage | int4 | 0 bis 366 | Anzahl der Tage mit konfigurierter Zeiterfassung |

### 5. Urlaubsverwaltung - Tage

Urlaubsanspruch und -verbrauch in Tagen.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_twta_vacation_days_initial` | Urlaubsanspruch Initial | numeric | Vertraglich vereinbarter Jahresurlaub (in Tagen) |
| `data_twta_vacation_days_carry_over` | Urlaubsübertrag | numeric | Aus Vorjahr übertragene Urlaubstage |
| `data_twta_vacation_days_consumed` | Urlaubsverbrauch | numeric | Bereits genommene Urlaubstage |
| `data_twta_vacation_days_modification` | Urlaub Modifikation | numeric | Manuelle Korrekturen (+ Zuschlag / - Abzug) |
| `data_twta_vacation_days_total` | Resturlaub | numeric | Verbleibende Urlaubstage (Initial + Übertrag + Mod - Verbrauch) |

💡 **Berechnung**: `Resturlaub = Initial + Übertrag + Modifikation - Verbrauch`

### 6. Urlaubsverwaltung - Zeit (in Minuten)

Urlaubsverbrauch in Minuten (für Teilzeiturlaub).

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_twta_vacation_time_consumed` | Urlaubszeit verbraucht | int4 | Verbrauchte Urlaubszeit in Minuten |
| `data_twta_vacation_time_modification` | Urlaubszeit Modifikation | int4 | Manuelle Korrekturen in Minuten |
| `data_twta_vacation_time_total` | Urlaubszeit Gesamt | int4 | Verbleibende Urlaubszeit in Minuten |

⚠️ **Wichtig**: Zeitwerte sind in **Minuten** gespeichert! Umrechnung in Stunden: `minuten / 60.0`

### 7. Sonderurlaub

Spezielle Urlaubsformen (z.B. Bildungsurlaub, Hochzeit, Umzug).

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_twta_vacation_special_days_consumed` | Sonderurlaub Tage verbraucht | numeric | Verbrauchte Sonderurlaubstage |
| `data_twta_vacation_special_time_consumed` | Sonderurlaub Zeit verbraucht | int4 | Verbrauchte Sonderurlaubszeit in Minuten |

💡 **Hinweis**: Sonderurlaub wird nicht vom regulären Urlaubskonto abgezogen.

### 8. Überstunden

Überstunden-Aufbau und -Abbau.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_twta_overtime` | Überstunden aktuell | int4 | Im laufenden Jahr aufgebaute Überstunden (in Minuten) |
| `data_twta_overtime_carry_over` | Überstunden Übertrag | int4 | Aus Vorjahr übertragene Überstunden (in Minuten) |
| `data_twta_overtime_modification` | Überstunden Modifikation | int4 | Manuelle Korrekturen (in Minuten) |
| `data_twta_total_overtime` | Überstunden Gesamt | int4 | Gesamte Überstunden (Aktuell + Übertrag + Mod, in Minuten) |

💡 **Berechnung**: `Gesamt-Überstunden = Aktuell + Übertrag + Modifikation`

### 9. Freizeitausgleich (Comp Time)

Abbau von Überstunden durch Freizeitausgleich.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_twta_comp_time_days_consumed` | Freizeitausgleich Tage verbraucht | numeric | Abgebaute Überstunden als Tage |
| `data_twta_comp_time_time_consumed` | Freizeitausgleich Zeit verbraucht | int4 | Abgebaute Überstunden in Minuten |

### 10. Gleitzeit (Flextime)

Gleitzeit-Saldo für flexible Arbeitszeitmodelle.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_twta_flextime` | Gleitzeit aktuell | int4 | Im laufenden Jahr aufgebaute Gleitzeit (in Minuten) |
| `data_twta_flextime_carry_over` | Gleitzeit Übertrag | int4 | Aus Vorjahr übertragene Gleitzeit (in Minuten) |
| `data_twta_flextime_modification` | Gleitzeit Modifikation | int4 | Manuelle Korrekturen (in Minuten) |
| `data_twta_flextime_total` | Gleitzeit Gesamt | int4 | Gesamter Gleitzeit-Saldo (Aktuell + Übertrag + Mod, in Minuten) |
| `data_twta_flextime_nonstop` | Gleitzeit Nonstop | int4 | Nicht übertragbare Gleitzeit ("Use it or lose it", in Minuten) |

💡 **Unterschied Überstunden vs. Gleitzeit**:

- **Überstunden**: Fest aufgebaut, typischerweise auszahlbar oder langfristig übertragbar
- **Gleitzeit**: Flexibler Saldo, muss oft in bestimmtem Zeitrahmen ausgeglichen werden

### 11. Modifikationen

Änderungshistorie und automatische Anpassungen.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_twta_auto_modifications` | Automatische Modifikationen | jsonb | Automatisch durchgeführte Änderungen als JSON (z.B. Feiertags-Anpassungen) |
| `data_twta_modifications` | Manuelle Modifikationen | jsonb | Manuell durchgeführte Änderungen als JSON (mit Begründung) |

## 💡 SQL-Beispiele

### Resturlaub-Übersicht pro Person

```sql
SELECT 
    data_person_label,
    data_twta_year,
    data_twta_vacation_days_initial AS urlaubsanspruch,
    data_twta_vacation_days_carry_over AS uebertrag,
    data_twta_vacation_days_consumed AS genommen,
    data_twta_vacation_days_total AS resturlaub
FROM timetrack_working_time_accounts
WHERE data_twta_year = 2024
ORDER BY data_twta_vacation_days_total DESC;
```

### Überstunden-Monitoring mit Freizeitausgleich

```sql
SELECT 
    data_person_label,
    data_twta_year,
    ROUND(data_twta_total_overtime / 60.0, 2) AS ueberstunden_stunden,
    ROUND(data_twta_comp_time_time_consumed / 60.0, 2) AS abgebaut_stunden,
    ROUND((data_twta_total_overtime - data_twta_comp_time_time_consumed) / 60.0, 2) AS noch_abzubauen_stunden
FROM timetrack_working_time_accounts
WHERE data_twta_year = 2024
    AND data_twta_total_overtime > 0
ORDER BY ueberstunden_stunden DESC;
```

### Gleitzeit-Entwicklung über Jahre

```sql
SELECT 
    data_person_label,
    data_twta_year,
    ROUND(data_twta_flextime_carry_over / 60.0, 2) AS gleitzeit_start_stunden,
    ROUND(data_twta_flextime / 60.0, 2) AS gleitzeit_aufbau_stunden,
    ROUND(data_twta_flextime_total / 60.0, 2) AS gleitzeit_ende_stunden
FROM timetrack_working_time_accounts
WHERE data_person_poool_uid = 'person_xyz'
ORDER BY data_twta_year;
```

### Team-Übersicht Resturlaub und Überstunden

```sql
SELECT 
    data_person_label,
    data_twta_vacation_days_total AS resturlaub_tage,
    ROUND(data_twta_total_overtime / 60.0, 1) AS ueberstunden_h,
    ROUND(data_twta_flextime_total / 60.0, 1) AS gleitzeit_h
FROM timetrack_working_time_accounts
WHERE data_twta_year = 2024
ORDER BY data_person_label;
```

### Mitarbeiter mit kritischen Werten (Compliance Alert)

```sql
SELECT 
    data_person_label,
    data_twta_year,
    data_twta_vacation_days_total AS resturlaub,
    ROUND(data_twta_total_overtime / 60.0, 1) AS ueberstunden_h,
    CASE 
        WHEN data_twta_vacation_days_total > 15 THEN '⚠️ Viel Resturlaub'
        WHEN data_twta_total_overtime > 4800 THEN '⚠️ Hohe Überstunden (>80h)'
        ELSE '✓ OK'
    END AS status
FROM timetrack_working_time_accounts
WHERE data_twta_year = 2024
    AND (
        data_twta_vacation_days_total > 15
        OR data_twta_total_overtime > 4800
    )
ORDER BY data_twta_total_overtime DESC;
```

### Urlaubsplanung: Verfügbarkeit pro Quartal

```sql
SELECT 
    data_person_label,
    data_twta_vacation_days_total AS resturlaub_gesamt,
    ROUND(data_twta_vacation_days_total / 4.0, 1) AS empfohlen_pro_quartal
FROM timetrack_working_time_accounts
WHERE data_twta_year = 2024
    AND data_twta_vacation_days_total > 0
ORDER BY resturlaub_gesamt DESC;
```

## ⚠️ Wichtige Hinweise

### Zeiteinheiten beachten

- **Urlaubstage**: Als Dezimalzahl (z.B. 25.5 Tage)
- **Zeitwerte** (vacation_time, overtime, flextime, comp_time): In **Minuten**
- **Umrechnung in Stunden**: `minuten / 60.0`
- **Umrechnung in Tage**: `minuten / 480` (bei 8h Arbeitstag)

### Berechnung von Salden

- **Resturlaub** = Initial + Übertrag + Modifikation - Verbrauch
- **Gesamt-Überstunden** = Aktuell + Übertrag + Modifikation
- **Gleitzeit** = Aktuell + Übertrag + Modifikation

### Datenaktualisierung

- Werte werden **täglich** basierend auf `timetrack_days` automatisch aktualisiert
- Manuelle Modifikationen über die Felder `*_modification`
- **Ein Datensatz pro Person und Jahr**

### Unterschied Überstunden vs. Gleitzeit

- **Überstunden**: Langfristig aufgebaut, meist unbegrenzt übertragbar oder auszahlbar
- **Gleitzeit**: Kurzfristige Schwankungen, oft mit begrenztem Übertragszeitraum (z.B. max. 3 Monate)
- **Freizeitausgleich**: Abbau von Überstunden durch freie Tage

## 🔗 Verwandte Tabellen

- **Contacts**: Mitarbeiter:innen-Stammdaten (verknüpft über `data_person_poool_uid`)
- **Timetrack Days**: Tägliche Zeiterfassung, die zu den Jahres-Salden führt
- **Timetrack Times**: Einzelne Zeitbuchungen
- **Meta Dates**: Datumsinformationen für erweiterte Zeitauswertungen