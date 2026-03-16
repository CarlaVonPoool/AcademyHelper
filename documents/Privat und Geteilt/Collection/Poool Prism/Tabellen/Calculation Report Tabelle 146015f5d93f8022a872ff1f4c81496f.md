# Calculation Report Tabelle

Article Description: Daten zur Umsatz- und Kostenplanung
Last Updated: 15. Dezember 2025
Published: Yes
Suggested: No

## 📋 Überblick

Die Tabelle `calculation_report` bildet die Grundlage für die **Quartalsauswertung** und enthält Umsatz- und Kostenplanungsdaten aus verschiedenen Bereichen. Sie aggregiert:

- **Offene Auftragsplanungspositionen** (Orders)
- **Ausgangsrechnungen** (Invoices)
- **Lead-Planungen** mit Gewichtung
- **Wiederkehrende Ausgangsrechnungen** (zukünftige Umsätze)
- **Kosten** (Eingangsrechnungen)
- **Wiederkehrende Eingangsrechnungen**
- **Gehaltsdaten** (Personalkosten)

### Wichtige Konzepte

💡 **Tagesbezogene Planung**: Jede Zeile repräsentiert einen Eintrag für ein bestimmtes Datum (`data_day_date`). Wiederkehrende Einträge werden auf einzelne Tage aufgeteilt.

🔥 **Item Types**: Das Feld `data_item_type` unterscheidet:

- `invoice` = Ausgangsrechnung
- `invoice_recurring` = Wiederkehrende Ausgangsrechnung
- `lead` = Lead-Planung
- `order` = Auftragsplanung
- `salary` = Gehalt
- `salary_extra` = Sondergehaltskosten

📊 **Account Types**:

- `receivable` = Umsatz (Einnahmen)
- `payable` = Kosten (Ausgaben)

## 🎯 Analyse-Ideen

- Umsatzauswertung zu geplanten Projekten
- Umsatz- und Liquiditätsplanung
- Zielauswertungen (Zieldaten aus externen Planungsdaten)
- Gegenüberstellung von Umsatz und Aufwänden
- Quartalsauswertungen mit Forecasts
- Lead-Pipeline-Analysen mit Gewichtung
- Personalkosten-Planung

## 📊 Feldstruktur

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
| `prism_source_reference → bill_recurring_id` | Wiederkehrende Rechnung-ID | text | ID der wiederkehrenden Rechnung |
| `prism_source_reference → company_id` | Unternehmen-ID | text | ID des Unternehmens im Quellsystem |
| `prism_source_reference → invoice_id` | Rechnungs-ID | text | ID der Ausgangsrechnung |
| `prism_source_reference → lead_id` | Lead-ID | text | ID des Leads |
| `prism_source_reference → order_value_plan_id` | Auftragsplanung-ID | text | ID der Auftragsplanung |
| `prism_source_reference → project_id` | Projekt-ID | text | ID des Projekts |
| `prism_source_reference → project_phase_id` | Projektphasen-ID | text | ID der Projektphase |
| `prism_source_reference → project_sketch_id` | Projektskizze-ID | text | ID der Projektskizze |

### 2. Meta-Informationen

Informationen zur Instanz und Umgebung.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `meta_domain_label` | Domain | varchar | Domain-Bezeichnung der Instanz |
| `meta_instance_label` | Instanz | varchar | Name der Poool-Instanz |
| `meta_instance_country` | Instanz-Land | varchar | Ländercode der Instanz (z.B. "DE", "AT") |

### 3. Datums-Informationen

Zeitbezogene Daten für Analysen.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_day_date` | Datum | date | - | Datum des Eintrags (Fälligkeit, Rechnungsdatum, Planzeitpunkt) |
| `data_day_cw` | Kalenderwoche | varchar | "2024-W47" | Kalenderwoche des Datums |
| `data_day_weekday` | Wochentag | varchar | Monday, Tuesday, ... | Wochentag als Text (englisch) |
| `data_day_weekday_index` | Wochentag Index | int4 | 1-7 | Wochentag als Zahl (1 = Montag) |

💡 **Tipp**: Verwende `data_day_date` für Joins mit der `meta_dates` Tabelle für erweiterte Zeitauswertungen!

### 4. Kundendaten (Receivables)

Informationen zum Kunden bei Umsatzdaten.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_client_company_poool_uid` | Kunden-UID | varchar | Eindeutige Poool-ID des Kunden |
| `data_client_number` | Kundennummer | varchar | Kundennummer |
| `data_client_label` | Kundenname | varchar | Anzeigename des Kunden |
| `data_client_label_legal` | Offizieller Name | varchar | Offizieller/rechtlicher Name |
| `data_client_label_token` | Kunden-Kürzel | varchar | Kurzkürzel des Kunden |
| `data_client_zip` | PLZ | varchar | Postleitzahl des Kunden |
| `data_client_country` | Land | varchar | Ländercode (z.B. "DE") |

### 5. Lieferantendaten (Payables)

Informationen zum Lieferanten bei Kostendaten.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_supplier_company_poool_uid` | Lieferanten-UID | varchar | Eindeutige Poool-ID des Lieferanten |
| `data_supplier_number` | Lieferantennummer | varchar | Nummer des Lieferanten |
| `data_supplier_label` | Lieferantenname | varchar | Name des Lieferanten |
| `data_supplier_label_legal` | Offizieller Name | varchar | Offizieller/rechtlicher Name |
| `data_supplier_label_token` | Lieferanten-Kürzel | varchar | Kurzkürzel |
| `data_supplier_zip` | PLZ | varchar | Postleitzahl des Lieferanten |
| `data_supplier_country` | Land | varchar | Ländercode (z.B. "DE") |

### 6. Projektdaten

Informationen zum zugeordneten Projekt.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_poool_uid` | Projekt-UID | varchar | - | Eindeutige Poool-ID des Projekts |
| `data_project_number` | Projektnummer | varchar | - | Projektnummer |
| `data_project_label` | Projekttitel | varchar | - | Titel des Projekts |
| `data_project_start_at` | Projektstart | date | - | Startdatum des Projekts |
| `data_project_due_at` | Projektende | date | - | Enddatum/Fälligkeit |
| `data_project_is_internal` | Internes Projekt | bool | true/false | Ist es ein internes Projekt? |
| `data_project_type` | Projekttyp | varchar | - | Typ des Projekts |
| `data_project_state` | Projektstatus | varchar | - | Aktueller Status |
| `data_project_state_is_finished` | Status: Abgeschlossen | bool | true/false | Ist das Projekt abgeschlossen? |
| `data_project_state_is_archived` | Status: Archiviert | bool | true/false | Ist das Projekt archiviert? |
| `data_project_team_label` | Team | varchar | - | Name des Teams |
| `data_project_team_token` | Team-Kürzel | varchar | - | Kürzel des Teams |
| `data_project_person_responsible_label` | Projektverantwortlicher | varchar | - | Name des Verantwortlichen |
| `data_project_person_responsible_email` | Verantwortlicher E-Mail | varchar | - | E-Mail des Verantwortlichen |
| `data_project_person_responsible_additional` | Weitere Verantwortliche | jsonb | - | Zusätzliche Verantwortliche als JSON |

### 7. Projektphasen-Daten

Informationen zur Projektphase.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_phase_poool_uid` | Phasen-UID | varchar | - | Eindeutige Poool-ID der Phase |
| `data_project_phase_label` | Phasenname | varchar | - | Bezeichnung der Projektphase |
| `data_project_phase_state` | Phasenstatus | varchar | - | Status der Phase |
| `data_project_phase_due_at` | Phasen-Fälligkeit | date | - | Fälligkeitsdatum |
| `data_project_phase_state_is_archived` | Phasenstatus: Archiviert | bool | true/false | Ist die Phase archiviert? |
| `data_project_phase_state_is_finished` | Phasenstatus: Abgeschlossen | bool | true/false | Ist die Phase abgeschlossen? |

### 8. Referenz-IDs

Verknüpfungen zu anderen Datensätzen.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_lead_poool_uid` | Lead-UID | varchar | Eindeutige Poool-ID des Leads (bei Lead-Einträgen) |
| `data_accounts_payable_poool_uid` | Accounts Payable UID | varchar | Verknüpfung zu Eingangsrechnungsposition |
| `data_calculation_poool_uid` | Kalkulations-UID | varchar | Verknüpfung zu Kalkulation (bei Aufträgen/Rechnungen) |

### 9. Eintragsdaten (Item)

Kernfelder zur Charakterisierung des Eintrags.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_item_account_type` | Buchungstyp | varchar | `payable`, `receivable` | Art der Buchung:
• `payable` = Kosten (Ausgaben)
• `receivable` = Umsätze (Einnahmen) |
| `data_item_type` | Eintragstyp | varchar | `invoice`, `invoice_recurring`, `lead`, `order`, `salary`, `salary_extra` | Art des Eintrags:
• `invoice` = Ausgangsrechnung
• `invoice_recurring` = Wiederkehrende AR
• `lead` = Lead-Planung
• `order` = Auftragsplanung
• `salary` = Gehalt
• `salary_extra` = Sondergehalt |
| `data_item_payed_state` | Zahlungsstatus | varchar | `open`, `payed` | Status der Zahlung |
| `data_item_timetrack_clearable` | Zeiterfassung abrechenbar | varchar | `clearable`, `not_clearable` | Sind Zeiten abrechenbar? |
| `data_item_due_at` | Fälligkeit | date | - | Fälligkeitsdatum des Eintrags |
| `data_item_is_forecast` | Ist Forecast | bool | true/false | Liegt der Eintrag in der Zukunft? |
| `data_item_is_cleared` | Ist abgerechnet | bool | true/false | Wurde der Eintrag abgerechnet? (true = abgerechnet, false = nicht abgerechnet) |
| `data_item_value_is_weighted` | Ist gewichtet | bool | true/false | Ist der Wert gewichtet? (z.B. bei Leads) |
| `data_item_value_weighted_percentage` | Gewichtung % | float8 | 0-100 | Gewichtungsprozentsatz (z.B. 25% bei unsicheren Leads) |
| `data_item_value` | Wert | float8 | - | Betrag des Eintrags (Umsatz oder Kosten) |

💡 **Lead-Gewichtung**: Bei Leads wird der `data_item_value` mit `data_item_value_weighted_percentage` multipliziert, um realistische Planungswerte zu erhalten.

### 10. Dynamische Attribute

Erweiterbare Zusatzinformationen.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_dynamic_attributes` | Dynamische Attribute | jsonb | Zusätzliche benutzerdefinierte Attribute als JSON |

## 💡 SQL-Beispiele

### Umsatz- und Kostenplanung nach Monat

```sql
SELECT 
    DATE_TRUNC('month', data_day_date) AS monat,
    data_item_account_type,
    SUM(CASE 
        WHEN data_item_value_is_weighted = true 
        THEN data_item_value * (data_item_value_weighted_percentage / 100.0)
        ELSE data_item_value 
    END) AS summe
FROM calculation_report
GROUP BY 
    DATE_TRUNC('month', data_day_date),
    data_item_account_type
ORDER BY monat, data_item_account_type;
```

### Lead-Pipeline mit Gewichtung

```sql
SELECT 
    data_project_label,
    data_item_value AS lead_wert_brutto,
    data_item_value_weighted_percentage AS gewichtung,
    (data_item_value * data_item_value_weighted_percentage / 100.0) AS lead_wert_gewichtet,
    data_item_due_at AS voraussichtlicher_abschluss
FROM calculation_report
WHERE data_item_type = 'lead'
    AND data_item_value_is_weighted = true
ORDER BY lead_wert_gewichtet DESC;
```

### Quartalsauswertung (Forecast vs. Actual)

```sql
SELECT 
    EXTRACT(QUARTER FROM data_day_date) AS quartal,
    EXTRACT(YEAR FROM data_day_date) AS jahr,
    data_item_account_type,
    data_item_is_forecast,
    COUNT(*) AS anzahl_eintraege,
    SUM(data_item_value) AS summe
FROM calculation_report
WHERE EXTRACT(YEAR FROM data_day_date) = 2024
GROUP BY 
    EXTRACT(QUARTER FROM data_day_date),
    EXTRACT(YEAR FROM data_day_date),
    data_item_account_type,
    data_item_is_forecast
ORDER BY quartal, data_item_account_type, data_item_is_forecast;
```

### Projekt-Umsatzauswertung mit Item-Type Breakdown

```sql
SELECT 
    data_project_number,
    data_project_label,
    data_item_type,
    COUNT(*) AS anzahl,
    SUM(data_item_value) AS umsatz_geplant
FROM calculation_report
WHERE data_item_account_type = 'receivable'
    AND data_project_number IS NOT NULL
GROUP BY 
    data_project_number,
    data_project_label,
    data_item_type
ORDER BY data_project_number, umsatz_geplant DESC;
```

### Personalkosten nach Team

```sql
SELECT 
    data_project_team_label AS team,
    DATE_TRUNC('month', data_day_date) AS monat,
    COUNT(*) AS anzahl_eintraege,
    SUM(data_item_value) AS personalkosten
FROM calculation_report
WHERE data_item_type IN ('salary', 'salary_extra')
    AND data_project_team_label IS NOT NULL
GROUP BY 
    data_project_team_label,
    DATE_TRUNC('month', data_day_date)
ORDER BY monat, personalkosten DESC;
```

### Offene vs. Bezahlte Umsätze

```sql
SELECT 
    data_client_label,
    data_item_payed_state,
    COUNT(*) AS anzahl_rechnungen,
    SUM(data_item_value) AS summe
FROM calculation_report
WHERE data_item_account_type = 'receivable'
    AND data_item_type = 'invoice'
GROUP BY 
    data_client_label,
    data_item_payed_state
ORDER BY data_client_label, data_item_payed_state;
```