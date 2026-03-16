# accounts_payable

Article Description: Daten zu Fremdkosten
Last Updated: 15. Dezember 2025
Published: Yes
Suggested: No

## 📋 Überblick

Die Tabelle `accounts_payable` enthält alle Eingangsrechnungspositionen mit ihrem vollständigen Kontext. Jede Zeile repräsentiert eine einzelne Position einer Eingangsrechnung mit allen zugehörigen Informationen zu Projekt, Auftrag, Lieferant, Budget und Abrechnung.

### Aufbau der Tabelle

Die Tabelle ist ähnlich aufgebaut wie der Export unter **Buchhaltung > Kostenmanagement**. Sie ermöglicht detaillierte Auswertungen über:

- **Fremdkosten**: Alle über Eingangsrechnungen erfassten Kosten
- **Projektzuordnung**: Welche Kosten wurden auf welche Projekte gebucht
- **Auftragsbezug**: Verknüpfung zu Bestellungen/Aufträgen
- **Budgetkontrolle**: Zuordnung zu Budgets und Kostenstellen
- **Abrechnungsstatus**: Clearable/Cleared Status für Weiterverrechnung

### Wichtige Konzepte

💡 **Eine Position = Eine Zeile**: Jede Eingangsrechnungsposition erscheint als separate Zeile. Rechnungsdaten (wie `data_invoice_netto`, `data_invoice_brutto`) wiederholen sich bei mehreren Positionen derselben Rechnung.

⚠️ **Beim Aggregieren beachten**:

- Rechnungssummen (`data_invoice_*`) nicht über Positionen summieren → GROUP BY oder MAX() verwenden
- Positionssummen (`data_invoice_position_*`) können summiert werden

## 🎯 Analyse-Ideen

- Zeitliche Verteilung der Buchungen innerhalb von Projekten
- Detaillierte Kostenentwicklung für Kunden und Projekte
- Kundendashboards zum Reporting von Kosten
- Kostenaufstellungen nach Unternehmensbereich
- Lieferantenauswertungen und Ausgabenanalyse
- Budget-Soll-Ist-Vergleiche
- Abrechnungscontrolling (Clearable vs. Cleared)

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
| `prism_source_reference → bill_incoming_id` | Eingangsrechnungs-ID | text | ID der Eingangsrechnung im Quellsystem |
| `prism_source_reference → bill_incoming_position_id` | Positions-ID | text | ID der Position im Quellsystem |
| `prism_source_reference → client_id` | Kunden-ID | text | ID des Kunden im Quellsystem |
| `prism_source_reference → company_id` | Unternehmen-ID | text | ID des Unternehmens im Quellsystem |
| `prism_source_reference → order_id` | Auftrags-ID | text | ID des zugeordneten Auftrags im Quellsystem |
| `prism_source_reference → project_id` | Projekt-ID | text | ID des Projekts im Quellsystem |
| `prism_source_reference → project_phase_id` | Projektphasen-ID | text | ID der Projektphase im Quellsystem |
| `prism_source_reference → project_state_id` | Projektstatus-ID | text | ID des Projektstatus im Quellsystem |

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
| `data_day_date` | Datum | date | - | Rechnungsdatum der Eingangsrechnung |
| `data_day_cw` | Kalenderwoche | varchar | "2024-W47" | Kalenderwoche des Rechnungsdatums |
| `data_day_weekday` | Wochentag | varchar | Monday, Tuesday, ... | Wochentag als Text (englisch) |
| `data_day_weekday_index` | Wochentag Index | int4 | 1-7 | Wochentag als Zahl (1 = Montag) |

💡 **Tipp**: Verwende `data_day_date` für Joins mit der `meta_dates` Tabelle für erweiterte Zeitauswertungen!

### 4. Kundendaten

Informationen zum Kunden, auf den die Kosten gebucht wurden.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_client_company_poool_uid` | Kunden-UID | varchar | Eindeutige Poool-ID des Kunden |
| `data_client_number` | Kundennummer | varchar | Kundennummer |
| `data_client_label` | Kundenname | varchar | Anzeigename des Kunden |
| `data_client_label_legal` | Offizieller Name | varchar | Offizieller/rechtlicher Name des Kunden |
| `data_client_label_token` | Kunden-Kürzel | varchar | Kurzkürzel des Kunden |
| `data_client_zip` | PLZ | varchar | Postleitzahl des Kunden |
| `data_client_country` | Land | varchar | Ländercode des Kunden (z.B. "DE") |

### 5. Projektdaten

Informationen zum Projekt, auf das die Kosten gebucht wurden.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_poool_uid` | Projekt-UID | varchar | - | Eindeutige Poool-ID des Projekts |
| `data_project_number` | Projektnummer | varchar | - | Projektnummer |
| `data_project_label` | Projekttitel | varchar | - | Titel des Projekts |
| `data_project_start_at` | Projektstart | date | - | Startdatum des Projekts |
| `data_project_due_at` | Projektende | date | - | Enddatum/Fälligkeit des Projekts |
| `data_project_is_internal` | Internes Projekt | bool | true/false | Ist es ein internes Projekt? |
| `data_project_type` | Projekttyp | varchar | - | Typ des Projekts |
| `data_project_state` | Projektstatus | varchar | - | Aktueller Status des Projekts |
| `data_project_state_is_finished` | Status: Abgeschlossen | bool | true/false | Ist das Projekt abgeschlossen? |
| `data_project_state_is_archived` | Status: Archiviert | bool | true/false | Ist das Projekt archiviert? |
| `data_project_team_label` | Team | varchar | - | Name des zuständigen Teams |
| `data_project_team_token` | Team-Kürzel | varchar | - | Kürzel des Teams |
| `data_project_person_responsible_label` | Projektverantwortlicher | varchar | - | Name des Projektverantwortlichen |
| `data_project_person_responsible_email` | Verantwortlicher E-Mail | varchar | - | E-Mail des Projektverantwortlichen |
| `data_project_person_responsible_additional` | Weitere Verantwortliche | jsonb | - | Zusätzliche Verantwortliche als JSON |

### 6. Projektphasen-Daten

Informationen zur Projektphase, falls zugeordnet.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_phase_poool_uid` | Phasen-UID | varchar | - | Eindeutige Poool-ID der Projektphase |
| `data_project_phase_label` | Phasenname | varchar | - | Bezeichnung der Projektphase |
| `data_project_phase_state` | Phasenstatus | varchar | - | Status der Phase |
| `data_project_phase_due_at` | Phasen-Fälligkeit | date | - | Fälligkeitsdatum der Phase |
| `data_project_phase_state_is_archived` | Phasenstatus: Archiviert | bool | true/false | Ist die Phase archiviert? |
| `data_project_phase_state_is_finished` | Phasenstatus: Abgeschlossen | bool | true/false | Ist die Phase abgeschlossen? |

### 7. Auftragsdaten

Informationen zum zugeordneten Auftrag (Bestellung).

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_order_number` | Auftragsnummer | varchar | - | Nummer des Auftrags |
| `data_order_label` | Auftragstitel | varchar | - | Titel/Bezeichnung des Auftrags |
| `data_order_date` | Auftragsdatum | date | - | Datum des Auftrags |
| `data_order_state` | Auftragsstatus | varchar | open, cleared, canceled | Status des Auftrags |
| `data_order_value_planned_internal` | Geplanter Wert Eigenleistung | float8 | - | Geplante Eigenleistungen im Auftrag |
| `data_order_value_planned_external` | Geplanter Wert Fremdleistung | float8 | - | Geplante Fremdleistungen im Auftrag |

### 8. Lieferantendaten

Informationen zum Lieferanten der Eingangsrechnung.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_supplier_company_poool_uid` | Lieferanten-UID | varchar | Eindeutige Poool-ID des Lieferanten |
| `data_supplier_number` | Lieferantennummer | varchar | Nummer des Lieferanten |
| `data_supplier_label` | Lieferantenname | text | Name des Lieferanten |
| `data_supplier_uid` | USt-ID | varchar | Umsatzsteuer-Identifikationsnummer |
| `data_supplier_zip` | PLZ | varchar | Postleitzahl des Lieferanten |
| `data_supplier_country` | Land | varchar | Ländercode des Lieferanten |

### 9. Eingangsrechnungsdaten

Kopfdaten der Eingangsrechnung (wiederholen sich bei mehreren Positionen).

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_accounts_payable_poool_uid` | Accounts Payable UID | varchar | - | Eindeutige Poool-ID des Eintrags |
| `data_invoice_is_draft` | Ist Entwurf | bool | true/false | Ist die Rechnung vorerfasst? |
| `data_invoice_is_expense` | Ist Spesenabrechnung | bool | true/false | Ist es eine Spesenabrechnung? |
| `data_invoice_number` | Rechnungsnummer | varchar | - | Rechnungsnummer des Lieferanten |
| `data_invoice_number_customer` | Kundennummer beim Lieferant | varchar | - | Eigene Kundennummer beim Lieferanten |
| `data_invoice_number_internal` | Interne Rechnungsnummer | varchar | - | Interne Nummer zur Verwaltung |
| `data_invoice_payment_reference` | Zahlungsreferenz | varchar | - | Referenz für die Zahlung |
| `data_invoice_title` | Rechnungstitel | varchar | - | Titel der Eingangsrechnung |
| `data_invoice_person_responsible_label` | Ersteller Name | varchar | - | Name des Erstellers |
| `data_invoice_person_responsible_email` | Ersteller E-Mail | varchar | - | E-Mail des Erstellers |
| `data_invoice_state` | Rechnungsstatus | varchar | open, payed, canceled | Status der Rechnung |
| `data_invoice_date` | Rechnungsdatum | date | - | Datum der Rechnung |
| `data_invoice_date_due` | Fälligkeitsdatum | date | - | Zahlungsfälligkeitsdatum |
| `data_invoice_netto` | Rechnungssumme Netto | float8 | - | Netto-Betrag der gesamten Rechnung |
| `data_invoice_brutto` | Rechnungssumme Brutto | float8 | - | Brutto-Betrag der gesamten Rechnung |
| `data_invoice_discount_date_due` | Skontofrist | date | - | Frist für Skontoabzug |
| `data_invoice_discount_percentage` | Skonto % | float8 | - | Skonto-Prozentsatz |
| `data_invoice_discount_sum` | Skonto Betrag | float8 | - | Absoluter Skonto-Betrag |
| `data_invoice_payed` | Bezahlt | float8 | - | Bereits bezahlter Gesamtbetrag |
| `data_invoice_payed_discount` | Bezahlt mit Skonto | float8 | - | Zahlungen innerhalb der Skontofrist |
| `data_invoice_payed_open` | Offener Betrag | float8 | - | Noch zu zahlender Restbetrag |
| `data_invoice_payed_balance` | Zahlungsdifferenz | float8 | - | Differenz (+ Überzahlung / - Unterzahlung) |
| `data_invoice_payed_state` | Zahlungsstatus | varchar | exact, over, under | Status der Zahlung |

⚠️ **Wichtig**: Diese Felder sind **pro Rechnung** identisch. Beim Aggregieren über Positionen nicht summieren, sondern `MAX()` oder `FIRST()` verwenden!

### 10. Positionsdaten

Daten der einzelnen Eingangsrechnungsposition.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_invoice_position_title` | Positionstitel | varchar | - | Bezeichnung der Position |
| `data_invoice_position_budget` | Budget | varchar | - | Zugeordnetes Budget |
| `data_invoice_position_cost_center` | Kostenstelle | varchar | - | Zugeordnete Kostenstelle |
| `data_invoice_position_apportion` | Aufteilung | varchar | - | Aufteilungsschlüssel |
| `data_invoice_position_reallocation_date` | Umlagedatum | date | - | Datum für Umlage/Reallokation |
| `data_invoice_position_netto` | Position Netto | float8 | - | Netto-Betrag der Position |
| `data_invoice_position_tax_percentage` | USt. % | float8 | - | Umsatzsteuer-Prozentsatz |
| `data_invoice_position_tax_sum` | USt. Betrag | float8 | - | Absoluter Umsatzsteuer-Betrag |
| `data_invoice_position_brutto` | Position Brutto | float8 | - | Brutto-Betrag der Position |
| `data_invoice_position_clearable` | Abrechenbar | bool | true/false | Kann diese Position weiterverrechnet werden? |
| `data_invoice_position_cleared` | Abgerechnet | bool | true/false | Wurde die Position bereits abgerechnet? |
| `data_invoice_position_cleared_drafted` | Abrechnung im Entwurf | bool | true/false | Existiert eine Entwurfsabrechnung? |
| `data_invoice_position_ordered` | Auftrag vorhanden | bool | true/false | Ist ein Auftrag zugeordnet? |
| `data_invoice_position_order_cleared` | Auftrag abgerechnet | bool | true/false | Wurde der zugeordnete Auftrag abgerechnet? |
| `data_invoice_position_order_cleared_partial` | Auftrag teilweise abgerechnet | bool | true/false | Wurde der Auftrag teilweise abgerechnet? |

💡 **Diese Felder können summiert werden** für Gesamtauswertungen über mehrere Positionen.

### 11. Dynamische Attribute

Erweiterbare Zusatzinformationen.

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `data_dynamic_attributes` | Dynamische Attribute | jsonb | Zusätzliche benutzerdefinierte Attribute als JSON |

## 💡 SQL-Beispiele

### Kostenauswertung pro Projekt

```sql
SELECT 
    data_project_number,
    data_project_label,
    COUNT(DISTINCT data_invoice_number) AS anzahl_rechnungen,
    COUNT(*) AS anzahl_positionen,
    SUM(data_invoice_position_netto) AS kosten_netto,
    SUM(data_invoice_position_brutto) AS kosten_brutto
FROM accounts_payable
WHERE data_project_number IS NOT NULL
GROUP BY data_project_number, data_project_label
ORDER BY kosten_brutto DESC;
```

### Lieferantenauswertung mit offenen Beträgen

```sql
SELECT 
    data_supplier_number,
    data_supplier_label,
    MAX(data_invoice_date) AS letzte_rechnung,
    COUNT(DISTINCT data_invoice_number) AS anzahl_rechnungen,
    SUM(data_invoice_position_netto) AS summe_netto,
    -- Offene Beträge: MAX() verwenden, da pro Rechnung identisch
    SUM(DISTINCT CASE 
        WHEN data_invoice_state = 'open' 
        THEN data_invoice_payed_open 
        ELSE 0 
    END) AS offene_betraege
FROM accounts_payable
GROUP BY data_supplier_number, data_supplier_label
ORDER BY summe_netto DESC;
```

### Budget-Auswertung

```sql
SELECT 
    data_invoice_position_budget AS budget,
    data_invoice_position_cost_center AS kostenstelle,
    COUNT(*) AS anzahl_positionen,
    SUM(data_invoice_position_netto) AS kosten_gesamt,
    SUM(CASE 
        WHEN data_invoice_position_cleared = true 
        THEN data_invoice_position_netto 
        ELSE 0 
    END) AS kosten_abgerechnet,
    SUM(CASE 
        WHEN data_invoice_position_cleared = false 
        THEN data_invoice_position_netto 
        ELSE 0 
    END) AS kosten_nicht_abgerechnet
FROM accounts_payable
WHERE data_invoice_position_budget IS NOT NULL
GROUP BY data_invoice_position_budget, data_invoice_position_cost_center
ORDER BY kosten_gesamt DESC;
```

### Zeitliche Kostenentwicklung pro Kunde

```sql
SELECT 
    data_client_number,
    data_client_label,
    DATE_TRUNC('month', data_day_date) AS monat,
    COUNT(*) AS positionen,
    SUM(data_invoice_position_netto) AS kosten_netto
FROM accounts_payable
WHERE data_day_date >= '2024-01-01'
GROUP BY 
    data_client_number, 
    data_client_label,
    DATE_TRUNC('month', data_day_date)
ORDER BY monat, kosten_netto DESC;
```

### Abrechnungscontrolling (Clearable vs. Cleared)

```sql
SELECT 
    data_project_number,
    data_project_label,
    COUNT(*) AS positionen_gesamt,
    SUM(CASE WHEN data_invoice_position_clearable = true THEN 1 ELSE 0 END) AS positionen_abrechenbar,
    SUM(CASE WHEN data_invoice_position_cleared = true THEN 1 ELSE 0 END) AS positionen_abgerechnet,
    SUM(data_invoice_position_netto) AS kosten_gesamt,
    SUM(CASE 
        WHEN data_invoice_position_clearable = true 
        AND data_invoice_position_cleared = false 
        THEN data_invoice_position_netto 
        ELSE 0 
    END) AS kosten_noch_abzurechnen
FROM accounts_payable
WHERE data_project_number IS NOT NULL
GROUP BY data_project_number, data_project_label
HAVING SUM(CASE 
    WHEN data_invoice_position_clearable = true 
    AND data_invoice_position_cleared = false 
    THEN data_invoice_position_netto 
    ELSE 0 
END) > 0
ORDER BY kosten_noch_abzurechnen DESC;
```