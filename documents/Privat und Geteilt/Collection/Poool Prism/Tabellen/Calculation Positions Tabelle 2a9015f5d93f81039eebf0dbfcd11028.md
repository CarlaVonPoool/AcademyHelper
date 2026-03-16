# Calculation Positions Tabelle

Article Description: Alle Kalkulationspositionen (Angebote, Aufträge, Rechnungen) mit Kunden-, Projekt- und Positionsdaten.
Last Updated: 12. November 2025
Published: Yes
Suggested: No

Die Calculation Positions Tabelle enthält **alle Kalkulationspositionen** aus Poool - von Angeboten über Aufträge bis zu Rechnungen.

## 🔑 System-Felder (Prism-Interna)

Diese Felder werden von Prism automatisch verwaltet und dienen der technischen Datenverwaltung.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `prism_uid` | Prism ID | Text | z.B. `abc123def456` | Eindeutige ID des Datensatzes in Prism |
| `prism_access_group` | Zugangsgruppe | Text | z.B. `company_abc` | Definiert, welche Unternehmensgruppe Zugriff auf diese Daten hat |
| `prism_access_user` | Zugriffsbenutzer | Text | z.B. `user_xyz` | Definiert, welcher Benutzer Zugriff auf diese Daten hat |
| `prism_source_system` | Quellsystem | Text | z.B. `poool` | Das Ursprungssystem, aus dem die Daten stammen |
| `prism_source_reference` | Quellreferenz (JSON) | JSON-Objekt | Enthält IDs aus Poool | Referenzen zu den Originaldaten im Quellsystem |
| `prism_created_at` | Erstellt am | Zeitstempel | z.B. `2024-11-12 10:30:00` | Zeitpunkt der ersten Synchronisation nach Prism |
| `prism_updated_at` | Aktualisiert am | Zeitstempel | z.B. `2024-11-12 14:15:00` | Zeitpunkt der letzten Aktualisierung in Prism |

### Quellreferenz-Details

Das Feld `prism_source_reference` enthält folgende Unter-Felder:

| Technischer Name | Anzeigename | Datentyp | Beschreibung |
| --- | --- | --- | --- |
| `prism_source_reference → entity_model` | Entitätstyp | Text | Art des Datensatzes in Poool (z.B. `calculation_position`) |
| `prism_source_reference → company_id` | Unternehmens-ID (Poool) | Text | ID des Unternehmens in Poool |
| `prism_source_reference → client_id` | Kunden-ID (Poool) | Text | ID des Kunden in Poool |
| `prism_source_reference → project_id` | Projekt-ID (Poool) | Text | ID des Projekts in Poool |
| `prism_source_reference → project_phase_id` | Projektphasen-ID (Poool) | Text | ID der Projektphase in Poool |
| `prism_source_reference → project_state_id` | Projektstatus-ID (Poool) | Text | ID des Projektstatus in Poool |
| `prism_source_reference → order_id` | Auftrags-ID (Poool) | Text | ID des Auftrags in Poool |
| `prism_source_reference → order_position_id` | Auftragspositions-ID (Poool) | Text | ID der Auftragsposition in Poool |
| `prism_source_reference → bill_id` | Rechnungs-ID (Poool) | Text | ID der Rechnung in Poool |
| `prism_source_reference → bill_position_id` | Rechnungspositions-ID (Poool) | Text | ID der Rechnungsposition in Poool |

---

## 📊 Meta-Informationen

Diese Felder enthalten Informationen über die Prism-Instanz.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `meta_domain_label` | Domain-Label | Text | z.B. `poool-software` | Kennzeichnung der Domain/Mandant in Prism |
| `meta_instance_label` | Instanz-Label | Text | z.B. `production` | Name der Poool-Instanz (z.B. Produktiv-, Test-System) |
| `meta_instance_country` | Instanz-Land | Text | z.B. `DE`, `AT`, `CH` | Länderkürzel der Instanz (relevant für Währung, Steuern, etc.) |

---

## 📅 Datums-Informationen

Diese Felder enthalten Datums- und Zeitinformationen zur Kalkulation.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_day_date` | Datum | Datum | z.B. `2024-11-12` | Kalkulationsdatum (Angebotsdatum, Auftragsdatum oder Rechnungsdatum) |
| `data_day_cw` | Kalenderwoche | Text | z.B. `2024-46` | Kalenderwoche des Kalkulationsdatums |
| `data_day_weekday` | Wochentag | Text | `Montag` bis `Sonntag` | Wochentag als Text |
| `data_day_weekday_index` | Wochentag (Index) | Zahl | `1` = Montag bis `7` = Sonntag | Wochentag als Zahl (ISO 8601 Standard) |

💡 **Hinweis:** Das `data_day_date` entspricht dem jeweiligen Kalkulationsdatum (`data_calculation_date`).

---

## 👤 Kundendaten

Diese Felder enthalten Informationen zum Kunden, für den die Kalkulation erstellt wurde.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_client_company_poool_uid` | Kunden-ID (Poool) | Text | z.B. `company_abc123` | Eindeutige Poool-ID des Kunden-Unternehmens |
| `data_client_number` | Kundennummer | Text | z.B. `K-12345` | Kundennummer des Unternehmens |
| `data_client_label` | Kundenname | Text | z.B. `Musterfirma GmbH` | Offizieller Firmenname des Kunden |
| `data_client_label_legal` | Kundenname (rechtlich) | Text | z.B. `Musterfirma Gesellschaft mit beschränkter Haftung` | Vollständiger rechtlicher Name |
| `data_client_label_token` | Kundenkürzel | Text | z.B. `MUST` | Kurzbezeichnung des Kunden |
| `data_client_zip` | Postleitzahl | Text | z.B. `80331` | PLZ des Kundensitzes |
| `data_client_country` | Land | Text | z.B. `DE`, `AT`, `CH` | Länderkürzel des Kundensitzes |

---

## 📁 Projektdaten

Diese Felder enthalten Informationen zum Projekt, dem die Kalkulation zugeordnet ist.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_poool_uid` | Projekt-ID (Poool) | Text | z.B. `project_xyz789` | Eindeutige Poool-ID des Projekts |
| `data_project_number` | Projektnummer | Text | z.B. `P-2024-001` | Projektnummer |
| `data_project_label` | Projekttitel | Text | z.B. `Website Relaunch 2024` | Titel des Projekts |
| `data_project_start_at` | Projektstartdatum | Datum | z.B. `2024-01-15` | Geplantes oder tatsächliches Startdatum |
| `data_project_due_at` | Projekt-Deadline | Datum | z.B. `2024-12-31` | Deadline/Frist des Projekts |
| `data_project_is_internal` | Ist internes Projekt | Ja/Nein | `true` / `false` | Kennzeichnet interne Projekte (keine Kundenabrechnung) |
| `data_project_type` | Projekttyp | Text | z.B. `Festpreis`, `T&M` | Art des Projekts |
| `data_project_state` | Projektstatus | Text | z.B. `In Bearbeitung`, `Abgeschlossen` | Aktueller Status des Projekts |
| `data_project_state_is_finished` | Status: Abgeschlossen | Ja/Nein | `true` / `false` | Kennzeichnet, ob Projekt abgeschlossen ist |
| `data_project_state_is_archived` | Status: Archiviert | Ja/Nein | `true` / `false` | Kennzeichnet archivierte Projekte |
| `data_project_team_label` | Projektteam | Text | z.B. `Team Digital` | Name des zuständigen Teams |
| `data_project_team_token` | Teamkürzel | Text | z.B. `DIG` | Kurzbezeichnung des Teams |
| `data_project_person_responsible_label` | Projektverantwortliche:r | Text | z.B. `Max Mustermann` | Name der projektverantwortlichen Person |
| `data_project_person_responsible_email` | E-Mail Projektverantwortliche:r | Text | z.B. [`max@firma.de`](mailto:max@firma.de) | E-Mail-Adresse der projektverantwortlichen Person |
| `data_project_person_responsible_additional` | Zusätzliche Projektverantwortliche (JSON) | JSON-Array | Array mit weiteren Personen | Liste weiterer projektverantwortlicher Personen |

---

## 🔵 Projektphasen-Daten

Diese Felder enthalten Informationen zur Projektphase, der die Kalkulation zugeordnet ist.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_phase_poool_uid` | Projektphasen-ID (Poool) | Text | z.B. `phase_abc456` | Eindeutige Poool-ID der Projektphase |
| `data_project_phase_label` | Phasentitel | Text | z.B. `Konzeptionsphase` | Titel der Projektphase |
| `data_project_phase_state` | Phasenstatus | Text | z.B. `In Bearbeitung`, `Abgeschlossen` | Aktueller Status der Phase |
| `data_project_phase_due_at` | Phasen-Deadline | Datum | z.B. `2024-06-30` | Deadline der Projektphase |
| `data_project_phase_state_is_finished` | Phasenstatus: Abgeschlossen | Ja/Nein | `true` / `false` | Kennzeichnet abgeschlossene Phasen |
| `data_project_phase_state_is_archived` | Phasenstatus: Archiviert | Ja/Nein | `true` / `false` | Kennzeichnet archivierte Phasen |

💡 **Hinweis:** Wenn keine Phase zugeordnet ist, sind diese Felder leer (`null`).

---

## 📋 Kalkulationsdaten

Diese Felder enthalten allgemeine Informationen zur gesamten Kalkulation (Angebot/Auftrag/Rechnung).

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_calculation_poool_uid` | Kalkulations-ID (Poool) | Text | z.B. `calc_xyz123` | Eindeutige Poool-ID der Kalkulation |
| `data_calculation_type` | Kalkulationstyp | Text | `offer`, `order`, `invoice` | Art der Kalkulation:
• `offer` = Angebot
• `order` = Auftrag
• `invoice` = Rechnung |
| `data_calculation_date` | Kalkulationsdatum | Datum | z.B. `2024-11-12` | Datum der Kalkulation (Angebots-/Auftrags-/Rechnungsdatum) |
| `data_calculation_creation_date` | Erstelldatum | Datum | z.B. `2024-11-10` | Datum, an dem die Kalkulation erstellt wurde |
| `data_calculation_performance_period_from` | Leistungszeitraum von | Datum | z.B. `2024-01-01` | Start des Leistungszeitraums |
| `data_calculation_performance_period_to` | Leistungszeitraum bis | Datum | z.B. `2024-12-31` | Ende des Leistungszeitraums |
| `data_calculation_number` | Kalkulationsnummer | Text | z.B. `A-2024-001`, `R-2024-042` | Fortlaufende Nummer (Angebots-/Auftrags-/Rechnungsnummer) |
| `data_calculation_title` | Kalkulationstitel | Text | z.B. `Website Relaunch - Phase 1` | Titel der Kalkulation |
| `data_calculation_state` | Kalkulationsstatus | Text | Je nach Typ unterschiedlich | Status der Kalkulation:
**Angebot:** `open`, `ordered`, `refused`
**Auftrag:** `open`, `cleared`, `canceled`
**Rechnung:** `draft`, `open`, `payed`, `canceled` |
| `data_calculation_reference_number` | Bestellnummer | Text | z.B. `BEST-2024-123` | Referenz-/Bestellnummer des Kunden |
| `data_calculation_activity_cost_set` | Preisliste | Text | z.B. `Standard 2024` | Name der verwendeten Preisliste/Aktivitätskostensätze |
| `data_calculation_person_responsible_label` | Kalkulationsverantwortliche:r | Text | z.B. `Maria Müller` | Name der für die Kalkulation verantwortlichen Person |
| `data_calculation_person_responsible_email` | E-Mail Kalkulationsverantwortliche:r | Text | z.B. [`maria@firma.de`](mailto:maria@firma.de) | E-Mail-Adresse der verantwortlichen Person |
| `data_calculation_tags` | Kalkulationstags (JSON) | JSON-Array | Array von Tag-Objekten | Schlagwörter zur Kategorisierung |
| `data_calculation_timetrack_clearable` | Zeiterfassung abrechenbar | Text | `clearable`, `not_clearable` | Gibt an, ob erfasste Zeiten abgerechnet werden können |

⚠️ **Wichtig:** Diese Felder sind pro Position identisch - sie beschreiben die gesamte Kalkulation. Beim Aggregieren **nicht summieren**, sondern gruppieren!

---

## 📦 Kalkulationsgruppen-Daten

Diese Felder enthalten Informationen zur Kalkulationsgruppe, der die Position zugeordnet ist.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_calculation_group_number` | Gruppennummer | Zahl | z.B. `1`, `2`, `3` | Fortlaufende Nummer der Gruppe innerhalb der Kalkulation |
| `data_calculation_group_type` | Gruppentyp | Text | `own_cost`, `external_cost` | Art der Leistung:
• `own_cost` = Eigenleistung
• `external_cost` = Fremdleistung |
| `data_calculation_group_optional` | Ist optional | Ja/Nein | `true` / `false` | Kennzeichnet optionale Gruppen (nicht im Hauptangebot enthalten) |
| `data_calculation_group_title` | Gruppentitel | Text | z.B. `Konzeption & Design` | Titel der Kalkulationsgruppe |
| `data_calculation_group_description` | Gruppenbeschreibung | Text | Freitext | Ausführliche Beschreibung der Gruppe |
| `data_calculation_group_account` | Buchhaltungskonto | Text | z.B. `8400` | Erlöskonto gemäß Steuersatz |
| `data_calculation_group_discount_percentage` | Gruppenrabatt (%) | Zahl | z.B. `10.5` | Rabatt in Prozent auf Gruppenebene |
| `data_calculation_group_global_overhead_percentage` | Globale Nebenkosten (%) | Zahl | z.B. `5.0` | Globaler Nebenkostenzuschlag in Prozent |
| `data_calculation_group_global_discount_percentage` | Globaler Rabatt (%) | Zahl | z.B. `3.0` | Globaler Rabatt in Prozent (auf alle Gruppen) |
| `data_calculation_group_total_sum` | Gruppensumme | Zahl | z.B. `12500.00` | Gesamtsumme der Gruppe (inkl. Rabatte/Zuschläge) |

⚠️ **Achtung:** Die `data_calculation_group_total_sum` ist **pro Position** hinterlegt! Beim Summieren entstehen falsche Werte. Gruppiere stattdessen nach `data_calculation_group_number` und nutze `MAX()` oder `FIRST()`.

---

## 📄 Positionsdaten

Diese Felder enthalten die eigentlichen Details zur einzelnen Kalkulationsposition.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_calculation_position_number` | Positionsnummer | Zahl | z.B. `1.1`, `1.2`, `2.1` | Fortlaufende Nummer der Position innerhalb der Gruppe |
| `data_calculation_position_activity_group_label` | Leistungsgruppe | Text | z.B. `Webentwicklung` | Funktions-/Leistungsgruppe aus der Preisliste |
| `data_calculation_position_activity_label` | Leistung | Text | z.B. `Frontend-Entwicklung` | Konkrete Funktion/Leistung aus der Preisliste |
| `data_calculation_position_material_label` | Material | Text | z.B. `Lizenz Adobe CC` | Materialbezeichnung aus der Materialliste |
| `data_calculation_position_quantity_unit_label` | Einheit | Text | z.B. `Stunden`, `Stück`, `Pauschal` | Mengeneinheit der Position |
| `data_calculation_position_quantity` | Anzahl/Menge | Zahl | z.B. `40.5` | Anzahl der Einheiten |
| `data_calculation_position_quantity_unit_multiplier` | Einheiten-Multiplikator | Text | z.B. `1`, `0.5` | Multiplikator für die Einheit (z.B. bei halben Tagen) |
| `data_calculation_position_title` | Positionstitel | Text | z.B. `Responsive Umsetzung` | Titel/Überschrift der Position |
| `data_calculation_position_description` | Positionsbeschreibung | Text | Freitext | Detaillierte Beschreibung der Position |
| `data_calculation_position_base_price_per_unit` | Einkaufspreis/Einheit | Zahl | z.B. `75.00` | Einkaufspreis (EK) pro Einheit (Basis für Kalkulation) |
| `data_calculation_position_sales_price_per_unit` | Verkaufspreis/Einheit | Zahl | z.B. `120.00` | Verkaufspreis (VK) pro Einheit (Kundenpreis) |
| `data_calculation_position_overhead_percentage` | Nebenkosten (%) | Zahl | z.B. `8.5` | Nebenkostenzuschlag in Prozent auf Positionsebene |
| `data_calculation_position_total_sum` | Positionssumme | Zahl | z.B. `4860.00` | Gesamtsumme der Position (Menge × VK × Zuschläge/Rabatte) |

💡 **Rechenbeispiel:**

```
Menge: 40 Stunden
Verkaufspreis/Einheit: 120 €
Nebenkosten: 8,5%
Gruppenrabatt: 10%

Positionssumme = 40 × 120 × (1 + 0,085) × (1 - 0,10) = 4.680 €
```

---

## 🔧 Dynamische Attribute

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_dynamic_attributes` | Dynamische Attribute (JSON) | JSON-Objekt | Individuelle Felder | Flexible, individuell konfigurierbare Felder pro Kunde (z.B. benutzerdefinierte Kategorien, Zusatzinformationen) |

💡 **Hinweis:** Dynamische Attribute sind kundenspezifische Zusatzfelder, die je nach Poool-Installation unterschiedlich sein können.

---

## 🎯 Typische Analysen mit dieser Tabelle

### Umsatzauswertungen

```sql
SELECT 
  data_client_label,
  data_calculation_type,
  SUM(data_calculation_position_total_sum) as umsatz
FROM calculation_positions
WHERE data_calculation_type = 'invoice'
  AND data_calculation_state = 'payed'
GROUP BY data_client_label, data_calculation_type
```

### Leistungsauswertung nach Aktivitäten

```sql
SELECT 
  data_calculation_position_activity_label,
  SUM(data_calculation_position_quantity) as menge,
  SUM(data_calculation_position_total_sum) as summe
FROM calculation_positions
WHERE data_calculation_type = 'invoice'
GROUP BY data_calculation_position_activity_label
```

### Projektübersicht (mit Gruppierung!)

```sql
SELECT 
  data_project_label,
  data_calculation_type,
  COUNT(DISTINCT data_calculation_poool_uid) as anzahl_kalkulationen,
  SUM(data_calculation_position_total_sum) as summe_positionen
FROM calculation_positions
GROUP BY data_project_label, data_calculation_type
```

⚠️ **Wichtig bei Gruppensummen:** Die `data_calculation_group_total_sum` **nicht summieren**, da sie bei jeder Position derselben Gruppe wiederholt wird! Verwende stattdessen:

```sql
SELECT 
  data_calculation_group_title,
  MAX(data_calculation_group_total_sum) as gruppensumme
FROM calculation_positions
GROUP BY 
  data_calculation_poool_uid,
  data_calculation_group_number,
  data_calculation_group_title
```