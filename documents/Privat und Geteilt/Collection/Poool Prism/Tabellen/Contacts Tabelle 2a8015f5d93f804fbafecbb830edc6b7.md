# Contacts Tabelle

Article Description: Alle Kontakte aus Poool: Unternehmen, Personen, Mitarbeitenden, Nutzern Enthält Stammdaten, Rechnungsinformationen, Teamstrukturen und Stundensätze.
Published: Yes
Suggested: No

# 🏢 Unternehmensdaten (Basis-Informationen)

Diese Felder enthalten die Stammdaten von Unternehmen, die in Poool unter **Kontakte > Unternehmen** erfasst werden.

## Kontakt-Typ

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_contact_type` | Kontakttyp | Text | `company`
`companySubsidiary`
`person` |  |

## **Firmen-Identifikation**

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_company_poool_uid` | Firmen-ID (Poool) | Text |  | Eindeutige Poool-ID des Unternehmens |
| `data_company_type` | Unternehmenstyp | Text | `company`
`person` | Unterscheidung zwischen:
• `company` = Juristische Person (GmbH, AG, etc.)
• `person` = Natürliche Person (Einzelunternehmer, Freiberufler) |
| `data_company_company_group` | Unternehmensgruppe | Text | Freitext | Zuordnung zu einer Unternehmensgruppe/Konzern |
| `data_company_uid` | Umsatzsteuer-ID | Text | z.B. `DE123456789` | USt-IdNr. des Unternehmens |

## **Firmennamen (verschiedene Varianten)**

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_company_name` | Firmenname (intern) | Text | Freitext | Interner Anzeigename in Poool |
| `data_company_name_normalized` | Firmenname (normalisiert) | Text | Freitext | Bereinigter Firmenname für Suchen |
| `data_company_name_legal` | Firmenname (rechtlich) | Text | Freitext | Offizieller Firmenname lt. Handelsregister |
| `data_company_name_token` | Firmenname (Token) | Text | Freitext | Suchbarer Token für Filterungen |

## Rechtsform & Registrierung

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_company_management` | Geschäftsführung | Text | Freitext | Namen der Geschäftsführer:innen |
| `data_company_jurisdiction` | Gerichtsstand | Text | Freitext | Zuständiges Gericht |
| `data_company_commercial_register` | Handelsregisternummer | Text | z.B. `HRB 12345` | Eintragung im Handelsregister |
| `data_company_data_privacy_number` | Datenschutznummer | Text | Freitext | Registrierungsnummer beim Datenschutzbeauftragten |

## **Personendaten (bei Einzelunternehmen)**

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_company_salutation` | Anrede | Text | `Herr`, `Frau`, `Divers` | Anrede des Einzelunternehmers/der Inhaberin |
| `data_company_title` | Titel | Text | `Dr.`, `Prof.`, etc. | Akademischer Titel |
| `data_company_firstname` | Vorname | Text | Freitext | Vorname bei Einzelunternehmen |
| `data_company_middlename` | Zweitname | Text | Freitext | Zweiter Vorname |
| `data_company_lastname` | Nachname | Text | Freitext | Nachname bei Einzelunternehmen |
| `data_company_birthday` | Geburtsdatum | Text | `YYYY-MM-DD` | Geburtsdatum (bei Einzelunternehmen) |

## **Organisation & Sonstiges**

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_company_position` | Position | Text | Freitext | Position/Rolle im Unternehmen |
| `data_company_function` | Funktion | Text | Freitext | Funktion/Aufgabenbereich |
| `data_company_department` | Abteilung | Text | Freitext | Abteilungszuordnung |
| `data_company_note` | Notiz | Langer Text | Freitext | Interne Notizen zum Unternehmen |
| `data_company_is_operator` | Ist Betreiber | Ja/Nein | `true` / `false` | Kennzeichnet ob das Unternehmen als Betreiber fungiert |
| `data_company_tags` | Tags (JSON) | JSON-Array | `["Tag1", "Tag2"]` | Flexible Kategorisierung durch Tags |

# 💼 Unternehmens-Rollen (Kunde/Lieferant)

Diese Felder kennzeichnen, ob ein Unternehmen als Kunde und/oder Lieferant geführt wird, inklusive Nummern und DATEV-Konten.

## Kunden-Daten

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_company_is_client` | Ist Kunde | Ja/Nein | `true` / `false` | Kennzeichnet, ob das Unternehmen ein Kunde ist |
| `data_company_client_number` | Kundennummer | Text | z.B. `K-10001` | Eindeutige Kundennummer in Poool |
| `data_company_client_accounting_account` | Debitorenkonto (DATEV) | Text | z.B. `10000` | DATEV-Kontennummer für Kunden (Debitoren) |
| `data_company_client_tenant_name` | Mandant (Kunde) | Text | z.B. `Mustermann GmbH` | Mandantenzuordnung für Kunden |
| `data_company_client_tenant_short_name` | Mandant Kurzname (Kunde) | Text | z.B. `MM` | Kurzbezeichnung des Mandanten |

## Lieferanten-Daten

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_company_is_supplier` | Ist Lieferant | Ja/Nein | `true` / `false` | Kennzeichnet, ob das Unternehmen ein Lieferant ist |
| `data_company_supplier_number` | Lieferantennummer | Text | z.B. `L-20001` | Eindeutige Lieferantennummer in Poool |
| `data_company_supplier_accounting_account` | Kreditorenkonto (DATEV) | Text | z.B. `70000` | DATEV-Kontennummer für Lieferanten (Kreditoren) |
| `data_company_supplier_tenant_name` | Mandant (Lieferant) | Text | z.B. `Mustermann GmbH` | Mandantenzuordnung für Lieferanten |
| `data_company_supplier_tenant_short_name` | Mandant Kurzname (Lieferant) | Text | z.B. `MM` | Kurzbezeichnung des Mandanten |

## Mandanten-Zuordnung

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_staff_tenant_name` | Mandant (Mitarbeiter) | Text | z.B. `Mustermann GmbH` | Mandantenzuordnung für Mitarbeiter:in |
| `data_staff_tenant_short_name` | Mandant Kurzname (Mitarbeiter) | Text | z.B. `MM` | Kurzbezeichnung des Mandanten |
| `data_staff_accounting_account` | Mitarbeiterkonto (DATEV) | Text | z.B. `50000` | DATEV-Kontennummer für Mitarbeiter-Abrechnungen |

# 🏢 Tochtergesellschaften/Niederlassungen

Diese Felder enthalten Informationen zu Tochtergesellschaften und Standorten eines Unternehmens.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_company_subsidiary_poool_uid` | Tochtergesellschaft-ID (Poool) | Text |  | Eindeutige Poool-ID der Tochtergesellschaft |
| `data_company_subsidiary_name` | Tochtergesellschaft Name | Text | Freitext | Anzeigename der Niederlassung/Tochter |
| `data_company_subsidiary_name_legal` | Tochtergesellschaft Name (rechtlich) | Text | Freitext | Offizieller rechtlicher Name |
| `data_company_subsidiary_name_token` | Tochtergesellschaft Token | Text | Freitext | Suchbarer Token für Filterungen |
| `data_company_subsidiary_name_location` | Standort | Text | z.B. `Berlin`, `München` | Geografischer Standort der Niederlassung |
| `data_company_subsidiary_uid` | Tochtergesellschaft UID | Text | z.B. `DE987654321` | USt-IdNr. der Tochtergesellschaft |
| `data_company_subsidiary_tags` | Tochtergesellschaft Tags (JSON) | JSON-Array | `["Standort", "Produktion"]` | Tags zur Kategorisierung der Niederlassung |

---

# 👤 Personendaten

Diese Felder enthalten Informationen zu Kontaktpersonen bei Kunden/Lieferanten sowie zu Mitarbeiter:innen.

## Personen-Identifikation

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_person_poool_uid` | Personen-ID (Poool) | Text | z.B. `pers_abc123` | Eindeutige Poool-ID der Person |
| `data_person_company_subsidiary_poool_uid` | Zugehörige Niederlassung-ID | Text | z.B. `sub_abc123` | Poool-ID der zugehörigen Niederlassung |
| `data_person_company_subsidiary_name` | Zugehörige Niederlassung | Text | Freitext | Name der Niederlassung, der die Person zugeordnet ist |
| `data_person_company_subsidiary_name_location` | Standort der Person | Text | z.B. `Berlin`, `München` | Geografischer Standort der Person |

## Namen & Anrede

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_person_full_name` | Vollständiger Name | Text | Freitext | Automatisch generierter vollständiger Name |
| `data_person_salutation` | Anrede | Text | `Herr`, `Frau`, `Divers` | Anrede der Person |
| `data_person_title` | Titel | Text | `Dr.`, `Prof.`, etc. | Akademischer Titel |
| `data_person_firstname` | Vorname | Text | Freitext | Vorname der Person |
| `data_person_middlename` | Zweitname | Text | Freitext | Zweiter Vorname |
| `data_person_lastname` | Nachname | Text | Freitext | Nachname der Person |
| `data_person_nickname` | Spitzname | Text | Freitext | Rufname oder Spitzname |
| `data_person_maiden_name` | Geburtsname | Text | Freitext | Geburtsname (falls abweichend) |

## Persönliche Daten

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_person_birthday` | Geburtsdatum | Text | `YYYY-MM-DD` | Geburtsdatum der Person |
| `data_person_gender` | Geschlecht | Text | `male`, `female`, `diverse` | Geschlecht der Person |

## Berufliche Zuordnung

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_person_position` | Position | Text | Freitext | Berufsbezeichnung/Position |
| `data_person_function` | Funktion | Text | Freitext | Funktion/Aufgabenbereich |
| `data_person_department` | Abteilung | Text | Freitext | Abteilungszuordnung |

## Sonstige Informationen

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_person_email_signature` | E-Mail-Signatur | Langer Text | HTML/Text | Hinterlegte E-Mail-Signatur der Person |
| `data_person_note` | Notiz | Langer Text | Freitext | Interne Notizen zur Person |
| `data_person_tags` | Tags (JSON) | JSON-Array | `["VIP", "Entscheider"]` | Flexible Kategorisierung durch Tags |
| `data_person_is_staff` | Ist Mitarbeiter | Ja/Nein | `true` / `false` | Kennzeichnet, ob die Person ein:e Mitarbeiter:in ist |

---

# 👥 Mitarbeiterdaten

Diese Felder enthalten spezifische Informationen zu Mitarbeiter:innen, die in Poool unter **Kontakte > Mitarbeiter** erfasst werden.

## Mitarbeiter-Identifikation

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_staff_poool_uid` | Mitarbeiter-ID (Poool) | Text | z.B. `staff_abc123` | Eindeutige Poool-ID des/der Mitarbeiter:in |
| `data_staff_name_token` | Namens-Token | Text | Freitext | Suchbarer Token für Filterungen |
| `data_staff_personio_id` | Personio-ID | Text | z.B. `12345` | ID aus dem HR-System Personio (falls verbunden) |

## Team-Zuordnung

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_staff_team` | Team (Hauptteam) | Text | z.B. `Design`, `Development` | Hauptteam des/der Mitarbeiter:in |
| `data_staff_team_token` | Team-Token | Text | Freitext | Suchbarer Token für Team-Filterungen |
| `data_staff_teams` | Teams (alle) (JSON) | JSON-Array | `["Design", "UX"]` | Alle Teams, denen der/die Mitarbeiter:in zugeordnet ist (Mehrfachzuordnung möglich) |

## Beschäftigungsverhältnis

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_staff_is_external` | Ist extern | Ja/Nein | `true` / `false` | Kennzeichnet externe Mitarbeiter:innen (Freelancer, Dienstleister) |
| `data_staff_start_at` | Startdatum | Datum | `YYYY-MM-DD` | Eintrittsdatum des/der Mitarbeiter:in |

# 💰 Kostenstrukturen (Stundensätze)

Diese Felder enthalten die aktuellen Stundensätze und Aktivitäts-Informationen der Mitarbeiter:innen.

## Aktuelle Aktivität

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_staff_activity_current_label` | Aktivitäts-Label | Text | z.B. `Senior Developer`, `Junior Designer` | Bezeichnung der aktuellen Aktivität/Rolle |
| `data_staff_activity_current_custom_label` | Aktivitäts-Label (individuell) | Text | Freitext | Individuelle Bezeichnung, falls vom Standard abweichend |
| `data_staff_activity_current_from` | Gültig ab | Datum | `YYYY-MM-DD` | Startdatum der aktuellen Aktivität/Stundensätze |
| `data_staff_activity_current_to` | Gültig bis | Datum | `YYYY-MM-DD` | Enddatum der aktuellen Aktivität (leer = unbefristet) |

## Stundensätze (aktuell)

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_staff_activity_current_internal_cost` | Interner Stundensatz | Zahl (Dezimal) | z.B. `65.00` | Interne Kostensatz pro Stunde (für Controlling) |
| `data_staff_activity_current_external_cost` | Externer Stundensatz | Zahl (Dezimal) | z.B. `85.00` | Externer Einkaufssatz (bei Freelancern/Dienstleistern) |
| `data_staff_activity_current_client_cost` | Kundenstundensatz | Zahl (Dezimal) | z.B. `120.00` | Verkaufssatz an Kunden (für Angebote/Rechnungen) |

<aside>
💡

**Hinweis:** Alle Stundensätze werden in Euro (€) angegeben.

</aside>

## Aktivitäts-Historie

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_staff_activity_history` | Aktivitäts-Historie (JSON) | JSON-Array | Historie aller Aktivitäten | Vollständige Historie aller vergangenen Aktivitäten und Stundensätze mit Zeiträumen |

---

# 👨‍💼 Benutzerverwaltung

Diese Felder enthalten Informationen zu User, die sich in Poool einloggen können.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_person_is_user` | Ist Benutzer | Ja/Nein | `true` / `false` | Kennzeichnet, ob die Person ein Poool-Benutzerkonto hat |
| `data_user_poool_uid` | Benutzer-ID (Poool) | Text | z.B. `user_abc123` | Eindeutige Poool-ID des Benutzerkontos |
| `data_user_email` | E-Mail-Adresse | Text | z.B. `max@firma.de` | Login-E-Mail-Adresse für Poool |
| `data_user_is_active` | Ist aktiv | Ja/Nein | `true` / `false` | Kennzeichnet, ob das Benutzerkonto aktiv ist (kann sich einloggen) |
| `data_user_is_admin` | Ist Administrator | Ja/Nein | `true` / `false` | Kennzeichnet, ob der/die Benutzer:in Administrator-Rechte hat |

---

## 🔑 System-Felder (Prism-Interna)

⚠️ **Hinweis:** Diese Felder werden von Poool automatisch befüllt und sind technische Datenbank-Informationen. **Für reguläre Auswertungen werden diese Felder nicht benötigt** – sie dienen ausschließlich der internen Datenverwaltung und Nachverfolgbarkeit.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `prism_uid` | Prism ID | Text | z.B. `abc123def456` | Eindeutige ID des Datensatzes in Prism |
| `prism_access_group` | Zugriffsgruppe | Text | Gruppenname | Zugriffsberechtigungen auf Gruppenebene |
| `prism_access_user` | Zugriffsbenutzer | Text | Benutzername | Zugriffsberechtigungen auf Benutzerebene |
| `prism_source_system` | Quellsystem | Text | `poool` | Das System, aus dem die Daten stammen |
| `prism_source_reference` | Quell-Referenz (JSON) | JSON | `{"company_id": "123", ...}` | Technische Referenzen zur Poool-Datenbank |
| `prism_source_reference → company_id` | Quell-ID: Firma | Text | z.B. `456` | Interne Poool-ID des Unternehmens |
| `prism_source_reference → entity_model` | Quell-Modell | Text | z.B. `Company`, `Person` | Typ des Datensatzes in Poool |
| `prism_source_reference → person_id` | Quell-ID: Person | Text | z.B. `789` | Interne Poool-ID der Person |
| `prism_created_at` | Erstellt am | Zeitstempel | `2025-11-10 14:30:00` | Zeitpunkt der Erstellung in Prism |
| `prism_updated_at` | Aktualisiert am | Zeitstempel | `2025-11-10 14:30:00` | Zeitpunkt der letzten Aktualisierung |
| `meta_domain_label` | Domain-Label | Text | z.B. Mandantenname | Übergeordnete Organisationseinheit |
| `meta_instance_label` | Instanz-Label | Text | z.B. Instanzname | Poool-Instanz (bei Multi-Tenant-Umgebungen) |
| `meta_instance_country` | Instanz-Land | Text | `DE`, `AT`, `CH` | Ländercode der Poool-Instanz |

# 🔧 Dynamische Attribute

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_dynamic_attributes` | Dynamische Attribute (JSON) | JSON-Objekt | Individuelle Felder | Flexible, individuell konfigurierbare Felder pro Kunde (z.B. benutzerdefinierte Kategorien, Zusatzinformationen) |

<aside>
💡

**Hinweis:** Dynamische Attribute sind kundenspezifische Zusatzfelder, die je nach Poool-Installation unterschiedlich sein können.

</aside>