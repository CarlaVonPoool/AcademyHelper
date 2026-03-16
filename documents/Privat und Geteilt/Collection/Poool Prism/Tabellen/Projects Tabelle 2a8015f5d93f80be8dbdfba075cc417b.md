# Projects Tabelle

Article Description: Alle Projektstammdaten und Projektphasen mit Status, Teams, Verantwortlichkeiten und Zeiterfassungs-Berechtigungen.
Last Updated: 11. November 2025
Published: Yes
Suggested: No

# 📊 Meta-Informationen

Diese Felder enthalten Informationen zur Poool-Umgebung und sind hauptsächlich für Multi-Tenant-Setups relevant.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `meta_domain_label` | Domain-Bezeichnung | Text | z.B. `firma.poool.cc` | Die Poool-Domain, auf der das Projekt angelegt wurde |
| `meta_instance_label` | Instanz-Bezeichnung | Text | z.B. `Production` | Die Poool-Instanz (z.B. Production, Staging) |

<aside>
💡

**Hinweis:** Diese Felder sind vor allem für Kunden mit mehreren Poool-Instanzen relevant (z.B. separate Test- und Produktiv-Umgebungen).

</aside>

---

# 🏢 Basis-Informationen

Diese Felder enthalten die grundlegenden Zuordnungen und die wichtige Unterscheidung zwischen Projekt und Projektphase.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_type` | Typ | Text | `project`<br>`projectPhase` | **Unterscheidung zwischen:**<br>• `project` = Hauptprojekt<br>• `projectPhase` = Projektphase |
| `data_project_poool_uid` | Projekt-ID (Poool) | Text | z.B. `proj_abc123` | Eindeutige Poool-ID des **Projekts** (nicht der Phase!) |
| `data_company_poool_uid` | Firmen-ID (Poool) | Text | z.B. `comp_abc123` | Verweis auf den Kunden in der Contacts-Tabelle |

<aside>
⚠️

**Wichtig:** Das Feld `data_type` ist entscheidend für alle Auswertungen!

- Für reine Projektlisten: `WHERE data_type = 'project'`
- Für Phasen: `WHERE data_type = 'projectPhase'`
- **Kleinschreibung beachten!** ❌ `'Project'` ist falsch, ✅ `'project'` ist richtig
</aside>

---

# 📋 Projekt-Stammdaten

Diese Felder enthalten die Kern-Informationen zu jedem Projekt.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_title` | Projekttitel | Text | z.B. `Website Relaunch 2024` | Der Titel/Name des Projekts |
| `data_project_number` | Projektnummer | Text | z.B. `P-2024-0123` | Die eindeutige Projektnummer (kann kundenspezifisch formatiert sein) |
| `data_project_reference_number` | Referenznummer | Text | z.B. `KD-2024-XYZ` | Externe Referenznummer (z.B. Kundennummer beim Kunden) |
| `data_project_description` | Projektbeschreibung | Text | Freitext | Detaillierte Beschreibung des Projekts |
| `data_project_comment` | Kommentar | Text | Freitext | Zusätzliche Kommentare und Notizen zum Projekt |
| `data_project_type_label` | Projekttyp | Text | z.B. `Beratung`, `Entwicklung` | Kategorisierung des Projekttyps (kundenspezifisch konfigurierbar) |
| `data_project_priority` | Priorität | Text | z.B. `Hoch`, `Mittel`, `Niedrig` | Prioritätsstufe des Projekts |
| `data_project_is_internal` | Ist internes Projekt | Ja/Nein | `true` / `false` | Kennzeichnet, ob es ein internes Projekt ist (keine Kundenabrechnung) |
| `data_project_is_main_project` | Ist Hauptprojekt | Ja/Nein | `true` / `false` | Kennzeichnet, ob es ein Hauptprojekt ist (für Projekt-Hierarchien) |

<aside>
💡

**Hinweis:** Die Felder `data_project_type_label` und `data_project_reference_number` enthalten kundenspezifisch konfigurierte Werte und können je nach Poool-Installation unterschiedlich aussehen.

</aside>

---

# ⚙️ Projekt-Status & Workflow

Diese Felder steuern den Status und Workflow-Zustand des Projekts.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_state_label` | Status-Bezeichnung | Text | z.B. `In Bearbeitung`, `Abgeschlossen`, `Angebot` | Die Bezeichnung des aktuellen Projekt-Status |
| `data_project_state_description` | Status-Beschreibung | Text | Freitext | Detaillierte Beschreibung des Status |
| `data_project_state_level` | Status-Level | Text | z.B. `active`, `finished`, `archived` | Technisches Status-Level zur Kategorisierung |
| `data_project_state_is_finished` | Ist abgeschlossen | Ja/Nein | `true` / `false` | Kennzeichnet, ob das Projekt abgeschlossen ist |
| `data_project_state_is_archived` | Ist archiviert | Ja/Nein | `true` / `false` | Kennzeichnet, ob das Projekt archiviert ist |
| `data_project_state_is_project_lock` | Projektsperre aktiv | Ja/Nein | `true` / `false` | Kennzeichnet, ob das Projekt gesperrt ist (keine Bearbeitung möglich) |
| `data_project_state_timetrack_blocked` | Zeiterfassung gesperrt (Status) | Ja/Nein | `true` / `false` | Kennzeichnet, ob die Zeiterfassung durch den Status gesperrt ist |

<aside>
💡

**Wichtig für Auswertungen:**

- Aktive Projekte: `WHERE data_project_state_is_finished = false AND data_project_state_is_archived = false`
- Abgeschlossene Projekte: `WHERE data_project_state_is_finished = true`
- Archivierte Projekte: `WHERE data_project_state_is_archived = true`
</aside>

<aside>
💡

**Hinweis:** Die Felder `data_project_state_label` enthalten kundenspezifisch konfigurierte Werte und können je nach Poool-Installation unterschiedlich aussehen.

</aside>

---

# 👥 Projekt-Team & Verantwortlichkeiten

Diese Felder enthalten Informationen zu Team-Zuordnungen und verantwortlichen Personen.

### Team-Zuordnung

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_team_label` | Team-Bezeichnung | Text | z.B. `Development Team`, `Marketing` | Die Bezeichnung des zugeordneten Teams |
| `data_project_team_token` | Team-Token | Text | z.B. `team_abc123` | Technischer Identifier des Teams |

---

# Projektverantwortung

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| data_project_person_responsible.label | Projektverantwortlicher | Text | z.B. `Max Mustermann` | Name der hauptverantwortlichen Person |
| `data_project_person_responsible → person_poool_uid` | Verantwortlicher ID (Poool) | Text | z.B. `person_abc123` | Verweis auf die Person in der Contacts-Tabelle |
| `data_project_person_responsible` | Projektverantwortlicher (JSON) | JSON-Objekt | `{"label": "...", "person_poool_uid": "..."}` | Vollständiges Objekt der hauptverantwortlichen Person |
| `data_project_person_responsible_additionals` | Zusätzliche Verantwortliche (JSON) | JSON-Array | Array von Personen-Objekten | Liste weiterer verantwortlicher Personen |
| `data_project_staffs` | Projekt-Mitarbeiter (JSON) | JSON-Array | Array von Mitarbeiter-Objekten | Liste aller dem Projekt zugeordneten Mitarbeiter:innen |

<aside>
💡

**Hinweis:** Die JSON-Felder enthalten strukturierte Daten. Für Power BI/Excel empfiehlt sich die Nutzung der aufgelösten Felder (`→ label` und `→ person_poool_uid`).

</aside>

---

# ⏱️ Zeiterfassung & Berechtigungen

Diese Felder steuern die Zeiterfassungs-Berechtigungen auf Projektebene.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_timetrack_blocked` | Zeiterfassung gesperrt (Projekt) | Ja/Nein | `true` / `false` | Kennzeichnet, ob die Zeiterfassung auf das Projekt generell gesperrt ist |
| `data_project_timetrack_clearable` | Zeiterfassung abrechenbar | Text | z.B. `clearable`, `not_clearable` | Gibt an, ob erfasste Zeiten abgerechnet werden können |
| `data_project_timetrack_permission` | Zeiterfassungs-Berechtigung | Text | z.B. `all`, `team`, `assigned` | Steuert, wer auf das Projekt Zeit erfassen darf:<br>• `all` = Alle Mitarbeiter:innen<br>• `team` = Nur Team-Mitglieder<br>• `assigned` = Nur zugewiesene Personen |

<aside>
⚠️

**Wichtig:** Es gibt **zwei** Zeiterfassungs-Sperren:

1. **Status-basiert:** `data_project_state_timetrack_blocked` (durch Projekt-Status, Einstellungen Zeiterfassung sperren)
2. **Projekt-basiert:** `data_project_timetrack_blocked` (Einstellungen Status am Projekt)

Beide Felder müssen `false` sein, damit Zeiterfassung möglich ist!

</aside>

---

# 📅 Termine & Deadlines

Diese Felder enthalten alle zeitlichen Informationen zum Projekt.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_start_at` | Projektstartdatum | Datum | z.B. `2024-01-15` | Das geplante oder tatsächliche Startdatum des Projekts |
| `data_project_deadline_at` | Projekt-Deadline | Datum | z.B. `2024-12-31` | Die Deadline/Frist für das Projekt |

<aside>
⚠️

**Hinweis:** Projektphasen haben eigene Deadlines im Feld `data_project_phase_deadline_at` (siehe Projektphasen-Daten weiter unten).

</aside>

---

# 🔗 JSON-Datenfelder & Zusatzinformationen

Diese Felder enthalten komplexe, strukturierte Daten im JSON-Format. Sie sind primär für erweiterte Auswertungen relevant.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_links` | Projekt-Links (JSON) | JSON-Array | Array von Link-Objekten | Externe Links zum Projekt (z.B. zu Dokumenten, Websites, Tools) |
| `data_project_tags` | Projekt-Tags (JSON) | JSON-Array | Array von Tag-Objekten | Schlagwörter und Kategorisierungen für das Projekt |
| `data_project_activity_costs` | Aktivitätskosten (JSON) | JSON-Array | Array von Kostensatz-Objekten | Hinterlegte Aktivitätskostensätze für das Projekt |
| `data_project_activity_cost_sets` | Aktivitätskostensätze (JSON) | JSON-Array | Array von Kostensatz-Sets | Sets von Aktivitätskostensätzen |
| `data_project_external_costs` | Externe Kosten (JSON) | JSON-Array | Array von Kosten-Objekten | Hinterlegte externe Kosten und Fremdleistungen |
| `data_dynamic_attributes` | Dynamische Attribute (JSON) | JSON-Objekt | Individuelle Felder | Flexible, individuell konfigurierbare Felder pro Kunde (z.B. benutzerdefinierte Kategorien, Zusatzinformationen) |

<aside>
⚠️

**Hinweis für Power BI/Excel-Nutzer:innen:**

- JSON-Felder können mit Power Query aufgelöst werden
- Für einfache Auswertungen sind die Standard-Textfelder meist ausreichend
- Dynamische Attribute sind kundenspezifisch und können je nach Poool-Installation unterschiedlich sein
</aside>

---

# 🔵 Projektphasen-Daten

Diese Felder sind **nur für Projektphasen** relevant (wenn `data_type = 'projectPhase'`). Bei Hauptprojekten sind diese Felder leer/null.

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `data_project_phase_poool_uid` | Projektphasen-ID (Poool) | Text | z.B. `phase_abc123` | Eindeutige Poool-ID der Projektphase |
| `data_project_phase_title` | Phasentitel | Text | z.B. `Konzeption`, `Umsetzung` | Der Titel/Name der Projektphase |
| `data_project_phase_state_label` | Phasen-Status | Text | z.B. `In Bearbeitung`, `Abgeschlossen` | Die Bezeichnung des aktuellen Phasen-Status |
| `data_project_phase_deadline_at` | Phasen-Deadline | Datum | z.B. `2024-06-30` | Die Deadline/Frist für die Projektphase |
| `data_project_phase_timetrack_blocked` | Zeiterfassung gesperrt (Phase) | Ja/Nein | `true` / `false` | Kennzeichnet, ob die Zeiterfassung auf die Phase gesperrt ist |
| `data_project_phase_timetrack_clearable` | Zeiterfassung abrechenbar (Phase) | Text | z.B. `clearable`, `not_clearable` | Gibt an, ob erfasste Zeiten auf die Phase abgerechnet werden können |
| `data_project_phase_tags` | Phasen-Tags (JSON) | JSON-Array | Array von Tag-Objekten | Schlagwörter und Kategorisierungen für die Projektphase |

<aside>
⚠️

**Wichtig für Auswertungen:**

- **Projektphasen filtern:** `WHERE data_type = 'projectPhase'`
- **Verknüpfung Projekt ↔ Phase:** Beide haben dasselbe `data_project_poool_uid`
- **Phasen eines bestimmten Projekts:**
</aside>

---

# 🔑 System-Felder (Prism-Interna)

<aside>
⚠️

**Hinweis:** Diese Felder werden von Poool automatisch befüllt und sind technische Datenbank-Informationen. **Für reguläre Auswertungen werden diese Felder nicht benötigt** – sie dienen ausschließlich der internen Datenverwaltung und Nachverfolgbarkeit.

</aside>

| Technischer Name | Anzeigename | Datentyp | Mögliche Werte | Beschreibung |
| --- | --- | --- | --- | --- |
| `prism_uid` | Prism ID | Text | z.B. `abc123def456` | Eindeutige ID des Datensatzes in Prism |
| `prism_access_group` | Zugriffsgruppe | Text | Gruppenname | Zugriffsberechtigungen auf Gruppenebene |
| `prism_access_user` | Zugriffsbenutzer | Text | Benutzername | Zugriffsberechtigungen auf Benutzerebene |
| `prism_source_system` | Quellsystem | Text | `poool` | Das System, aus dem die Daten stammen |
| `prism_updated_at` | Letztes Update (Prism) | Zeitstempel | z.B. `2024-11-10 14:30:00` | Zeitpunkt der letzten Aktualisierung in Prism |
| `prism_source_reference` | Quell-Referenz (JSON) | JSON-Objekt | `{"entity_model": "Project", ...}` | Technische Referenzen zur Poool-Datenbank |
| `prism_source_reference → entity_model` | Quell-Modell | Text | `Project` oder `ProjectPhase` | Typ des Quell-Objekts in Poool |
| `prism_source_reference → project_id` | Quell-ID: Projekt | Text | z.B. `12345` | Interne Poool-Projekt-ID |
| `prism_source_reference → projectPhase_id` | Quell-ID: Projektphase | Text | z.B. `67890` | Interne Poool-Projektphasen-ID |