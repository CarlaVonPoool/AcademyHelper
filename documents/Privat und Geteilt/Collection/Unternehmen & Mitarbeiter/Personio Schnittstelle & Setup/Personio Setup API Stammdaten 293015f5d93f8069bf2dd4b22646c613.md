# Personio Setup API: Stammdaten

Article Description: Mitarbeiter-Stammdaten automatisch synchronisieren über die Poool-Personio Schnittstelle.
Last Updated: 21. Oktober 2025
Published: No
Suggested: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀🚀 Praxis |

Über die API-Schnittstelle zwischen **Personio und Poool** werden Mitarbeiter-Stammdaten automatisch synchronisiert. Damit ersparen Sie sich doppelte Pflege und stellen sicher, dass Rollen, Arbeitszeiten und Abteilungen in beiden Systemen immer aktuell sind. In dieser Anleitung zeigen wir Ihnen Schritt für Schritt, wie Sie die Verbindung gemeinsam mit dem Poool-Support einrichten und testen können.

## ✅ Systemvoraussetzungen & Einstellungen

- Zugriff auf die **Personio API**
- **API Client ID** und **API Secret** aus Personio
- Anfrage an **Poool-Support** zur API-Einrichtung

## 📌 Prozessbeschreibung: Schritt-für-Schritt Anleitung

- **Schritt 1:** API-Daten in Personio zur Verfügung stellen
    - In Personio neuen API-Client anlegen
    - Berechtigung für den Bereich **Employees** aktivieren
    - **Client ID** und **Client Secret** generieren
- **Schritt 2:** API-Zugangsdaten an Poool übermitteln
    - Die erzeugten Zugangsdaten sicher an den Poool-Support übergeben
    - Der Support hinterlegt die Daten serverseitig in Poool
- **Schritt 3:** Felder und Mapping prüfen
    - Gemeinsam mit dem Poool-Team prüfen, ob alle **benötigten Felder** (z. B. Name, E-Mail, Abteilung, Anstellungsart, Arbeitszeitmodell) von Personio übergeben werden
    - Abgleich: welche Felder werden in Poool importiert, welche sind optional
- **Schritt 4:** Mapping in Poool konfigurieren
    - Das Support-Team liefert einen Vorschlag wie die Personio-Felder in Poool abgebildet werden (f*irst_name* → *Vorname*, *department* → *Abteilung)*
- **Schritt 5:** Test in der Testumgebung
    - In einer separaten Testinstanz werden die Daten erstmalig eingelesen
    - Überprüfung: sind alle Mitarbeiter:innen und Felder korrekt importiert?
    - Korrekturen am Mapping können hier noch problemlos vorgenommen werden
- **Schritt 6:** Go-Live der Schnittstelle
    - Nach erfolgreichem Test wird die Synchronisierung in der Live-Umgebung aktiviert
    - Personio überträgt ab sofort alle Stammdaten regelmäßig automatisch an Poool

## 🧩 Verfügbare Felder für das Mapping

- Vorname
- Nachname
- E-Mail
- Kürzel
- Aktiver Benutzer
- Sprache
- Geburtstag
- Eintrittsdatum
- Austrittsdatum
- X-Board Widget Globale Einstellung
- Kanban als Startseite
- Verantwortlich für Abwesenheiten
- Funktion
- Funktionstitel
- Funktion gültig ab
- Mitarbeiterkonto
- Gehalt
- Mandant / Firmenzugehörigkeit
- Urlaubstage
- Feiertagskalender
- Zeiterfassung verpflichtend
- Gleitzeitkonto führen
- Arbeitszeit Startdatum
- Arbeitsbeginn (Uhrzeit)
- Pause Startzeit (Uhrzeit)
- Pause Dauer (Zeit)
- Max. Überstunden (Stunden)

## 💼 Anwendungsfälle

- **Automatische Pflege von Mitarbeiterstammdaten** in Poool
- **Vermeidung von manuellen Eingabefehlern** bei Abteilungen, Rollen, Arbeitszeiten
- **Nahtloser Datenfluss** zwischen HR- und Projektmanagementsystem

## ✅ Vorteile im Überblick

- **Zeitersparnis:** Kein doppeltes Pflegen von Mitarbeiterdaten in zwei Systemen
- **Datenkonsistenz:** Änderungen in Personio werden automatisch an Poool übergeben
- **Transparenz:** Einheitliche Stammdatenbasis für Projekte, Abrechnungen und Reporting
- **Sicherheit:** Gemeinsames Setup mit Poool-Support sorgt für korrekte Authentifizierung

---

## 🔍 **Zusammenfassung des Prozesses**

- Erstellung eines API-Clients in Personio
- Übergabe der Zugangsdaten an Poool
- Gemeinsame Prüfung des Feld-Mappings (Abgleich der Datenfelder beider Systeme)
- Test des Datentransfers in einer Testumgebung
- Nach erfolgreichem Test: dauerhafte, automatisierte Synchronisation im Hintergrund