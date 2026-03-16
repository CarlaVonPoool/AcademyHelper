# Nummernkreise

Article Description: Nummernkreise ermöglichen die automatische Vergabe von Rechnungsnummern, Angebotsnummern, Auftragsnummern, Projektnummern, Kundennummern, Ticketnummer, Bestellnummer, Inventarnummer, Lieferantennummer und andere Nummerierungen in Poool
Last Updated: 22. September 2025
Published: No
Suggested: No

# ⚙️Nummernkreise anlegen

In Poool haben Sie die Möglichkeit individuelle **Nummernkreise** für verschiedene Datensätze wie **Rechnungen**, **Angebote** oder **Kunden** anzupassen und zu verwalten. Diese Funktion ermöglicht es Ihnen, eine **einheitliche** und **organisierte Struktur** für die Nummerierung Ihrer **Dokumente** und **Datensätze** zu schaffen, die sowohl für Ihr **Unternehmen** als auch für Ihre **Kunden** leicht verständlich ist.

[https://vimeo.com/806347398](https://vimeo.com/806347398)

## ✅ Systemvoraussetzungen & Einstellungen

- Zugriff auf `System → Nummernkreisläufe`
- Kenntnisse über gewünschte Platzhalter (z. B. Datum, Projekttyp, Seriennummer)
- Definition von Startnummern und Stellenanzahl für Seriennummern

<aside>
💡

Achten Sie darauf, dass Änderungen an Nummernkreisen Auswirkungen auf bestehende Dokumente und Kalkulationen haben können.

</aside>

## 📌 Prozessbeschreibung: Schritt-für-Schritt Anleitung

**Schritt 1: Bereich öffnen**

- Gehen Sie zu `System → Nummernkreisläufe`.

![image.png](Nummernkreise/image.png)

**Schritt 2: Nummernkreis auswählen**

- Wählen Sie den gewünschten Nummernkreis (z. B. Angebot, Rechnung, Kunde) aus, um die Einstellungen zu bearbeiten.

**Schritt 3: Muster definieren**

- Legen Sie fest, wie sich der Nummernkreis zusammensetzen soll.
- Nutzen Sie **Platzhalter** (z. B. `{DATE:yyyy}`, `{SERIAL}`)
- Ergänzen Sie **Textelemente** (z. B. `AN`, `RE`)
- Schließen Sie optionale Teile in **[ ]** ein, wenn diese nicht für die Eindeutigkeitsprüfung herangezogen werden sollen.
    - **Beispiel 1**
        
        `[AN][-]{PROJECT-TYPE}[-]{DATE:yyyy}[-][INT][-]{SERIAL}` → `AN-DESIGN-2025-001`
        
    - **Beispiel 2**
        
        `[AN][-]{DATE:y}[-]{SERIAL}` → `AN-23-001`
        

**Schritt 4: Startwerte & Stellen festlegen**

- Definieren Sie die **Startnummer** und die **Anzahl der Stellen** für die Seriennummer (z. B. dreistellig mit führenden Nullen).
    
    ![image.png](Nummernkreise/image%201.png)
    

**Schritt 5: Speichern & testen**

- Überprüfen Sie das gewählte Muster und stellen Sie sicher, dass es logisch und konsistent aufgebaut ist.

## 🏗️ Mögliche Bestandteile eines Nummernkreises

Mithilfe dieser Bausteine können Sie einen individuellen Nummernkreis definieren – z. B. DESIGN-2025-001. Einige Platzhalter wirken sich auf die Eindeutigkeit der Projektnummer aus, andere sind rein informativ. Achten Sie bei der Erstellung auf eine sinnvolle Kombination und Struktur.

Dabei stehen folgende Platzhalter zur Verfügung: 

| **Platzhalter** | **Quelle** | **Beispiel** |
| --- | --- | --- |
| `{COMPANY-NAME-TOKEN}` | Das Kundenkürzel wird im CRM gepflegt. | `POOOL` |
| `{PROJECT-TYPE}` | Projekttyp wird beim Projekt ausgewählt. (Angelegt unter *Einstellungen → Projekte*) | `DESIGN` |
| `{PROJECT-TYPE-NUMBER}` | Projekttyp als Zahl, z. B. `01` für "Design", `02` für "Bau" | `01` |
| `{PROJECT-NUMBER}` | Interne Projektnummer (z. B. manuell gepflegt oder synchronisiert) | `2023-02` |
| `{CLIENT-NUMBER}` | Die Kundennummer aus dem CRM (Reiter *Kunde*) | `KD-0218` |
| `{DATE:yyyy}` | Jahr, vierstellig | `2025` |
| `{DATE:y}` | Jahr, zweistellig | `25` |
| `{DATE:m}` | Monat | `04` |
| `{DATE:d}` | Tag | `07` |
| `{DATE:ymd}` | Jahr-Monat-Tag | `20250715` |
| `{SERIAL}` | Fortlaufende Seriennummer (z. B. pro Jahr oder Projektart) | `001`, `045`, `1001` |

## 💼 Anwendungsfälle

- Einheitliche Nummerierung für Rechnungen, Angebote und Aufträge
- Automatische Vergabe von **Projektnummern** anhand von Datum, Projekttyp oder Kunde
- Unterstützung bei der eindeutigen Identifikation von Dokumenten im **CRM** und im **Buchhaltungsexport**

## ✅ Vorteile im Überblick

- **Klarheit**: Einheitliche Nummernstruktur für interne und externe Dokumente
- **Flexibilität**: Kombination aus Platzhaltern und Textelementen für individuelle Anforderungen
- **Automatisierung**: Fortlaufende Seriennummern ohne manuelles Nachführen

## 🔍 Zusammenfassung des Prozesses

Unter `System → Nummernkreisläufe` können Sie eigene Muster mit Platzhaltern und Textelementen definieren. So legen Sie fest, wie Ihre Angebote, Rechnungen oder Kunden systematisch durchnummeriert werden. Mit Startwerten, Seriennummern und optionalen Bausteinen stellen Sie sicher, dass Ihre Nummernkreise konsistent und eindeutig bleiben.

---

## 📊 Übersicht: Alle Nummernkreisläufe im Detail

| Nummerntyp | Mögliche Bestandteile | Beispiel |
| --- | --- | --- |
| **Rechnungsnummer** | Jahr, Monat, fortlaufende Nr. | 2025-01-0001 |
| **Angebotsnummer** | Jahr, Abteilung, fortlaufende Nr. | ANG-VT-2025-001 |
| **Auftragsnummer** | Jahr, Kundengruppe, fortlaufende Nr. | A-2025-B2B-0123 |
| **Projektnummer** | Projekttyp, Jahr, Kunde, fortlaufende Nr. | WEB-2025-BMW-001 |
| **Kundennummer** | Kundentyp, Region, fortlaufende Nr. | K-DE-10001 |
| **Lieferantennummer** | Land, Kategorie, fortlaufende Nr. | L-DE-IT-001 |
| **Eingangsrechnunsnummer** | Jahr, Lieferant, fortlaufende Nr. | ER-2025-001 |
| **Ticketnummer** | Priorität, Bereich, fortlaufende Nr. | T-HIGH-IT-0001 |
| **Bestellnummer** | Jahr, Abteilung, fortlaufende Nr. | BST-2025-001 |
| **Inventarnummer** | Standort, Kategorie, fortlaufende Nr. | INV-MUC-IT-001 |
| **Plannernummer** | Team, Quartal, fortlaufende Nr. | PLN-DEV-Q1-001 |

---

Vorheriger Artikel:

[Umsatzsteuersätze](Umsatzsteuers%C3%A4tze%20268015f5d93f80c3b408d895bb28fcc0.md) 

Nächster Artikel:

[Dokumenten- und Textvorlagen](../../Setup%20&%20Konfiguration/Poool%20Setup/Dokumenten-%20und%20Textvorlagen%20f9ca4ac980c74972bf8c9b248bb0d258.md) 

---

## Weiterführende Themen

[Firmendaten](../../Setup%20&%20Konfiguration/Poool%20Setup/Firmendaten%20d08490c6414f4fc2867ccb06cf18744b.md) 

[Preislisten](../../Setup%20&%20Konfiguration/Poool%20Setup/Preislisten%2032d214225e994d0998a815ac98ed3d95.md) 

[Anlage der Mitarbeiter](../../Setup%20&%20Konfiguration/Poool%20Setup/Anlage%20der%20Mitarbeiter%20546662b8693f4916890e027a8ad502d9.md) 

[Umsatzsteuersätze](../../Setup%20&%20Konfiguration/Poool%20Setup/Umsatzsteuers%C3%A4tze%205bc2aebafa8d40d9a820b05bfe19d57e.md) 

[Nummernkreise](../../Setup%20&%20Konfiguration/Poool%20Setup/Nummernkreise%20ff726faf208d43be813a91184cf85d5b.md) 

[Dokumenten- und Textvorlagen](../../Setup%20&%20Konfiguration/Poool%20Setup/Dokumenten-%20und%20Textvorlagen%20f9ca4ac980c74972bf8c9b248bb0d258.md) 

[Mahnwesen](../../Setup%20&%20Konfiguration/Poool%20Setup/Mahnwesen%2077f0369b035c4418a383c39f72cd5f1e.md)