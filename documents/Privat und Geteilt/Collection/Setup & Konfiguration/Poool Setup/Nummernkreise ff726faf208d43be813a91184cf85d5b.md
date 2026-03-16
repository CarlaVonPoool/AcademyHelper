# Nummernkreise

Article Description: Nummernkreise ermöglichen die automatische Vergabe von Rechnungsnummern, Angebotsnummern, Auftragsnummern und andere Nummerierungen in Poool
Last Updated: 18. August 2025
Published: Yes
Suggested: No

# Nummernkreise anlegen

In Poool haben Sie die Möglichkeit individuelle **Nummernkreise** für verschiedene Datensätze wie **Rechnungen**, **Angebote** oder **Kunden** anzupassen und zu verwalten. Diese Funktion ermöglicht es Ihnen, eine **einheitliche** und **organisierte Struktur** für die Nummerierung Ihrer **Dokumente** und **Datensätze** zu schaffen, die sowohl für Ihr **Unternehmen** als auch für Ihre **Kunden** leicht verständlich ist.

[https://vimeo.com/806347398](https://vimeo.com/806347398)

Um die **Nummernkreise** anzupassen, navigieren Sie zu `System → Nummernkreisläufe`. In diesem Bereich finden Sie eine Übersicht über die verschiedenen Arten von **Nummernkreise**, die Sie anpassen können. Wählen Sie den entsprechenden **Nummernkreis** aus, um das **Muster** und die **Einstellungen** zu bearbeiten. 

![nummernkreislauf.png](Nummernkreise/nummernkreislauf.png)

Die Anpassung des **Nummernkreises** basiert auf einem **Muster**, welches aus sogenannten **Platzhaltern** und **Textelementen** besteht. Poool bietet eine Reihe von vordefinierten **Platzhaltern**, die Ihnen dabei helfen, das **Muster** schnell und effizient zu erstellen. Einige dieser **Platzhalter** können Informationen wie das aktuelle **Datum**, den **Firmennamen** oder eine fortlaufende **Seriennummer** darstellen.

Bei der Erstellung des **Musters** ist es wichtig, die Struktur und die Reihenfolge der **Platzhalter** und **Textelemente** sorgfältig zu planen, um ein konsistentes und logisches **Nummerierungssystem** zu gewährleisten. 

💡 Achten Sie darauf, die richtige **Startnummer** und die **Anzahl der Stellen** für die fortlaufende **Nummerierung** anzugeben. Dies stellt sicher, dass Ihre **Nummernkreise** in Zukunft korrekt fortgeführt werden.

Zusätzlich zu den vordefinierten **Platzhaltern** können Sie auch eigene **Textelemente** in das **Muster** einfügen, um die **Nummernkreise** besser an Ihre individuellen Anforderungen anzupassen. Wenn bestimmte Teile des **Musters** **nicht** zur **Eindeutigkeitsprüfung** herangezogen werden sollen, sollten diese in eckigen Klammern **[ ]** eingeschlossen werden.

## **Beispiele für Nummernkreislauf-Syntax**

Beispiel: **AN-23-001**

Muster mit Platzhaltern: **AN[-]{DATE:y}[-]{SERIAL}**

Für die Eindeutigkeitsprüfung wird der Term AN23{SERIAL} verwendet. Die fortlaufende Seriennummer wird immer von der Eindeutigkeitsprüfung ausgenommen.

Beispiel: **RE-POOOL-2023-001**

Muster mit Platzhaltern: **RE[-]{COMPANY-NAME-TOKEN}-{DATE:yyyy}-{Serial}**

In diesem Beispiel wird der Firmenname und das Jahr in den **Nummernkreis** eingefügt. Die fortlaufende **Seriennummer** am Ende bleibt dabei stets eindeutig.

![nummernkreislauf-2.png](Nummernkreise/nummernkreislauf-2.png)

---

## Mögliche Bestandteile eines Nummernkreises

Die Anpassung des Nummernkreises basiert auf einem Muster, das aus festen **Textelementen** und dynamischen **Platzhaltern** besteht. Diese Platzhalter ziehen ihre Werte z. B. aus dem Datum, dem CRM oder den Projekteinstellungen.

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

### Eigene Textelemente & optionale Teile

Zusätzlich zu den Platzhaltern können Sie **statische Textelemente** in Ihr Muster einfügen – z. B. `AN`, `RE`, `AU`, etc. Teile des Musters, die **nicht zur Eindeutigkeitsprüfung** herangezogen werden sollen, können Sie in **eckige Klammern** setzen. 

- **Beispiel 1**
    
    `[AN][-]{PROJECT-TYPE}[-]{DATE:yyyy}[-][INT][-]{SERIAL}` → `AN-DESIGN-2025-001`
    
- **Beispiel 2**
    
    `[AN][-]{DATE:y}[-]{SERIAL}` → `AN-23-001`
    

<aside>
ℹ️

### Tipp:

Planen Sie die Struktur und Reihenfolge Ihrer Platzhalter sorgfältig. Eine saubere Logik hilft nicht nur bei der Wiedererkennung, sondern stellt auch sicher, dass Nummern eindeutig bleiben. Wichtig dabei:

- **Startnummer und Stellenanzahl** festlegen (z. B. immer dreistellig mit führenden Nullen)
- Optional verwendete Platzhalter klar definieren
- Einheitliche Nutzung der Projektarten (Zahl/Text) sicherstellen
</aside>

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

[Umsatzsteuersätze](Umsatzsteuers%C3%A4tze%205bc2aebafa8d40d9a820b05bfe19d57e.md) 

Nächster Artikel:

[Dokumenten- und Textvorlagen](Dokumenten-%20und%20Textvorlagen%20f9ca4ac980c74972bf8c9b248bb0d258.md) 

---

## Weiterführende Themen

[Firmendaten](Firmendaten%20d08490c6414f4fc2867ccb06cf18744b.md) 

[Preislisten](Preislisten%2032d214225e994d0998a815ac98ed3d95.md) 

[Anlage der Mitarbeiter](Anlage%20der%20Mitarbeiter%20546662b8693f4916890e027a8ad502d9.md) 

[Umsatzsteuersätze](Umsatzsteuers%C3%A4tze%205bc2aebafa8d40d9a820b05bfe19d57e.md) 

[Nummernkreise](Nummernkreise%20ff726faf208d43be813a91184cf85d5b.md) 

[Dokumenten- und Textvorlagen](Dokumenten-%20und%20Textvorlagen%20f9ca4ac980c74972bf8c9b248bb0d258.md) 

[Mahnwesen](Mahnwesen%2077f0369b035c4418a383c39f72cd5f1e.md)