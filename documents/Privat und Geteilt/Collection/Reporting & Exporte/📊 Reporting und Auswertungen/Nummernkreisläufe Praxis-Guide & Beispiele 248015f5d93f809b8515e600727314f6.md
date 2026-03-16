# Nummernkreisläufe: Praxis-Guide & Beispiele

Article Description: Nummernkreise als Strukturelement für einzelne Leistungsabgrenzungen. 
Published: Yes
Suggested: No
Neues UI 2026: No

Mit Nummernkreisläufen steuern Sie die automatische Vergabe von Nummern in Poool - von Rechnungs- und Projektnummern bis hin zu Kunden- und Lieferantennummern.

Diese Übersicht zeigt Ihnen:

- Welche Nummerntypen Sie anpassen können
- Bewährte Strukturen aus der Praxis
- Kreative Möglichkeiten für Ihre Organisation
- Wichtige Dos & Don'ts

<aside>
💡

**Tipp:** Die richtige Nummerierung spart Zeit, schafft Ordnung und hilft bei der Auswertung Ihrer Daten.

</aside>

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

## 💡 Erweiterte Möglichkeiten

**Besonders bei Projektnummern** können Sie viele Informationen codieren:

- **Projekttyp**: WEB, APP, DESIGN, SEO
- **Projekthierarchie**: Hauptprojekt, Teilprojekt, Unterprojekt
- **Standort**: MUC, BER, HAM, VIE, ZRH
- **Währung**: EUR, USD, CHF, GBP
- **Jahr**: 2025, 25
- **Kunde**: Kundenkürzel oder -nummer
- **Abteilung**: DEV, SALES, MARKETING
- **Fortlaufende Nummer**: 001, 002, 003...

## 📝 Praxisbeispiel: Bewährter Standard

**Empfohlene Standard-Struktur für Projektnummern:**

**Projekt/Hauptprojekt:** `[Kundenkürzel]-[Jahr]-[Nummer]`

- Beispiel: `BMW-2025-001`
- Beispiel: `BOSCH-2025-042`

**Teilprojekt:** `[Kundenkürzel]-[Jahr]-[Nummer]-[Teilprojektnummer]`

- Beispiel: `BMW-2025-001-01` (Teilprojekt 1 von BMW-2025-001)
- Beispiel: `BMW-2025-001-02` (Teilprojekt 2 von BMW-2025-001)
- Beispiel: `BOSCH-2025-042-A` (Teilprojekt A von BOSCH-2025-042)

Diese Struktur hat sich in der Praxis bewährt, weil sie:

- **Übersichtlich** bleibt
- Die **wichtigsten Infos** enthält (Kunde, Jahr, Projektnummer)
- **Hierarchien** klar abbildet (Teilprojekte hängen erkennbar am Hauptprojekt)
- Trotzdem **kurz und merkbar** ist

### **Weitere Beispiele für verschiedene Anwendungsfälle:**

**Standortbezogene Nummern:**

- `MUC-2025-001` = Projekt aus dem **Standort München**
- `BER-2025-047` = Projekt aus dem **Standort Berlin**
- `HAM-2025-023` = Projekt aus dem **Standort Hamburg**
- `R-MUC-2025-001` = **Rechnung** aus München
- `A-BER-2025-123` = **Angebot** aus Berlin

**Währungsbezogene Nummern:**

- `P-EUR-2025-001` = Projekt in **Euro**
- `P-USD-2025-042` = Projekt in **US-Dollar**
- `P-CHF-2025-015` = Projekt in **Schweizer Franken**
- `R-USD-2025-0123` = Rechnung in US-Dollar

---

<aside>
⚠️

### **Achtung: Die richtige Balance finden**

**Zu komplexe Projektnummern können kontraproduktiv sein!**

</aside>

Auch wenn es technisch möglich ist, sehr viele Informationen in die Projektnummer zu packen, sollten Sie bedenken:

- Lange Nummern sind **fehleranfällig** beim Abtippen
- Sie werden **unübersichtlich** in Listen und Berichten
- Mitarbeiter können sich komplexe Nummern **schwer merken**

<aside>
💡

**Empfehlung:** Nutzen Sie die Projektnummer nur für die **wichtigsten 2-3 Informationen**!

</aside>

**Weitere Projektinformationen können Sie besser über diese Funktionen abbilden:**

- **Tags**: Für flexible Kategorisierung (z.B. "Fremdwährung", "Eilauftrag", "VIP-Kunde")
- **Projektfelder**: Standort, Währung, Abteilung direkt am Projekt pflegen
- **Team & Verantwortliche**: Direkt am Projekt zuweisen
- **Projekttyp**: Als eigenes Feld am Projekt
- **Notizen**: Für zusätzliche Informationen

**Gute Balance:**

- ✅ `BMW-2025-001` (Standard: Kunde + Jahr + Nummer)
- ✅ `MUC-2025-042` (Standort + Jahr + Nummer)
- ✅ `WEB-BMW-001` (Typ + Kunde + Nummer)

<aside>
⚠️

**Zu komplex:**

- ❌ `TP-MUC-USD-WEB-2025-Q3-BMW-PRIORITY-001-A-FINAL`
</aside>

### 🔧 Tipps für die Praxis wenn es mehr sein soll

| Anforderung | Empfohlene Lösung |
| --- | --- |
| Jahreswechsel | mit Jahresprefix anlegen |
| Verschiedene Abteilungen | Prefix pro Abteilung (z.B. IT-, HR-, SALES-) |
| Mehrere Standorte | Standortkürzel verwenden (MUC-, BER-, HAM-) |
| Internationale Projekte | Länderkürzel einbauen (DE-, AT-, CH-) |
| Fremdwährungsprojekte | Währungskürzel integrieren (EUR-, USD-, CHF-) |
| Fortlaufende Nummerierung | Startnummer hoch setzen (z.B. 10000) |
| Testdaten erkennen | Eigenen Nummernkreis "TEST-" verwenden |
| Projekthierarchie abbilden | Zusätzliche Nummer für Teilprojekte anhängen |

---

## Weiterführende Themen

[Firmendaten](../../Setup%20&%20Konfiguration/Poool%20Setup/Firmendaten%20d08490c6414f4fc2867ccb06cf18744b.md) 

[Preislisten](../../Setup%20&%20Konfiguration/Poool%20Setup/Preislisten%2032d214225e994d0998a815ac98ed3d95.md) 

[Anlage der Mitarbeiter](../../Setup%20&%20Konfiguration/Poool%20Setup/Anlage%20der%20Mitarbeiter%20546662b8693f4916890e027a8ad502d9.md) 

[Umsatzsteuersätze](../../Setup%20&%20Konfiguration/Poool%20Setup/Umsatzsteuers%C3%A4tze%205bc2aebafa8d40d9a820b05bfe19d57e.md) 

[Nummernkreise](../../Setup%20&%20Konfiguration/Poool%20Setup/Nummernkreise%20ff726faf208d43be813a91184cf85d5b.md) 

[Dokumenten- und Textvorlagen](../../Setup%20&%20Konfiguration/Poool%20Setup/Dokumenten-%20und%20Textvorlagen%20f9ca4ac980c74972bf8c9b248bb0d258.md) 

[Mahnwesen](../../Setup%20&%20Konfiguration/Poool%20Setup/Mahnwesen%2077f0369b035c4418a383c39f72cd5f1e.md)