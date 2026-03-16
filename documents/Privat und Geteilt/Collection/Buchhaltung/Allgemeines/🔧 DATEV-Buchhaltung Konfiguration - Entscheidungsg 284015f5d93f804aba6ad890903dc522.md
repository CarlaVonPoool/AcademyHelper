# 🔧 DATEV-Buchhaltung Konfiguration - Entscheidungsguide

Article Description: Ihr kompletter Leitfaden zur optimalen DATEV-Buchhaltungsintegration in Poool - von der Nummernverwaltung bis zum automatisierten Export, mit allen wichtigen Entscheidungen strukturiert aufbereitet für Ihr Gespräch mit dem Steuerberater.
Published: No
Suggested: No

| **Artikel-Typ** | Tutorial |
| --- | --- |
| **Schwierigkeitsgrad** | 🚀🚀 Fortgeschritten |
| **Zeitaufwand** | ca. 30-45 Minuten (Phase 1 & 2) |

### Willkommen zur Buchhaltungsintegration!

Mit unserer jahrelangen Erfahrung in der Umsetzung zum papierlosen Büro und effizienter Buchhaltung begleiten wir Sie auf dem Weg zur digitalen Transformation. Wir wissen genau, worauf es ankommt und welche Stolpersteine es zu vermeiden gilt.

Um Ihre Buchhaltung optimal in Poool einzurichten, durchlaufen wir gemeinsam drei bewährte Phasen.

**Unser Ziel:** Wir bilden Ihre vorhandene Buchhaltung in Poool ab und gestalten die Prozesse so einfach wie möglich. Je mehr Informationen wir erhalten und je vollständiger die Stammdaten in Poool gepflegt sind, desto effizienter wird die Buchhaltung für beide Seiten - für Sie und Ihren Steuerberater.

### 📤 **Was liefert Poool bei jedem Export?**

- **Stammdaten-Export:** Alle neuen und geänderten Stammdaten werden automatisch mitgeliefert
- **Belegbilder:** Jede Rechnung wird digital mit übertragen
- **Buchungssätze:** Fertig vorkontiert gemäß Ihrer Konfiguration mit Verknüpfungslink zu den Belegen
- **Änderungen:** Der Steuerberater erhält automatisch alle Updates

**Der Clou:** Ihr Steuerberater kann direkt aus DATEV über den Link auf die Originalbelege zugreifen - kein Suchen mehr!

### 🤝 **Gemeinsam zum Erfolg**

**Nehmen Sie Ihren Steuerberater mit ins Boot!**

Aus unserer Erfahrung wissen wir: Ein gemeinsamer Abstimmungstermin mit Ihrem Steuerberater ist Gold wert. So bekommen wir alle Informationen aus erster Hand und können:

- Die optimale Konfiguration direkt abstimmen
- Missverständnisse vermeiden
- Rückfragen sofort klären
- Die Schnittstelle perfekt einrichten

**📞 Abstimmungstermin vereinbaren:**
Kontaktieren Sie uns für einen 3er-Termin mit Ihrem Steuerberater. In 30-45 Minuten klären wir alle technischen Details und Ihre individuellen Anforderungen.

## 🎯 Übersicht

**Gemeinsam treffen wir alle wichtigen Entscheidungen für Ihre Buchhaltungsintegration**

---

# 1.1 Nummernverwaltung

## 🎯 Die Kernfrage

**Sollen Kunden- und Lieferantennummern in Poool identisch mit den DATEV-Nummern sein?**

## ⚡ Schnellentscheidung

### ✅ Unsere Empfehlung:

**Option 1: DATEV-Nummern übernehmen + Poool zählt automatisch hoch**

## 🔍 Ihre Optionen

|  | ✅**Option 1: 
DATEV-Nummern übernehmen**  | **Option 2:
Unabhängige Nummerierung**  |
| --- | --- | --- |
| 🔧 **Funktionsweise** | Kunde 10456 heißt überall 10456 | Poool: KD001, KD002
DATEV: 10000, 10001 |
| ✅ **Vorteile** | • Eine Wahrheit
• Alle sprechen von derselben Nummer
• Keine Übersetzung nötig
• Saubere Reports | • Durchgehende Nummerierung in Poool
• Keine Lücken |
| ⚠️ **Nachteile** | • Eventuell Lücken in Nummernkreis | • Doppelte Verwaltung
• Verwirrung
• Fehleranfällig
• Mapping-Tabelle nötig |
| 💡 **Zusatz-Feature** | Optional: Präfix (KD-/LF-)
• Wird beim Export automatisch entfernt
• Bessere Übersicht in Poool | - |
| 🎯 **Empfohlen für** | ✅ **95% aller Fälle** | ⚠️ Nur wenn Steuerberater explizit fordert |

## 🤖 Wer vergibt neue Nummern?

|  | ✅**Option A:
Poool automatisch**  | **Option B:
Steuerberater manuell**  |
| --- | --- | --- |
| 📋 **Ablauf** | • Höchste DATEV-Nummer + 1
• Automatische Vergabe | • Anfrage an Steuerberater
• Wartezeit
• Nummer erhalten |
| ⚡ **Geschwindigkeit** | Sofort verfügbar | 1-3 Tage Verzögerung |
| 💼 **Aufwand** | Kein zusätzlicher Aufwand | Zusätzliche Kommunikation |
| 🎯 **Fehleranfälligkeit** | Keine Lücken oder Dubletten | Mögliche Fehler bei manueller Übertragung |
| ✅ **Empfohlen für** | **Standard-Empfehlung** | Nur bei speziellen Anforderungen |

## 📋 Ihre Entscheidung

### 📝 Hauptentscheidung:

| Option | Auswahl |
| --- | --- |
| ✅ **Option 1: DATEV-Nummern übernehmen** | [ ] |
| ❌ Option 2: Unabhängige Nummern | [ ] |

### 📝 Zusatz-Features:

| Feature | Auswahl |
| --- | --- |
| 💡 Präfix verwenden (KD-/LF-) | [ ] Ja [ ] **Nein** ✅ |

### 📝 Nummernvergabe:

| Option | Auswahl |
| --- | --- |
| ✅ **Option A: Poool automatisch** | [ ] |
| ❌ Option B: Steuerberater manuell | [ ] |

## 📝 Benötigte Infos vom Steuerberater:

- ✏️ Höchste Debitorennummer: _____
- ✏️ Höchste Kreditorennummer: _____
- ✏️ Sondernummernkreise vorhanden? Ja Nein

---

# 1.2 Kontierung Eingangsrechnungen

## 🎯 Die Kernfrage

**Wie sollen Eingangsrechnungen auf Aufwandskonten verteilt werden?**

## ⚡ Schnellentscheidung

### ✅ Unsere Empfehlung:

**Option 2: Feste Zuordnung im CRM**

## 🔍 Ihre Optionen

|  | **Option 1:
Manuelle Vorkontierung**  | **Option 2:
Feste Zuordnung im CRM** |
| --- | --- | --- |
| 🔧 **Funktionsweise** | • Flexible Kontierung pro Rechnung
• Aufteilung auf mehrere Konten möglich | • Jeder Lieferant = EIN festes Aufwandskonto
• Automatische Zuordnung |
| ✅ **Vorteile** | • Maximale Flexibilität
• Genaueste Kostenverteilung
• Aufteilungen möglich
• Weniger Stornobuchungen | • Schnell und automatisch
• Keine Schulung nötig
• Minimaler Aufwand |
| ⚠️ **Nachteile** | • Schulung erforderlich
• Manueller Aufwand | • Unflexibel
• Bei verschiedenen Kostenarten: Mehrere Lieferanten-Einträge nötig |
| 📊 **Benötigt** | **Konten-Matrix vom Steuerberater** | Liste mit festen Zuordnungen |
| 🎯 **Empfohlen für** | **Standard-Empfehlung**
• Mehr als 20 Lieferanten
• Variable Kostenarten | • Weniger als 20 Lieferanten
• Immer gleiche Kostenart pro Lieferant |

## 📊 Konten-Matrix Beispiel

### 📝 Allgemeine Regeln:

| 🏷️ **Kostenart** | 📊 **Konto** | 📝 **Beschreibung** |
| --- | --- | --- |
| 📦 Büromaterial | 4930 | Stifte, Papier, etc. |
| 📞 Telefon/Internet | 4920 | Telekom, Vodafone |
| ✈️ Reisekosten | 4670 | Hotel, Bahn, Flug |
| 🍽️ Bewirtung | 4650 | Geschäftsessen |
| 🚗 Fahrzeugkosten | 4530 | Benzin, Wartung |

### 📝 Lieferantenspezifische Regeln:

| 🏢 **Lieferant** | 📊 **Standard-Konto** | ⚠️ **Ausnahmen** |
| --- | --- | --- |
| 📦 Amazon | 4930 | IT → 4985, Bücher → 4940 |
| 📞 Deutsche Telekom | 4920 | Immer gleich |
| ⛽ Shell | 4530 | Immer Fahrzeugkosten |

## 📋 Ihre Entscheidung

### 📝 Kontierungsmethode:

| Option | Auswahl |
| --- | --- |
| ✅ **Option 1: Manuelle Vorkontierung** | [ ] |
| ❌ Option 2: Feste Zuordnung | [ ] |

### 📝 Bei Option 1 benötigt:

- [ ]  Konten-Matrix erstellen lassen
- [ ]  Mitarbeiterschulung einplanen

## 📝 Benötigte Infos vom Steuerberater:

- ✏️ Kontenrahmen: SKR03 SKR04
- ✏️ Konten-Matrix vorhanden? Ja Nein → erstellen lassen
- ✏️ Aufwandskonten am Lieferanten

---

# 1.3 Kontierung Ausgangsrechnungen

## 🎯 Die Kernfrage

**Wie sollen Ihre Umsätze auf Ertragskonten verteilt werden?**

## ⚡ Schnellentscheidung

### ✅ Unsere Empfehlung:

**Option 1: Standard-Kontierung über Steuersatz**

## 🔍 Ihre Optionen

|  | ✅ **Option 1:
Standard über Steuersatz**  | **Option 2:
Funktionsbezogene Ertragskonten** |
| --- | --- | --- |
| 🔧 **Funktionsweise** | • Steuersatz bestimmt Ertragskonto
• 19% → 8400
• 7% → 8300
• 0% → 8338 | • Getrennte Konten pro Geschäftsbereich
• Manuelle Zuordnung bei Rechnungserstellung |
| ✅ **Vorteile** | • Einfach und automatisch
• Keine Konfiguration
• Fehlerfrei | • Detaillierte Auswertung
• Besseres Controlling
• Profitabilität pro Bereich |
| ⚠️ **Nachteile** | • Keine Trennung nach Geschäftsbereich | • Konfigurationsaufwand
• Schulung nötig
• Fehleranfällig |
| 📊 **Benötigt** | Nichts - läuft automatisch | Liste der Geschäftsbereiche und Konten |
| 🎯 **Empfohlen für** | **90% der Unternehmen**
• Ein Hauptgeschäft
• Standard-Auswertungen ausreichend | • Mehrere Geschäftsbereiche
• Detailliertes Controlling gewünscht |

## 📋 Ihre Entscheidung

### 📝 Kontierungsmethode:

| Option | Auswahl |
| --- | --- |
| ✅ **Option 1: Standard über Steuersatz** | [ ] |
| ❌ Option 2: Funktionsbezogene Ertragskonten | [ ] |

### 📝 Bei Option 2 benötigt:

- [ ]  Liste der Geschäftsbereiche
- [ ]  Zuordnung zu Ertragskonten

---

# 1.4 Kreditkarten & PayPal

## 🎯 Die Kernfrage

**Wie sollen Zahlungen über Kreditkarte und PayPal verbucht werden?**

## ⚡ Schnellentscheidung

### ✅ Unsere Empfehlung:

**Option 1: Eigener Kreditor pro Zahlungsmittel**

## 🔍 Ihre Optionen

|  | **Option 1: Eigener Kreditor** ✅ | **Option 2: Verrechnungskonten** | **Option 3: Nur Belegtypen** |
| --- | --- | --- | --- |
| 🔧 **Funktionsweise** | • Pro Karte ein Kreditor
• Z.B. "Firmenkreditkarte VISA" = 79001 | • Buchung über Verrechnungskonten
• Z.B. Konto 1361 | • Unterscheidung nur über Nummernkreise |
| ✅ **Vorteile** | • Übersichtliche Abstimmung
• Klare Trennung
• Einfache Kontrolle | •Buchhaltungskonform
• Keine Kreditorenanlage | • Minimal-Setup |
| ⚠️ **Nachteile** | • Kreditorenanlage nötig | • Komplexere Abstimmung
• Mehr Buchungsaufwand | • Keine saubere Trennung
• Abstimmung schwierig |
| 🎯 **Empfohlen für** | **Standard-Empfehlung** | Wenn Steuerberater es bevorzugt | Nicht empfohlen |

## 📊 Beispiel-Setup für Option 1:

| 💳 **Zahlungsmittel** | 🔢 **Kreditor-Nr** | 📝 **Name im System** |
| --- | --- | --- |
| 💳 VISA Karte | 79001 | Firmenkreditkarte VISA |
| 💳 MasterCard | 79002 | Firmenkreditkarte MC |
| 💰 PayPal | 79003 | PayPal Geschäftskonto |
| 🛒 Amazon Business | 79004 | Amazon Business Konto |

## 📋 Ihre Entscheidung

### 📝 Verwaltungsmethode:

| Option | Auswahl |
| --- | --- |
| ✅ **Option 1: Eigener Kreditor** | [ ] |
| ⚠️ Option 2: Verrechnungskonten | [ ] |
| ❌ Option 3: Nur Belegtypen | [ ] |

### 📝 Welche Zahlungsmittel nutzen Sie?

- [ ]  Firmenkreditkarte(n) - Anzahl: ___
- [ ]  PayPal
- [ ]  Amazon Business
- [ ]  Weitere: _________

---

# 1.5 Zahlungsmittel & Bankkonten

## 🎯 Die Kernfrage

**Welche Zahlungswege gibt es und von welchen Konten wird gezahlt?**

## 🔍 Zahlungsmittel definieren

### 🚫 OHNE Zahlungsvorgang

*Automatisch als bezahlt markiert:*

| 💳 **Zahlungsmittel** | 🔧 **Besonderheit** | 📊 **Kreditor nötig?** |
| --- | --- | --- |
| 💳 Kreditkarte | Monatliche Abrechnung | Ja (z.B. 79001) |
| 💰 PayPal | Sofortabbuchung | Ja (z.B. 79002) |
| 🏦 Lastschrift | Einzugsermächtigung | Nein |
| 💳 EC-Karte | Sofortzahlung | Optional |

### ✅ MIT Zahlungsvorgang

*Manuelle Zahlung nötig:*

| 💸 **Zahlungsmittel** | 🏦 **Von welchem Konto?** | 📄 **SEPA möglich?** |
| --- | --- | --- |
| 💸 Überweisung | Hauptkonto | Ja |
| 🏦 SEPA-Zahlung | Definiertes Geschäftskonto | Ja |
| 💵 Barzahlung | Kasse | Nein |

## 📋 Ihre Bankkonten

| 🏦 **Verwendungszweck** | 🏢 **Bank** | 📝 **IBAN** | ✅ **Standard?** |
| --- | --- | --- | --- |
| 🏦 Hauptkonto |  | _________________ | [ ] |
| 💼 Nebenkonto |  | _________________ | [ ] |
| 📊 Projektkonto |  | _________________ | [ ] |

## 🎯 Standard-Zahlungsmittel pro Lieferant

| 🏢 **Lieferant** | 💳 **Standard-Zahlungsmittel** | 🔄 **Alternative** |
| --- | --- | --- |
| 📦 Amazon | Kreditkarte | - |
| 📞 Deutsche Telekom | SEPA-Lastschrift | Überweisung |
| 🏢 Büroausstatter | Überweisung | Kreditkarte |
| ⛽ Tankstelle | EC-Karte | - |

## 📋 Ihre Entscheidung

### 📝 Zahlungsmittel OHNE Zahlungsvorgang:

- [ ]  Kreditkarte(n) - Anzahl: ___
- [ ]  PayPal-Konten - Anzahl: ___
- [ ]  Lastschriftverfahren
- [ ]  EC-Karten - Anzahl: ___

### 📝 Zahlungsmittel MIT Zahlungsvorgang:

- [ ]  SEPA-Überweisung
- [ ]  Standard-Überweisung
- [ ]  Barzahlung/Kasse

---

# 1.6 Belegverwaltung

## 🎯 Die Kernfrage

**Welche Kosten erfassen und wie kommen Belege ins System?**

## 🔍 Teil 1: Welche Kosten erfassen?

|  | **Option 1: Alle Kosten** ✅ | **Option 2: Nur Teilkosten** |
| --- | --- | --- |
| 📊 **Umfang** | Wirklich alles | • Nur Fixkosten
• Nur Gemeinkosten
• Nur Einzelkosten |
| ✅ **Vorteile** | • Vollständige Transparenz
• Optimale Auswertungen
• Keine Lücken | • Weniger Aufwand
• Übersichtlicher |
| ⚠️ **Nachteile** | • Mehr Belege zu verarbeiten | • Unvollständige Kostensicht
• Lücken in Auswertungen |
| 🎯 **Empfohlen für** | **Standard-Empfehlung** | Sehr kleine Unternehmen |

### 👥 Spezialfall: Mitarbeiterkosten

| 💳 **Art** | 🔧 **Umsetzung** | 📝 **Beispiel** |
| --- | --- | --- |
| 💳 Mitarbeiter-Kreditkarten | Eigene Kreditorenkonten | Max Mustermann = 79101 |
| 📄 Spesenabrechnungen | Sammelkonto oder Individual | Reisekosten, Bewirtung |
| 💳 EC-Karten | Pro Mitarbeiter Kreditor | Tankkosten, kleine Einkäufe |

## 🔍 Teil 2: Wie kommen Belege ins System?

|  | **Option 1: Kombination** ✅ | **Option 2: Nur E-Mail** | **Option 3: Nur Upload** |
| --- | --- | --- | --- |
| 📧 **E-Mail** | ✅ rechnungen@firma.de | ✅ Zentrale Adresse | ❌ Nicht verfügbar |
| 📤 **Upload** | ✅ Direkter Upload | ❌ Nicht verfügbar | ✅ Nur manuell |
| ✅ **Vorteile** | • Maximale Flexibilität
• Verschiedene Wege
• Backup-Option | • Automatisch
• Zentral
• Einfach | • Kontrolle
• Sofort verfügbar |
| ⚠️ **Nachteile** | • Zwei Kanäle zu überwachen | • Nur ein Weg | • Nur manuell |
| 🎯 **Empfohlen für** | **Standard-Empfehlung** | Kleine Teams | Sehr kleine Teams |

## 📋 Ihre Entscheidung

### 📝 Kostenumfang:

| Option | Auswahl |
| --- | --- |
| ✅ **Option 1: Alle Kosten** | [ ] |
| ⚠️ Option 2: Nur Teilkosten | [ ] |

### 📝 Mitarbeiterkosten:

- [ ]  Mit erfassen → Eigene Kreditoren Sammelkonto
- [ ]  Separat verwalten

### 📝 Belegeingang:

| Option | Auswahl |
| --- | --- |
| ✅ **Option 1: Kombination** | [ ] |
| ⚠️ Option 2: Nur E-Mail | [ ] |
| ⚠️ Option 3: Nur Upload | [ ] |

### 📝 Bei E-Mail:

- 📧 Adresse: rechnungen@_________

---

# 1.7 Datentransfer zum Steuerberater

## 🎯 Die Kernfrage

**Wie und wie oft übertragen wir die Daten an den Steuerberater?**

## 🔍 Wer macht den Export?

|  | **Option 1: Sie exportieren selbst** ✅ | **Option 2: Steuerberater hat Zugang** |
| --- | --- | --- |
| 🔧 **Ablauf** | • Sie erstellen Export
• Download ZIP-Datei
• Versand an StB | • StB loggt sich ein
• Erstellt Export selbst
• Direkter Download |
| ✅ **Vorteile** | • Volle Kontrolle
• Überblick was exportiert wird
• Flexibilität | • Kein Aufwand für Sie
• StB kann bei Bedarf |
| ⚠️ **Nachteile** | • Monatlicher Aufwand | • Keine Kontrolle
• Datenschutz-Thema
• Abhängigkeit |
| 🎯 **Empfohlen für** | **Standard-Empfehlung** | Nur bei vollem Vertrauen |

## 📅 Export-Rhythmus

| 📅 **Rhythmus** | 🎯 **Empfohlen für** | 💼 **Aufwand** |
| --- | --- | --- |
| 📅 **Monatlich** ✅ | Standard-Empfehlung | Optimal |
| 📅 Quartalsweise | Kleine Unternehmen (<50 Belege) | Gering |
| 📅 Wöchentlich | Große Volumen (>500 Belege) | Hoch |
| 📅 Bei Bedarf | Sehr kleine Unternehmen | Variabel |

## 🔒 Übertragungsweg

| 🔒 **Methode** | 🔐 **Sicherheit** | 💡 **Komfort** | 🎯 **Empfehlung** |
| --- | --- | --- | --- |
| ☁️ **Cloud** | ✅ Verschlüsselt | ✅ Praktisch | **Standard** ✅ |
| 📧 E-Mail | ⚠️ Unverschlüsselt | ✅ Einfach | Nicht empfohlen |
| 🏢 DATEV-Portal | ✅ Höchste Sicherheit | ⚠️ Aufwändig | Bei Anforderung |

### ☁️ Cloud-Optionen:

- [ ]  Dropbox
- [ ]  Google Drive
- [ ]  OneDrive
- [ ]  WeTransfer
- [ ]  Eigene Cloud: _________

## 📋 Ihre Entscheidung

### 📝 Export-Verantwortung:

| Option | Auswahl |
| --- | --- |
| ✅ **Option 1: Wir exportieren selbst** | [ ] |
| ⚠️ Option 2: Steuerberater bekommt Zugang | [ ] |

### 📝 Export-Rhythmus:

- [ ]  Wöchentlich
- [ ]  **Monatlich** ✅
- [ ]  Quartalsweise
- [ ]  Bei Bedarf

### 📝 Übertragungsweg:

- [ ]  E-Mail (nicht empfohlen)
- [ ]  **Cloud** ✅ → Welche: _________
- [ ]  DATEV-Portal

### 📝 Termine:

- 📅 Export immer am: ___ des Monats
- 📅 Übertragung bis: ___ des Folgemonats

---

# 1.8 Belegtypen & Nummernkreise

## 🎯 Die Kernfrage

**Wie strukturieren wir die Belegnummern für optimale Übersicht?**

## 🔍 Ihre Optionen

### 📊 Option 1: Ein Mandant (häufigster Fall) ✅

| 📄 **Belegtyp** | 📝 **Beispiel** | 🎯 **Verwendung** |
| --- | --- | --- |
| 📄 **ER** | ER-0001 | Eingangsrechnungen |
| 💳 **KK** | KK-00001 | Kreditkarte |
| 💰 **PP** | PP-00001 | PayPal |
| 💵 **KA** | KA-00001 | Kasse/Bar |
| 👥 **SP** | SP-00001 | Spesen |

### 🏢 Option 2: Mehrere Mandanten

| 🏢 **Mandant** | 🔢 **Nummernkreis** | 📝 **Beispiel** | 🏢 **Firma** |
| --- | --- | --- | --- |
| 🏢 **A** | 1.000.000 - 1.999.999 | A-1000001 | Hauptfirma GmbH |
| 🏢 **B** | 2.000.000 - 2.999.999 | B-2000001 | Tochterfirma GmbH |
| 🏢 **C** | 3.000.000 - 3.999.999 | C-3000001 | Holding AG |

## 📋 Ihre Entscheidung

### 📝 Unternehmensstruktur:

| Option | Auswahl |
| --- | --- |
| ✅ **Ein Mandant** | [ ] |
| ⚠️ Mehrere Mandanten → Anzahl: ___ | [ ] |

### 📝 Bei einem Mandanten - Belegtypen nutzen:

- [ ]  Eingangsrechnungen (ER)
- [ ]  Kreditkarte (KK)
- [ ]  PayPal (PP)
- [ ]  Kasse/Bar (KA)
- [ ]  Spesen (SP)
- [ ]  Weitere: _________

---

# 1.9 Optionale Features

## 🎯 Die Kernfrage

**Welche Zusatzfunktionen möchten Sie nutzen?**

## 🔍 Verfügbare Features

### 📊 Kostenstellenrechnung

| 🔧 **Aspekt** | 📝 **Details** |
| --- | --- |
| 💡 **Was ist das?** | • Aufteilung der Kosten auf Abteilungen/Projekte
• Detaillierte Profitabilitätsanalyse |
| ✅ **Vorteile** | • Bessere Kostenkontrolle
• Projektrentabilität erkennbar |
| 📊 **Benötigt** | • Kostenstellenliste vom Steuerberater
• Zuordnungsregeln |
| 🎯 **Empfohlen bei** | • Mehr als 20 Mitarbeiter
• Projektgeschäft
• Mehrere Standorte |

### ⏰ Zahlungsziele & Fälligkeit

| 🔧 **Feature** | ✅ **Vorteile** | 📊 **Benötigt** |
| --- | --- | --- |
| ⏰ **Zahlungsziele** | • Keine verpassten Skonti
• Automatische Berechnung | Zahlungsziele pro Lieferant |
| 📅 **Fälligkeitsverwaltung** | • Dashboard mit Ampelsystem
• E-Mail-Erinnerungen | Aktivierung genügt |

### 💶 SEPA-Dateien (EMPFOHLEN) ✅

| 🔧 **Aspekt** | 📝 **Details** |
| --- | --- |
| 💡 **Was ist das?** | Zahlungsdateien direkt aus Poool für Online-Banking |
| ✅ **Vorteile** | • 5+ Stunden Zeitersparnis/Monat
• Keine Tippfehler
• Automatische Zuordnung |
| 📊 **Voraussetzung** | • IBANs vollständig gepflegt
• Bankverbindungen hinterlegt |
| 🎯 **Empfehlung** | **Unbedingt aktivieren!** |

# 1.10 Sonderanforderungen

## 🎯 Die Kernfrage

**Haben Sie spezielle Anforderungen oder Sonderfälle in Ihrer Buchhaltung?**

## 🔍 Häufige Sonderfälle

### 📄 Gutschriften & Eigenbelege

### 💼 Besondere Geschäftsvorfälle

### 🏢 Organisatorische Besonderheiten

### 🔄 Wiederkehrende Buchungen

### 📝 Sonstige Anforderungen:

## 📋 Ihre Entscheidung

### 📝 Welche Features aktivieren?

**💰 Zahlungsverkehr:**

- [ ]  **SEPA-Dateien erstellen** ✅ EMPFOHLEN

**📊 Kostenkontrolle:**

- [ ]  Kostenstellenrechnung → Liste vom StB anfordern
- [ ]  Zahlungsziele pflegen
- [ ]  Fälligkeitsverwaltung

**📤 Export-Optionen:**

- [ ]  Separate Exporte nach Belegtypen
- [ ]  Separate Exporte nach Mandanten

### 📝 Aktivierung:

- [ ]  Sofort alle gewählten
- [ ]  Schrittweise nach Go-Live
- [ ]  Erstmal ohne, später ergänzen

---

# 📊 Zusammenfassung & Checkliste

## ✅ Für den Steuerberater-Termin

### 📝 Grunddaten erfragen:

- [ ]  Beraternummer
- [ ]  Mandantennummer
- [ ]  Höchste Debitoren-/Kreditorennummer
- [ ]  Kontenrahmen (SKR03/04)

### 📝 Listen anfordern:

**📊 Bei manueller Kontierung (Option 1):**

- [ ]  Aufwandskonten-Liste
- [ ]  Lieferantenliste mit IBANs
- [ ]  **Konten-Matrix** (wichtig!)

**📊 Bei fester Kontierung (Option 2):**

- [ ]  Lieferantenliste mit Kreditorennummer + IBAN + Aufwandskonto

### 📝 Weitere Infos:

- [ ]  Bankkonten (alle IBANs)
- [ ]  Kostenstellenliste (falls gewünscht)
- [ ]  Bevorzugter Datenübertragungsweg
- [ ]  Deadline für Monatsabschluss

---

## 💡 Gut zu wissen

- ✅ Alle Entscheidungen können später angepasst werden
- ✅ Starten Sie einfach, erweitern Sie bei Bedarf
- ✅ Der Steuerberater ist Ihr wichtigster Partner
- ✅ Wir unterstützen Sie bei jedem Schritt

**🚀 Bereit? Dann lassen Sie uns gemeinsam starten!**