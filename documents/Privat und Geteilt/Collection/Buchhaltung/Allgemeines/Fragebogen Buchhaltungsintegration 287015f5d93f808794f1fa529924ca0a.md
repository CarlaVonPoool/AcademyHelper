# Fragebogen Buchhaltungsintegration

Published: No
Suggested: No

**DATEV / BMD Setup für Poool**

---

## 👋 Kurz vorab

Wir richten Ihre Buchhaltungsintegration ein. Dieser Fragebogen hilft uns dabei.

**Zeitaufwand:** ~15 Minuten | **Rückfragen:** buchhaltung@poool.cc

---

## 1️⃣ Grunddaten

### Buchhaltungssystem

| System | Ihre Wahl |
| --- | --- |
| **DATEV** (Deutschland) |  |
| **BMD** (Österreich) |  |
| **Anderes** | System: |

### Stammdaten

|  | Ihre Angabe |
| --- | --- |
| **Beraternummer** |  |
| **Mandantennummer** |  |

---

## 2️⃣ Aktuelle Arbeitsweise

### Wie arbeiten Sie heute mit Belegen?

### **Option A: Direktübertragung**

- [ ]  **DATEV Unternehmen Online** - Alle Belege landen direkt dort
- [ ]  **BMD.COM** - Alle Belege landen direkt dort

### **Option B: Export aus anderem System**

- [ ]  Wir exportieren aus:                               und importieren dann

### **Option C: Manuelle Erfassung**

- [ ]  Wir buchen alles manuell

## ⚠️ Bei Direktübertragung (DATEV UO / BMD.COM)

### Soll das so bleiben?

- [ ]  **Ja, weiter so** → Belege weiter direkt in DATEV Unternehmen Online / BMD com hochladen.

**➡️ Sie können zu Abschnitt [X] springen**

- [ ]  **Nein, auf Export umstellen** → Wir möchten alle Kosten in poool haben und exportieren.

**➡️ Bitte alle folgenden Abschnitte ausfüllen**

- [ ]  **Teilweise** → Nur bestimmte Belege (z.b. Projektbezogene) in poool, Rest direkt in DATEV Unternehmen Online / BMD com

**➡️ Bitte alle folgenden Abschnitte ausfüllen**

---

## 3️⃣ Zukünftige Arbeitsweise - Kontierung

### Wie möchten Sie Eingangsrechnungen kontieren?

**Diese Entscheidung bestimmt Ihren täglichen Arbeitsaufwand:**

- [ ]  **Option A: Automatisch über Lieferantenstamm** ✅
    - Wir hinterlegen pro Lieferant:
        - Kreditorennummer
        - Standard-Aufwandskonto
        - Zahlungsmittel (Bank / Lastschrift)
    - **Vorteil:** Rechnung anlegen = fertig kontiert
    - **Nachteil:** Keine Flexibilität - Zuordnung kann vom Kunden nicht geändert werden
    - **Lösung bei mehreren Konten:** Lieferant mehrfach anlegen mit der gleichen Kreditorennummer, jedoch unterschiedlichem Aufwandskonto (z.B. "Amazon - Büro", "Amazon - Waren", "Amazon - Marketing") , Dafür müssen es aber unterschiedliche Rechnungen sein.
    - **Aufwand:** Einmalige Einrichtung, dann automatisch
    - **Ideal für:** Unternehmen mit festen Lieferanten-Konten-Zuordnungen
    - **PLUS: Wiederkehrende Rechnungen** 🔄
        - Für Miete, Versicherungen, Abos etc.
        - Vorlagen mit fester Kontierung
        - Automatische Erstellung nach Rhythmus
        - Nur noch Beleg zuordnen

- [ ]  **Option B: Manuelle Vorkontierung** 📝
- Zahlungsmittel wird am Lieferant hinterlegt (für SEPA)
- Bei jeder Rechnung wählt der Mitarbeiter das Aufwandskonto
- **PLUS: Wiederkehrende Rechnungen** 🔄
    - Für Miete, Versicherungen, Abos etc.
    - Vorlagen mit fester Kontierung
    - Automatische Erstellung nach Rhythmus
    - Nur noch Beleg zuordnen
- **Vorteil:** Maximale Flexibilität - jede Rechnung individuell
- **Nachteil:** Mehr Aufwand pro Rechnung
- **Aufwand:** 1-2 Min pro Rechnung (außer wiederkehrende)
- **Ideal für:** Unternehmen mit wechselnden Aufwandskonten oder Lieferanten mit mehreren Konten

---

### Wiederkehrende Rechnungen

**Welche wiederkehrenden Rechnungen haben Sie?**

| Typ | Vorhanden | Rhythmus | Beispiele |
| --- | --- | --- | --- |
| **Miete/Pacht** |  | [ ] Monatlich [ ] Quartalsweise | _________ |
| **Versicherungen** |  | [ ] Monatlich [ ] Jährlich | _________ |
| **Software-Abos** |  | [ ] Monatlich [ ] Jährlich | _________ |
| **Leasing** |  | [ ] Monatlich | _________ |
| **Telefon/Internet** |  | [ ] Monatlich | _________ |
| **Strom/Gas** |  | [ ] Monatlich [ ] Abschlag | _________ |
| **_________** |  | _________ | _________ |

### SEPA-Zahlungsdateien erwünscht?

- [ ]  **Ja** → Spart ~5h/Monat bei Zahlungsläufen

[ ] **Nein** → Manuelle Überweisung

---

## 4️⃣ Nummernkreise (nur bei Export-Lösung)

### Können wir die Kunden-/Lieferantennummern für die Buchhaltung nutzen?

<aside>
💡

**Hintergrund:** In der Buchhaltung werden Kunden als Debitoren und Lieferanten als Kreditoren geführt. Idealerweise verwenden wir dafür die gleichen Nummern wie in Poool.

</aside>

### Nutzt der Kunde die Nummern in Poool anderweitig?

# **Nein - Nummern sind frei verfügbar** ✅

- Wir passen die Nummern an Ihre Buchhaltung an
- **Vorteile:**
    - ✅ Automatische Hochzählung bei neuen Einträgen
    - ✅ Keine Dubletten möglich
    - ✅ Unabhängig vom Steuerberater
    - ✅ Gleiche Nummerierung in allen Auswertungen
    - ✅ Debitorennummer auf Ausgangsrechnungen = Verwendungszweck für Überweisungen
    - ✅ Nahtlose Integration
- **So funktioniert's:** Lieferantennummer = Kreditorennummer (z.B. L-0037)
- **Benötigte Infos:**
    - Höchste Debitorennummer in DATEV/BMD: _________
    - Höchste Kreditorennummer in DATEV/BMD: _________

<aside>
💡

### Präfix-Option (für beide Varianten)

**Präfixe für bessere Lesbarkeit:**

- Wir können IMMER Präfixe verwenden (z.B. KD-10234, L-70567)
- Verbessert die Übersichtlichkeit im System
- Wird beim Export automatisch entfernt
- Buchhaltung erhält die reinen Nummern

**Möchten Sie Präfixe nutzen?**
[ ] **Ja** → Kunden: _____ Lieferanten: _____
[ ] **Nein, reine Nummern**

</aside>

# **Ja - Nummern werden anderweitig genutzt**

- **Lösung - je nach Kontierungsmethode:**
    
    **Bei Option A (automatische Kontierung):**
    
    - Feld "Kundennummer" = Kreditorennummer
    - Feld "Buchhaltungskonto" = Aufwandskonto
    - Beide Felder werden im Lieferantenstamm gepflegt
    
    **Bei Option B (manuelle Kontierung):**
    
    - Feld "Buchhaltungskonto" = Kreditorennummer
    - Aufwandskonto wird bei jeder Rechnung manuell gewählt
    - Nur Kreditorennummer im Lieferantenstamm
- **Nachteile:**
    - ⚠️ Keine automatische Hochzählung
    - ⚠️ Manuelle Pflege nötig
    - ⚠️ Abhängigkeit vom Steuerberater bei neuen Nummern
- **Nummernvergabe:**
    - [ ]  Steuerberater vergibt neue Nummern (Anfrage → Vergabe → Eintragen)
    - [ ]  Kunde pflegt eigene Liste/Excel
    - [ ]  Andere: _________
- **Aktuell höchste Nummern in DATEV/BMD:**
    - Debitoren: _________
    - Kreditoren: _________

---