# Projektauswertung: Deine Schaltzentrale für profitables Projektmanagement

Article Description: Die richtige Vorgehensweise für die Projektauswertung.
Last Updated: 20. August 2025
Published: Yes
Suggested: No
Neues UI 2026: No

## 📊 Artikel-Metadaten

- **Artikel-Typ:** Best Practice
- **Schwierigkeitsgrad:** 🚀🚀 Praxis

## Grundlegendes Verständnis der Projektauswertung

<aside>
💡

Hier sind die Erläuterungen zu den Kennzahlen zu finden:

[Kennzahlen im Reporting](../%F0%9F%93%8A%20Reporting%20und%20Auswertungen/Kennzahlen%20im%20Reporting%20b5f3387f004d496bb209d8975b39e68a.md)

</aside>

Die Projektauswertung zeigt dir auf einen Blick die finanzielle Situation deiner Projekte. Bevor wir in die Details gehen, ist es wichtig zu verstehen, welche Werte sich durch die verschiedenen Button-Kombinationen verändern und welche immer gleich bleiben.

![image.png](Projektauswertung%20Deine%20Schaltzentrale%20f%C3%BCr%20profita/image.png)

### Was bleibt immer gleich?

Egal welche Button-Kombination du wählst, diese Werte ändern sich nicht:

- **Abgerechnet:** Die bereits in Rechnung gestellten Beträge
- **Kosten:** Die Einkaufskosten (EK) für Fremdleistungen
- **Stunden:** Die tatsächlich geleisteten Arbeitsstunden
- **Auftrag:** Das noch offene beauftragte Projektvolumen
- **Auftrag + Abgerechnet:** Der Gesamtwert

### Was kann sich verändern?

Die drei Button-Bereiche beeinflussen unterschiedliche Spalten:

1. **Stundenkosten-Button:** Verändert "Soll Honorar" ↔ "Interne Kosten"
2. **Rohertrag-Button:** Verändert die Basis der Rohertrag-Berechnung
3. **Rohertrag/Diff %-Button:** Wandelt Euro-Beträge in Prozente um

## Vergleichstabellen: So verändern sich die Werte

### Prozent-Button aktivieren (Rohertrag/Diff: %)

![image.png](Projektauswertung%20Deine%20Schaltzentrale%20f%C3%BCr%20profita/image%201.png)

| Rohertrag/Diff Button | Abgerechnet | Kosten | Rohertrag | Soll Honorar | Diff Soll-Ist | Stunden | Auftrag | Auftrag + Abgerechnet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **€ (deaktiviert)** | € 88.441 | € 10.750 | € 77.691 | € 27.402 | € 50.289 | 201:54 | € 245.562 | € 334.003 |
| **% (aktiviert)** | € 88.441 | € 10.750 | **88%** | € 27.402 | **57%** | 201:54 | € 245.562 | € 334.003 |

**Was passiert beim Aktivieren des %-Buttons?**

Der %-Button ist der einfachste - er wandelt nur die Darstellung um:

- **Rohertrag:** € 77.691 → 88%
    - Zeigt das Verhältnis von Rohertrag zu Abgerechnet
    - Berechnung: (Rohertrag / Abgerechnet) × 100
- **Diff Soll-Ist:** € 50.289 → 57%
    - Zeigt prozentual, wie profitabel das Projekt ist
    - Höher = bessere Rendite

**Alle anderen Werte bleiben unverändert!**

**Wann ist diese Ansicht nützlich?**

- Für schnellen Überblick über die Profitabilität
- Zum Vergleich verschiedener Projekte unabhängig von ihrer Größe
- Für Management-Reports ("Projekt X hat 88% Profit")

### Rohertrag-Button auf "Auftrag + Ausgangsrechnung" umschalten

![image.png](Projektauswertung%20Deine%20Schaltzentrale%20f%C3%BCr%20profita/image%202.png)

| Rohertrag Button | Abgerechnet | Kosten | Rohertrag | Soll Honorar | Diff Soll-Ist | Stunden | Auftrag | Auftrag + Abgerechnet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Abgerechnet** | € 88.441 | € 10.750 | € 77.691 | € 27.402 | € 50.289 | 201:54 | € 245.562 | € 334.003 |
| **Auftrag+Ausgangsrechnung** | € 88.441 | € 10.750 | **€ 323.253** | € 27.402 | **€ 295.851** | 201:54 | € 245.562 | € 334.003 |

**Was passiert beim Umschalten auf "Auftrag + Ausgangsrechnung"?**

Die Berechnungsbasis für den Rohertrag ändert sich komplett:

- **Rohertrag steigt drastisch:** € 77.691 → € 323.253
    - Vorher: Abgerechnet (€ 88.441) - Kosten (€ 10.750) = € 77.691
    - Jetzt: Auftrag+Abgerechnet (€ 334.003) - Kosten (€ 10.750) = € 323.253
- **Diff Soll-Ist steigt ebenfalls:** € 50.289 → € 295.851
    - Zeigt das volle Potenzial des Projekts
    - Was noch an Rohertrag möglich wäre, wenn alles abgerechnet wird

**Wann ist diese Ansicht nützlich?**

- Um das gesamte Projektpotenzial zu sehen
- Für die Planung: "Was können wir noch erreichen?"
- Bei der Projektbewertung: "Lohnt sich die Fortsetzung?"

<aside>
🚨

**Wichtig:** Diese Ansicht zeigt **POTENZIAL**, nicht die aktuelle Realität!

</aside>

### Zwei Buttons aktiviert (Auftrag + Ausgangsrechnung + %)

![image.png](Projektauswertung%20Deine%20Schaltzentrale%20f%C3%BCr%20profita/image%203.png)

| Button-Status | Abgerechnet | Kosten | Rohertrag | Soll Honorar | Diff Soll-Ist | Stunden | Auftrag | Auftrag + Abgerechnet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ausgangslage:** Kunde + Abgerechnet + € | € 88.441 | € 10.750 | € 77.691 | € 27.402 | € 50.289 | 201:54 | € 245.562 | € 334.003 |
| **Zum Vergleich:** Kunde + Abgerechnet + % | € 88.441 | € 10.750 | **88%** | € 27.402 | **57%** | 201:54 | € 245.562 | € 334.003 |
| **Jetzt:** Kunde + Auftrag+Ausgang + % | € 88.441 | € 10.750 | **97%** | € 27.402 | **89%** | 201:54 | € 245.562 | € 334.003 |

**Was zeigt der Vergleich?**

- **Rohertrag:** Steigt von 88% auf 97%
- **Diff Soll-Ist:** Steigt von 57% auf 89%

**Warum steigen die Prozente?**

- Bei "Abgerechnet": Basis ist € 88.441 (nur was bereits abgerechnet wurde)
- Bei "Auftrag+Ausgang": Basis ist € 334.003 (das gesamte Projektvolumen)
- Die Kosten bleiben gleich, aber die Erlösbasis vervierfacht sich fast!

<aside>
🚨

**Wichtig:** Diese Ansicht zeigt **POTENZIAL**, nicht die aktuelle Realität!

</aside>

### Stundenkosten-Button auf "Intern" umschalten

![image.png](Projektauswertung%20Deine%20Schaltzentrale%20f%C3%BCr%20profita/image%204.png)

| Button-Status | Abgerechnet | Kosten | Rohertrag | Soll Honorar → Interne Stundenkosten | Diff Soll-Ist | Stunden | Auftrag | Auftrag + Abgerechnet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ausgangslage:** Stundenkosten: Kunde | € 88.441 | € 10.750 | € 77.691 | € 27.402 | € 50.289 | 201:54 | € 245.562 | € 334.003 |
| **Jetzt:** Stundenkosten: Intern | € 88.441 | € 10.750 | € 77.691 | **€ 8.384** | **€ 69.307** | 201:54 | € 245.562 | € 334.003 |

**Was passiert beim Umschalten auf "Intern"?**

- **Spaltenname ändert sich:** "Soll Honorar" wird zu "Interne Stundenkosten"
- **Wert sinkt drastisch:** € 27.402 → € 8.384
    - Das sind die echten internen Kosten für 201:54 Stunden
    - Berechnet mit den internen Stundensätzen der Mitarbeiter, die am Projekt gearbeitet haben
- **Diff Soll-Ist steigt:** € 50.289 → € 69.307
    - Das ist der DB II (Deckungsbeitrag II)
    - Zeigt die echte Profitabilität aus Unternehmenssicht
    - Je höher dieser Wert, desto profitabler das Projekt

**Wichtig:** Der Rohertrag bleibt bei € 77.691 gleich, da sich weder Abgerechnet noch Kosten ändern!

**Wann ist diese Ansicht wichtig?**

- Für internes Controlling
- Bei der Frage: "Verdienen wir wirklich Geld mit diesem Projekt?"
- Zur Ressourcenplanung: Lohnt sich weiterer Mitarbeitereinsatz?

### Interne Sicht mit Prozentanzeige (Intern + %)

![image.png](Projektauswertung%20Deine%20Schaltzentrale%20f%C3%BCr%20profita/image%205.png)

| Button-Status | Abgerechnet | Kosten | Rohertrag | Interne Stundenkosten | Diff Soll-Ist | Stunden | Auftrag | Auftrag + Abgerechnet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ausgangslage:** Kunde + Abgerechnet + € | € 88.441 | € 10.750 | € 77.691 | € 27.402 (Soll Honorar) | € 50.289 | 201:54 | € 245.562 | € 334.003 |
| **Vergleich:** Intern + Abgerechnet + € | € 88.441 | € 10.750 | € 77.691 | € 8.384 | € 69.307 | 201:54 | € 245.562 | € 334.003 |
| **Jetzt:** Intern + Abgerechnet + % | € 88.441 | € 10.750 | **88%** | € 8.384 | **78%** | 201:54 | € 245.562 | € 334.003 |

**Was zeigt diese Kombination?**

- **Rohertrag: 88%** - Hohe prozentuale Profitabilität
- **Diff Soll-Ist: 78%** - Der DB II in Prozent
- **Interne Stundenkosten bleiben bei € 8.384** - ändern sich nicht zu Prozent

**Die Kernaussage dieser Ansicht:**
Von jedem abgerechneten Euro bleiben 88% als Rohertrag. Nach Abzug der internen Stundenkosten bleiben 78% als echter Deckungsbeitrag (DB II). Das ist eine sehr profitable Projektsituation!

**Wann nutzt du diese Ansicht?**

- Für schnelle Profitabilitäts-Checks
- Im Controlling-Meeting: "Projekt X hat 78% DB II"
- Zum Vergleich verschiedener Projekte unabhängig von der Größe

### Interne Sicht mit Gesamtpotenzial (Intern + Auftrag+Ausgangsrechnung + €)

![image.png](Projektauswertung%20Deine%20Schaltzentrale%20f%C3%BCr%20profita/image%206.png)

| Button-Status | Abgerechnet | Kosten | Rohertrag | Interne Stundenkosten | Diff Soll-Ist | Stunden | Auftrag | Auftrag + Abgerechnet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ausgangslage:** Kunde + Abgerechnet + € | € 88.441 | € 10.750 | € 77.691 | € 27.402 (Soll Honorar) | € 50.289 | 201:54 | € 245.562 | € 334.003 |
| **Vergleich:** Intern + Abgerechnet + € | € 88.441 | € 10.750 | € 77.691 | € 8.384 | € 69.307 | 201:54 | € 245.562 | € 334.003 |
| **Jetzt:** Intern + Auftrag+Ausgang + € | € 88.441 | € 10.750 | **€ 323.253** | € 8.384 | **€ 314.870** | 201:54 | € 245.562 | € 334.003 |

**Was zeigt diese komplexe Kombination?**

Diese Ansicht kombiniert:

- **Interne Kostensicht** (€ 8.384 statt € 27.402)
- **Gesamtes Projektpotenzial** (€ 334.003 als Basis)

**Die beeindruckenden Zahlen:**

- **Rohertrag: € 323.253** - Das volle Potenzial des Projekts
- **Diff Soll-Ist: € 314.870** - Der maximal mögliche DB II!

**Was bedeutet das?**
Wenn das gesamte Projektvolumen abgerechnet wird und nur die internen Kosten (€ 8.384) berücksichtigt werden, könnte ein DB II von über € 314.000 erreicht werden!

**Wann ist diese Ansicht wichtig?**

- Bei der strategischen Projektbewertung
- Für Investitionsentscheidungen: "Lohnt es sich, mehr Ressourcen einzusetzen?"
- Als Motivations-Tool: "Schaut, was wir erreichen können!"

<aside>
🚨

**Wichtig:** Dies zeigt **POTENZIAL**, nicht die aktuelle Situation!

</aside>

### Maximale Profitabilitätsansicht (Intern + Auftrag+Ausgangsrechnung + %)

![image.png](Projektauswertung%20Deine%20Schaltzentrale%20f%C3%BCr%20profita/image%207.png)

| Button-Status | Abgerechnet | Kosten | Rohertrag | Interne Stundenkosten | Diff Soll-Ist | Stunden | Auftrag | Auftrag + Abgerechnet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ausgangslage:** Kunde + Abgerechnet + € | € 88.441 | € 10.750 | € 77.691 | € 27.402 (Soll Honorar) | € 50.289 | 201:54 | € 245.562 | € 334.003 |
| **Vergleich:** Intern + Auftrag+Ausgang + € | € 88.441 | € 10.750 | € 323.253 | € 8.384 | € 314.870 | 201:54 | € 245.562 | € 334.003 |
| **Jetzt:** Intern + Auftrag+Ausgang + % | € 88.441 | € 10.750 | **97%** | € 8.384 | **94%** | 201:54 | € 245.562 | € 334.003 |

**Was zeigt diese ultimative Controlling-Ansicht?**

Alle drei Buttons sind aktiviert:

- **Intern:** Zeigt echte Kosten (€ 8.384)
- **Auftrag+Ausgang:** Zeigt Gesamtpotenzial (Basis € 334.003)
- **%:** Macht es vergleichbar

**Die Spitzenwerte:**

- **Rohertrag: 97%** - Fast das gesamte Projektvolumen ist Rohertrag
- **Diff Soll-Ist: 94%** - Maximaler DB II in Prozent!

**Was bedeutet 94% DB II?**
Von jedem Euro Projektvolumen bleiben nach Abzug aller internen Kosten 94 Cent als Deckungsbeitrag! Das ist außergewöhnlich profitabel.

**Wann nutzt du diese Ansicht?**

- Für Executive-Dashboards
- Bei der Projektpriorisierung: "Welche Projekte sind am profitabelsten?"
- Für strategische Entscheidungen: "Wo setzen wir unsere besten Leute ein?"

**Der Clou:** Diese Ansicht zeigt das theoretische Maximum der Profitabilität - das Ziel, das es zu erreichen gilt!

## 🎯 Zusammenfassung: So nutzt du die Projektauswertung richtig

### Für das monatliche Controlling-Meeting

**1. IST-Stand zum Monatsende prüfen (für Abrechnung):**

- **Einstellung:** Stundenkosten: Kunde + Rohertrag: Abgerechnet + €
- **Fokus:** Was wurde bereits abgerechnet? Was kann noch in Rechnung gestellt werden?
- **Wichtige Fragen:**
    - Welche Leistungen sind noch nicht abgerechnet?
    - Stimmt die Diff Soll-Ist noch oder läuft das Projekt aus dem Ruder?
    

**2. Projektsteuerung und Zukunftsplanung:**

- **Einstellung:** Stundenkosten: Intern + Rohertrag: Auftrag+Ausgang + %
- **Fokus:** Wie profitabel sind unsere Projekte wirklich? Wo müssen wir nachsteuern?

### 📊 DB II richtig interpretieren

<aside>
🚨

**Wichtig:** Die Bewertung des DB II hängt stark von deiner Kalkulation der internen Stundensätze ab!

</aside>

1. **Wenn interne Stundensätze "auf Kante" kalkuliert sind** (nur Gehalt + Sozialabgaben):
    - DB II über 70% = sehr profitabel
    - DB II 50-70% = solide
    - DB II 30-50% = Handlungsbedarf
    - DB II 20-30% = kritisch
    - DB II unter 20% = unrentabel
    
2. **Wenn interne Stundensätze mit ordentlichem Puffer kalkuliert sind** (inkl. Gemeinkosten, Gewinnzuschlag):
    - DB II über 50% = sehr profitabel
    - DB II 30-50% = solide
    - DB II 20-30% = akzeptabel
    - DB II 10-20% = prüfen
    - DB II unter 10% = kritisch (außer bei Prestige-Projekten)
    

### 💡 Profi-Tipps für die Praxis

1. **Wöchentliche Routine:**
    1. Montags: Alle Projekte in der Ansicht "Kunde + Abgerechnet + €" durchgehen
    2. Prüfen: Wo ist die Diff Soll-Ist kritisch (< 10%)?
    3. Bei kritischen Projekten sofort handeln!
    
2. **Monatliches Controlling:**
    1. **Anfang des Monats:** IST-Stand dokumentieren
    2. **Mitte des Monats:** Kritische Projekte identifizieren (alle 3 Buttons aktiv)
    3. **Ende des Monats:** Abrechnungsvorbereitung
    
3. **Quartalsplanung:**
    - Nutze die Ansicht mit allen 3 Buttons aktiv
    - Sortiere nach DB II (%)
    - Fokussiere Ressourcen auf die profitabelsten Projekte

### ⚠️ Warnsignale, auf die du achten solltest

1. **Diff Soll-Ist < 0 in Kundensicht** = Sofort Nachtragsangebot erstellen!
2. **DB II unter deinem Zielwert** = Projekt überprüfen oder Preise anpassen
3. **Große Differenz zwischen Kunde/Intern** = Ineffizienzen im Team?

### 

# 📚 Weiterführende Artikel

[Projektauswertung](../%F0%9F%93%8A%20Reporting%20und%20Auswertungen/Projektauswertung%207e367ce538e940a2ada2bd8dc418bccf.md)

[Projektauswertung - Export](../%F0%9F%93%A4%20Exportm%C3%B6glichkeiten/Projektauswertung%20-%20Export%202c3015f5d93f81bfbf9dee8e38ba31f4.md)