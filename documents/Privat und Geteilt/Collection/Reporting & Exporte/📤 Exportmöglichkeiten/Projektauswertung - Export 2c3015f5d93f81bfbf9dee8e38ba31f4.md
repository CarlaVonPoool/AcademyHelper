# Projektauswertung - Export

Article Description: Export zur wirtschaftlichen Betrachtung und Soll-Ist Vergleich. 
Last Updated: 14. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## 🎯 Was ist der Projektauswertung-Export?

Der Projektauswertung-Export liefert eine **wirtschaftliche Betrachtung auf Projektebene aus Controlling-Sicht**. Er zeigt Rohertrag und Deckungsbeitrag (DB II) für jedes Projekt – ideal für Geschäftsführung, Controlling und alle, die einen schnellen Überblick über die Rentabilität ihrer Projekte benötigen.

## 🔄 Zwei Perspektiven – zwei Exports

Je nachdem, aus welcher Perspektive Sie Ihre Projekte betrachten möchten, empfehlen wir unterschiedliche Exports:

### 📊 Controlling-Perspektive → Projektauswertung-Export

**Zielgruppe:** Geschäftsführung, Controlling, Projektverantwortliche

**Fragestellungen:**

- Wie profitabel sind meine Projekte insgesamt?
- Wie entwickeln sich Rohertrag und DB II?
- Welche Kunden oder Teams performen am besten?
- Wie sieht der Soll-Ist-Vergleich auf Projektebene aus?

### 🗂️ Projektmanager-Perspektive → Projektmanagement-Export

**Zielgruppe:** Projektmanager:innen, Projektleitung

**Fragestellungen:**

- Was wurde genau beauftragt?
- Welche Aufgaben/Tickets haben welches Budget?
- Wie verteilen sich Eigen- und Fremdleistungen?
- Wie sieht die Aufteilung auf Phasen und Aufträge aus?

💡 **Tipp:** Für die operative Projektsteuerung mit Detailinformationen nutzen Sie den [**Projektmanagement-Export**](Projektmanagement%20-%20Export%202b8015f5d93f809ebf7dfb3db4554377.md).

## 📍 Wo finden Sie den Export?

Den Projektauswertung-Export erreichen Sie in 5 Schritten:

1. Klicken Sie in der Hauptnavigation auf **Reporting**
2. Wählen Sie im Untermenü **Projektauswertung**
3. Klappen Sie den Filterbereich auf (Pfeil rechts)
4. Wählen Sie Ihre gewünschten **Aggregationsoptionen** (siehe nächster Abschnitt)
5. Klicken Sie auf den Button **→ Export (xlsx)**

![image.png](Projektauswertung%20-%20Export/image.png)

<aside>
💡

**Tipp:** Über den Auswertungszeitraum können Sie den Export auf bestimmte Zeiträume eingrenzen – z.B. „aktuelles Jahr“ oder einen individuellen Datumsbereich.

</aside>

## ⚙️ Aggregationsoptionen: Was soll im Export enthalten sein?

Vor dem Export wählen Sie über das Zahnrad-Symbol, welche Aggregationsebenen Ihre Datei enthalten soll:

| Option | Beschreibung | Wann sinnvoll? |
| --- | --- | --- |
| Summe pro Kunde | Zwischensummen auf Kundenebene | Für schnelle Kundenübersichten |
| Summe pro Projekttyp | Zwischensummen nach Projektart | Für Auswertungen nach Projekttypen |
| Einzelsummen Projekt | Jedes Projekt als einzelne Zeile | Für Pivot-Tabellen und PowerQuery |

### Welche Option für welchen Anwendungsfall?

<aside>
⚠️

**Wichtig:** Ihre Wahl hängt davon ab, wie Sie die Daten weiterverarbeiten möchten!

</aside>

**Für Pivot-Tabellen oder PowerQuery:**

Wählen Sie nur **Einzelsummen Projekt** und deaktivieren Sie die Zwischensummen. Pivot-Tabellen und PowerQuery erstellen Aggregationen selbstständig – vorhandene Zwischensummen würden Ihre Berechnungen verfälschen.

**Für einfache Excel-Listen:**

Aktivieren Sie **Summe pro Kunde** und/oder **Summe pro Projekttyp**. So haben Sie direkt lesbare Übersichten mit Zwischensummen, ohne selbst rechnen zu müssen.

<aside>
💡

**Gut zu wissen:** In Excel gibt es einen wichtigen Unterschied zwischen **Listen** und **Tabellen**:

- **Listen** sind einfache Datenreihen – Zwischensummen sind hier praktisch
    - **Tabellen** (über „Als Tabelle formatieren“) sind strukturierte Datenbereiche für Pivot-Analysen – hier stören Zwischensummen
</aside>

## ✨ Vorteile

- **Wirtschaftlichkeit auf einen Blick:** Rohertrag, Kosten und Differenzen pro Projekt sofort erkennbar
- **Zwei Perspektiven:** Kundenansicht (extern) und Interne Ansicht (mit internen Stundenkosten)
- **Flexibel filterbar:** Nach Team, Projektstatus, Zeitraum oder Kunde eingrenzbar
- **Excel-kompatibel:** Perfekt für eigene Pivot-Tabellen, PowerQuery und Dashboards

## 📋 Enthaltene Felder im Überblick

Der Export umfasst **31 Felder**, aufgeteilt in 5 thematische Bereiche:

### Bereich 1: Kundenstammdaten (5 Felder)

| Feldname | Beschreibung | Beispiel |
| --- | --- | --- |
| Kundennummer | Eindeutige Kunden-ID | KD-100003 |
| Kundenkürzel | Abkürzung des Kundennamens | MÜLLER |
| Kunde | Vollständiger Kundenname | Müller Brot AG |
| Kundenverantwortlich | Zuständige Person für den Kunden | Harry Hirsch |
| Kundentag | Kategorie/Tag des Kunden | Freelancer:innen |

### Bereich 2: Projektstammdaten (6 Felder)

| Feldname | Beschreibung | Beispiel |
| --- | --- | --- |
| Projekttyp | Art des Projekts | Werbung, Strategie, People & Culture |
| Projektverantwortlich | Zuständige Person für das Projekt | Harry Hirsch |
| Projektnummer | Eindeutige Projekt-ID | Müller-250153 |
| Team | Zugeordnetes Team | Design, Motion |
| Projektstatus | Aktueller Status | In Arbeit, Abgeschlossen, On Hold, Angeboten |
| Projekttitel | Name des Projekts | Müller Online Marketing |

### Bereich 3: Abrechnungs- und Auftragswerte (5 Felder)

| Feldname | Beschreibung | Hinweis |
| --- | --- | --- |
| Abgerechnet | Bereits abgerechneter Betrag (€) | Was dem Kunden in Rechnung gestellt wurde |
| Angebot | Summe aller Angebote (€) | Potenzielle Auftragswerte |
| Auftrag | Beauftragter Wert (€) | Vom Kunden bestätigte Aufträge |
| Auftrag + Abgerechnet | Kombinierter Wert (€) | Auftrag plus bereits abgerechnete Beträge |
| Stunden | Geleistete Arbeitsstunden | Gesamtstunden auf dem Projekt |

### Bereich 4: Kosten und Rohertrag (7 Felder)

| Feldname | Beschreibung |
| --- | --- |
| Kosten | Entstandene Projektkosten (€) – Fremdleistungen, Materialkosten etc. |
| Soll Honorar | Honorar basierend auf gebuchten Stunden * Kundenkosten(€) |
| Interne Stundenkosten | Kalkulierte interne Kosten (€) für Rentabilitätsberechnung |
| Rohertrag Abgerechnet | Abgerechnet minus Kosten (€) |
| Rohertrag Abgerechnet % | Rohertrag als Prozentsatz |
| Rohertrag Auftrag + abgerechnet | Erweiterter Rohertrag inkl. offener Aufträge (€) |
| Rohertrag Auftrag + abgerechnet % | Erweiterter Rohertrag inkl. offener Aufträge in % |

### Bereich 5: Soll-Ist-Vergleich (8 Felder)

**Kundenperspektive (extern):**

| Feldname | Beschreibung |
| --- | --- |
| Diff Soll-Ist Abgerechnet Kunde | Abweichung Soll-Honorar zu abgerechneten Kosten (€) |
| Diff Soll-Ist Abgerechnet Kunde % | Abweichung in Prozent |
| Diff Soll-Ist / Auftrag + abgerechnet Kunde | Abweichung inkl. Aufträge (€) |
| Diff Soll-Ist / Auftrag + abgerechnet Kunde % | Abweichung inkl. Aufträge in % |

**Interne Perspektive:**

| Feldname | Beschreibung |
| --- | --- |
| Diff Soll-Ist Abgerechnet Intern | Abweichung mit internen Stundenkosten (€) |
| Diff Soll-Ist Abgerechnet Intern % | Abweichung intern in Prozent |
| Diff Soll-Ist / Auftrag + abgerechnet Intern | Abweichung intern inkl. Aufträge (€) |
| Diff Soll-Ist / Auftrag + abgerechnet Intern % | Abweichung intern inkl. Aufträge in % |

## 🔍 Wichtige Kennzahlen verstehen

### Rohertrag vs. DB II – Was zeigt dieser Export?

Dieser Export zeigt sowohl den **Rohertrag** als auch die Basis für den **Deckungsbeitrag II (DB II)**:

| Kennzahl | Berechnung | Feld im Export |
| --- | --- | --- |
| **Rohertrag** | Umsatz − Fremdkosten | Rohertrag Abgerechnet |
| **DB II** | Rohertrag − Stundenkosten | Berechnung über Diff Soll-Ist Intern |

**Die Berechnungen im Detail:**

```
Rohertrag = Umsatz − Fremdkosten (Kosten)

DB II = Rohertrag − Interne Stundenkosten
```

<aside>
💡

**Gut zu wissen:** Dieser Export unterscheidet nicht zwischen Umsatz aus Eigenleistung und Umsatz aus Fremdleistung. 

</aside>

### Kundenperspektive vs. interne Perspektive

- **Kundenperspektive:** Basiert auf den dem Kunden kommunizierten Stundensätzen
- **Interne Perspektive:** Berücksichtigt die tatsächlichen internen Stundenkosten → zeigt den echten DB II

<aside>
💡

**Tipp:** Die interne Perspektive zeigt Ihnen die „echte“ Rentabilität – nutzen Sie diese für strategische Entscheidungen.

</aside>

### Soll-Ist-Differenz

Die Soll-Ist-Differenz vergleicht das geplante Honorar (basierend auf gebuchten Stunden × Stundensatz) mit den tatsächlich abgerechneten bzw. beauftragten Werten. So erkennen Sie, ob Projekte im geplanten Rahmen liegen.

## 💼 Anwendungsfälle

### Szenario 1: Monatliches Projekt-Controlling (DB II-Analyse)

Sie möchten wissen, welche Projekte nach Abzug aller Fremdkosten und Stundenkosten profitabel sind.

**So gehen Sie vor:**

1. Exportieren Sie alle Projekte mit Status „In Arbeit“
2. Prüfen Sie die Spalte „Diff Soll-Ist Abgerechnet Intern“ für den DB II
3. Projekte mit negativen Werten oder unterhalb des vereinbarten Schwellenwertes, benötigen Aufmerksamkeit

<aside>
⚠️

**Beachten Sie:** Ein niedriger DB II kann bedeuten:

- Hohe Fremdkosten im Projekt (z.B. viel externe Produktion)
- Zu niedrige Stundensätze kalkuliert
- Mehr Stunden geleistet als abgerechnet
</aside>

### Szenario 2: Team-Performance analysieren (mit Pivot)

Sie möchten die wirtschaftliche Leistung Ihrer Teams vergleichen.

**So gehen Sie vor:**

1. Exportieren Sie mit Option **Einzelsummen Projekt** (ohne Zwischensummen!)
2. Formatieren Sie die Daten in Excel als Tabelle
3. Erstellen Sie eine Pivot-Tabelle
4. Gruppieren Sie nach „Team“, summieren Sie „Abgerechnet“ und „Rohertrag Abgerechnet“
5. Berechnen Sie den durchschnittlichen Rohertrag und DB II pro Team

### Szenario 3: Schnelle Kundenübersicht (ohne Pivot)

Sie brauchen eine einfache Liste mit Kundensummen zum Ausdrucken.

**So gehen Sie vor:**

1. Exportieren Sie mit Option **Summe pro Kunde** aktiviert
2. Öffnen Sie die Datei – Zwischensummen sind bereits enthalten
3. Drucken oder teilen Sie die Datei direkt

### Szenario 4: Projekte mit hohen Fremdkosten identifizieren

Sie möchten herausfinden, bei welchen Projekten der Rohertrag durch hohe Fremdkosten gedrückt wird.

**So gehen Sie vor:**

1. Exportieren Sie alle Projekte
2. Berechnen Sie in Excel: Fremdkostenanteil = Kosten ÷ Abgerechnet
3. Sortieren Sie nach diesem Anteil

💡 **Tipp:** Bei Projekten mit hohem Fremdkostenanteil ist ein niedrigerer Rohertrag normal. Vergleichen Sie daher nur Projekte mit ähnlicher Struktur miteinander.

## ❌ Häufige Fehler vermeiden

❌ **Fehler:** Zwischensummen im Export für Pivot-Tabellen verwenden

✅ **Besser:** Nur „Einzelsummen Projekt“ wählen – Pivot erstellt Summen selbst

❌ **Fehler:** Leere Werte als „Null“ interpretieren

✅ **Besser:** Leere Felder bedeuten, dass keine Daten vorliegen – z.B. noch keine Abrechnung erfolgt ist

❌ **Fehler:** Nur auf Rohertrag % schauen

✅ **Besser:** Kombinieren Sie Prozentsatz mit absolutem Wert – 80% Marge bei 100€ ist weniger relevant als 20% bei 100.000€

❌ **Fehler:** Rohertrag zwischen unterschiedlichen Projekttypen vergleichen

✅ **Besser:** Ein Projekt mit hohem Fremdkostenanteil hat naturgemäß einen niedrigeren Rohertrag – vergleichen Sie nur ähnliche Projektstrukturen

❌ **Fehler:** Rohertrag mit DB II verwechseln

✅ **Besser:** Rohertrag = Umsatz − Fremdkosten | DB II = Rohertrag − Stundenkosten