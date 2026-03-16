# Abbildung der Gehälter

Article Description: Methodenübersicht für ein sauberes Gehaltsbild.
Last Updated: 19. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: No

## 📊 Artikel-Metadaten

- **Artikel-Typ:** Erklärung
- **Schwierigkeitsgrad:** 🚀🚀 Fortgeschritten

## 💰 Abbildung der Gehälter

Sie möchten Personalkosten in Ihrer Quartalsauswertung und Liquiditätsplanung sehen, um:

- einen **Forecast** zu erstellen, wann welche Kosten anfallen
- zu wissen, wie hoch Ihre **Umsätze** sein müssen, um kostendeckend zu arbeiten
- Ihr **Kostenmanagement** langfristig zu planen

Ohne Gehaltskosten fehlt Ihnen ein wesentlicher Kostenblock – besonders bei Dienstleistungsunternehmen, wo Personalkosten oft 60-80% der Gesamtkosten ausmachen.

## 💡 Die Lösung

Poool bietet zwei Wege, um Gehälter abzubilden:

| Methode | Best für | Flexibilität | Datenschutz |
| --- | --- | --- | --- |
| **Stammdaten Mitarbeiter** | Automatisierte Kostenverteilung | Gering | Hoch |
| **Wiederkehrende Eingangsrechnung** | Budget-Kontrolle & Detailauswertungen | Hoch | Mittel |

Beide Methoden können auch parallel genutzt werden – Sie sehen sie dann separat in der Quartalsauswertung.

## 🛠️ Variante 1: Stammdaten Mitarbeiter

### So funktioniert's

Die Gehaltskosten werden direkt beim Mitarbeiter hinterlegt. In der Quartalsauswertung werden diese als Gesamtsumme ausgegeben – entweder Bruttogehälter oder Brutto inklusive AG-Anteil.

**Pfad:** `Benutzer` → `Mitarbeiter` → gewünschten Mitarbeiter auswählen → `Stammdaten` → `Historie Gehalt`

### 📌 Einrichtung

- Navigieren Sie zu `Benutzer` → `Mitarbeiter`
- Wählen Sie den gewünschten Mitarbeiter aus
- Gehen Sie zum Reiter `Stammdaten`
- Scrollen Sie zum Bereich `Historie Gehalt`
- Klicken Sie auf `+ Hinzufügen`
- Tragen Sie ein:
    - **Gültig ab:** Datum, ab dem das Gehalt gilt
    - **Monatsgehalt:** Bruttogehalt inkl. AG-Nebenkosten
    

<aside>
🔑

 **Wichtig:** Tragen Sie das **Bruttogehalt inklusive Arbeitgeber-Nebenkosten** ein (ca. 20-25% Aufschlag auf das Bruttogehalt). Nur so bilden sich die tatsächlichen Personalkosten korrekt ab. Bei Gehaltsänderungen einen neuen Eintrag mit neuem "Gültig ab"-Datum anlegen. So bleibt die Historie erhalten und Auswertungen für vergangene Zeiträume bleiben korrekt.

</aside>

### ⭐️ Vorteile

- **Berechtigungssteuerung:** Nur autorisierte Personen sehen Gehaltsdaten
- **Automatische Anpassung:** Bei Personalwechsel oder Zeitumbuchungen aktualisieren sich die Kosten
- **Datenschutz:** Gehälter sind nicht in Rechnungslisten sichtbar
- **Wenig Pflegeaufwand:** Einmal einrichten, automatisch verteilen
- **Sonderzahlungen möglich:** Boni und Prämien können erfasst werden

### ❌ Nachteile

- **Nur Gesamtsumme:** Die Gehälter werden nur als Gesamtsumme angezeigt – es ist nicht erkennbar, wie sich die Summe zusammensetzt
- **Keine Budget-Kontrolle:** Gehälter können keinem Budget zugeordnet werden
- **Steueranpassungen separat:** Nachträgliche Steueranpassungen müssen zusätzlich als Rechnung eingebucht werden

## 🛠️ Variante 2: Wiederkehrende Eingangsrechnung

### So funktioniert's

Sie erfassen die monatlichen Gehaltskosten als wiederkehrende Eingangsrechnung. Das gibt Ihnen volle Kontrolle über Zuordnung, Budgets und Auswertungen.

**Pfad:** `Buchhaltung` → `Eingangsrechnungen` → `Wiederkehrend`

### 📌 Einrichtung

- Navigieren Sie zu `Buchhaltung` → `Eingangsrechnungen`
- Wechseln Sie zum Reiter `Wiederkehrend`
- Klicken Sie auf `+ Neue wiederkehrende Rechnung`
- Füllen Sie die Kopfdaten aus:
    - **Lieferant:** z.B. "Lohnbuchhaltung intern" oder Ihr Lohnbüro
    - **Intervall:** Monatlich
    - **Startdatum:** Erster des aktuellen Monats

### 💼 Empfohlener Aufbau mit 3 Positionen

Für maximale Flexibilität und korrekte Liquiditätsplanung empfehlen wir drei separate Positionen:

| Position | Beschreibung | Beispielbetrag | Fälligkeit |
| --- | --- | --- | --- |
| 1 | Gehälter netto (Auszahlung an MA) | 40.672,00 € | Monatsende |
| 2 | SV-Beiträge (AG + AN-Anteil) | 15.400,00 € | 15. Folgemonat |
| 3 | Lohnsteuer + Nebenabgaben | 10.390,00 € | 15. Folgemonat |
| **Gesamt** |  | **66.462,00 €** |  |

<aside>
🔑

Die Aufteilung in drei Rechnungen ermöglicht eine genauere Liquiditätsplanung, da Gehälter am Monatsende, aber Steuern und Abgaben erst im Folgemonat fällig werden.

</aside>

### Zuordnungsmöglichkeiten pro Position

Jede Position kann individuell zugeordnet werden:

- **Projekt:** Direktzuordnung zu einem Projekt
- **Budget:** Kontrolle gegen ein definiertes Budget
- **Kostenstelle:** Für die Kostenstellenrechnung
- **Aufteilung:** Prozentuale Verteilung auf mehrere Projekte/Kostenstellen

### ⭐️ Vorteile

- **Budget-Kontrolle:** Gehälter gegen Budget prüfen (z.B. 500.000 € Jahresbudget)
- **Hohe Flexibilität:** Sonderzahlungen, Boni, 13./14. Gehalt einfach ergänzen
- **Detaillierte Auswertungen:** Nach Kostenstelle, Projekt, Zeitraum filterbar
- **Liquiditätsplanung:** Unterschiedliche Fälligkeiten abbildbar

### ❌ Nachteile

- **Manuelle Pflege:** Bei Personalwechsel muss die Rechnung angepasst werden
- **Kein automatischer Datenschutz:** Gehälter in Rechnungslisten sichtbar (Berechtigungen prüfen!)
- **Mehr Aufwand:** Monatliche Kontrolle und ggf. Anpassung nötig

## 📊 Auswertung in der Praxis

### Quartalsauswertung

In der Quartalsauswertung sehen Sie beide Methoden – falls Sie sie parallel nutzen.

Die Auswertung zeigt:

- **Gehälter (aus Stammdaten):** Automatisch verteilte Kosten basierend auf Zeitbuchungen
- **Gehälter (aus Eingangsrechnungen):** Manuell zugeordnete Kosten

### Budget-Kontrolle

Bei Variante 2 können Sie die Gehälter gegen ein Budget prüfen.

⚠️ **Wichtig:** Eine Budget-Überschreitung zeigt, warum regelmäßige Kontrolle wichtig ist!

## 🔄 Vergleich der Methoden

| Kriterium | Stammdaten | Wiederkehrende ER |
| --- | --- | --- |
| **Einrichtungsaufwand** | Gering | Mittel |
| **Laufender Pflegeaufwand** | Sehr gering | Mittel |
| **Budget-Kontrolle** | ❌ Nein | ✅ Ja |
| **Sonderzahlungen abbildbar** | ✅ Ja | ✅ Einfach |
| **Liquiditätsplanung** | ✅ Ja (nicht detailliert) | ✅ Ja |
| **Datenschutz** | ✅ Hoch | ⚠️ Berechtigungen prüfen |
| **Detailauswertungen** | ⚠️ Begrenzt | ✅ Umfangreich |

## 🔍 Entscheidungshilfe

### Wählen Sie Stammdaten, wenn:

- Sie eine **automatische Verteilung** bei Ein- und Austritten wollen
- **Datenschutz** wichtig ist und nur wenige Personen Gehälter sehen sollen
- Sie **wenig Pflegeaufwand** haben möchten
- Ihre Gehaltsstruktur **stabil** ist (wenig Sonderzahlungen)

### Wählen Sie wiederkehrende Eingangsrechnung, wenn:

- Sie **Budget-Kontrolle** für Personalkosten brauchen
- Sie eine **detaillierte Liquiditätsplanung** benötigen
- Sie Gehälter **bestimmten Kostenstellen** zuordnen möchten

### Kombinieren Sie beide Methoden, wenn:

- Sie sowohl **automatische Projektverteilung** als auch **Budget-Kontrolle** möchten
- Sie verschiedene **Kostenarten separat** auswerten wollen

## 💡 Praxistipps

### Lohnjournal nutzen

Fordern Sie monatlich das **Lohnjournal** von Ihrem Lohnbüro an. Es enthält alle relevanten Summen für die wiederkehrende Eingangsrechnung: Nettogehälter, SV-Beiträge (AG + AN) und Lohnsteuer mit Nebenabgaben.

### Liquiditätsplanung optimieren

Bei Variante 2 können Sie **separate wiederkehrende Rechnungen** pro Zahlungsart anlegen: Gehälter fällig am Monatsende, Steuern/Abgaben fällig am 15. des Folgemonats. So wird Ihre Liquiditätsplanung präziser.

### Buchhaltungsexport beachten

⚠️ **Wichtig für den Buchhaltungsexport:**

Wenn Sie Gehälter über Eingangsrechnungen erfassen, erscheinen sie auch im Buchhaltungsexport. Um **Doppelungen in der Finanzbuchhaltung** zu vermeiden:

- Legen Sie einen **separaten Nummernkreis** für Gehaltsrechnungen an (z.B. "GH-2025-001")
- Schließen Sie diesen Nummernkreis im **Buchhaltungsexport aus**
- So bleiben die Daten in Poool für Auswertungen, werden aber nicht doppelt an die Buchhaltung übergeben

**Pfad:** `Einstellungen` → `Nummernkreise` → Neuen Nummernkreis anlegen

🚨 **Kein Gehaltsexport aus Poool möglich**

Aus Poool ist **kein Export von Gehaltsdaten** möglich – weder zu Reporting-Zwecken noch an die Lohnbuchhaltung. Die Gehaltsdaten in Poool dienen ausschließlich der **internen Kostenplanung und Auswertung**.

Die eigentliche Lohnabrechnung, Gehaltsauszahlung und Meldung an Behörden erfolgt weiterhin über Ihr Lohnbüro oder Ihre Lohnsoftware.

## ❓ Häufige Fragen

**Q: Kann ich beide Methoden gleichzeitig nutzen?**

A: Ja, das ist möglich und sinnvoll, wenn Sie verschiedene Auswertungsziele haben. In der Quartalsauswertung werden beide Kostenarten separat ausgewiesen.

**Q: Was passiert bei Gehaltserhöhungen?**

A: Bei Stammdaten legen Sie einen neuen Eintrag mit "Gültig ab"-Datum an. Bei wiederkehrenden Eingangsrechnungen passen Sie die Beträge in den Positionen an.

**Q: Sind die Gehaltsdaten für alle Benutzer sichtbar?**

A: Bei Stammdaten nur für Benutzer mit entsprechender Berechtigung. Bei Eingangsrechnungen müssen Sie die Berechtigungen für den Rechnungsbereich prüfen.