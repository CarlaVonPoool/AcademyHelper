# Status ≠ Status: Benutzerdefiniert vs. Offen/Geschlossen

Article Description: Fortschrittanzeige über benutzerdefinierte Ticketstatus.
Last Updated: 26. August 2025
Published: Yes
Suggested: No

## 📊 Artikel-Metadaten

- **Artikel-Typ:** Anleitung
- **Erfahrungslevel:** 🚀 Einsteiger

---

# Benutzerdefinierter Status: Ticket-Zustände definieren und anpassen

## Was sind benutzerdefinierte Ticket-Status?

Ticket-Status sind die verschiedenen Bearbeitungszustände, die ein Ticket während seines Lebenszyklus durchläuft. Sie helfen Ihnen und Ihrem Team:

- **Den Überblick zu behalten:** Auf einen Blick erkennen, in welcher Phase sich ein Ticket befindet
- **Arbeitsabläufe zu strukturieren:** Klare Prozesse vom Eingang bis zur Lösung definieren
- **Prioritäten zu setzen:** Dringende Status visuell hervorheben
- **Fortschritte zu messen:** Sehen, wie viele Tickets in welchem Stadium sind

### Typischer Status-Workflow

Ein Ticket durchläuft normalerweise mehrere Phasen:

1. **Eingang** (Neu/Offen) → 2. **Bearbeitung** (In Arbeit) → 3. **Überprüfung** (Review) → 4. **Abschluss** (Gelöst/Geschlossen)

Die Status-Verwaltung ermöglicht es Ihnen, diesen Workflow genau an Ihre Bedürfnisse anzupassen - mit eigenen Bezeichnungen, Farben und Zwischenschritten.

## Schritt-für-Schritt

![image.png](Status%20%E2%89%A0%20Status%20Benutzerdefiniert%20vs%20Offen%20Geschlo/image.png)

![image.png](Status%20%E2%89%A0%20Status%20Benutzerdefiniert%20vs%20Offen%20Geschlo/image%201.png)

![image.png](Status%20%E2%89%A0%20Status%20Benutzerdefiniert%20vs%20Offen%20Geschlo/image%202.png)

# ⚠️ Der Unterschied: Zwei Status-Arten

## 🎨 Benutzerdefinierte Status (Was Sie konfiguriert haben)

- **Zweck:** Visualisierung des Arbeitsfortschritts
- **Anpassbar:** Name, Farben, Anzahl - alles frei wählbar
- **Beispiele:** "Neu", "In Bearbeitung", "Review", "Abgeschlossen"
- **Auswirkung auf Zeiterfassung:** KEINE

## 🔒 System-Status (Offen/Geschlossen)

- **Zweck:** Steuert die Zeiterfassung
- **Optionen:** Nur zwei - Offen oder Geschlossen
- **Erkennbar an:**
    - Graues Häkchen (✓) = Offen → Zeitbuchung möglich
    - Grünes Häkchen (✓) = Geschlossen → Zeitbuchung GESPERRT
- **Nicht anpassbar:** Fest im System verankert

![image.png](Status%20%E2%89%A0%20Status%20Benutzerdefiniert%20vs%20Offen%20Geschlo/image%203.png)

## 💡 Was bedeutet das in der Praxis?

**Beispiel 1:** Ein Ticket hat den Status "100% Fertig" (benutzerdefiniert), ist aber noch "Offen" (System-Status)
→ **Ergebnis:** Zeitbuchung ist weiterhin möglich!

**Beispiel 2:** Ein Ticket hat den Status "50% In Bearbeitung", wurde aber bereits "Geschlossen"
→ **Ergebnis:** Keine Zeitbuchung mehr möglich, obwohl der Status noch nicht bei 100% ist!

## 🚨 Goldene Regel

**Schließen Sie ein Ticket erst (grünes Häkchen), wenn:**

- ✅ Alle Arbeiten abgeschlossen sind
- ✅ Alle Zeiten erfasst wurden
- ✅ Keine weiteren Buchungen erwartet werden

<aside>
💡

**Tipp:** Der benutzerdefinierte Status dient nur der Organisation - die tatsächliche Kontrolle über die Zeiterfassung hat allein der System-Status!

</aside>

---

# 🔍 Filtern nach beiden Status-Arten

### So nutzen Sie die Filter richtig:

Die Ticketliste bietet **zwei separate Filteroptionen**, die den beiden Status-Arten entsprechen:

### 1. Filter "Ticketstatus" (Benutzerdefinierte Status)

**Was er zeigt:** Ihre selbst angelegten Workflow-Status

- 0% Neu/Offen
- 10% In Planung
- 25% Begonnen
- 50% In Bearbeitung
- 75% Fortgeschritten
- 90% Review/QA
- 100% Abgeschlossen

**Verwendung:** Filtern Sie nach Arbeitsfortschritt und Projektphase

### 2. Filter "Ticket" (System-Status)

**Was er zeigt:** Den technischen Buchungsstatus

- **√ Offen** = Zeiterfassung möglich
- **√ Abgeschlossen** = Zeiterfassung gesperrt

**Verwendung:** Finden Sie schnell alle Tickets, auf die noch gebucht werden kann (oder nicht mehr)

![image.png](Status%20%E2%89%A0%20Status%20Benutzerdefiniert%20vs%20Offen%20Geschlo/image%204.png)

### 💡 Praxis-Tipp: Kombinierte Filter

Sie können beide Filter gleichzeitig verwenden. Diese Filterkombination hilft Ihnen, Unstimmigkeiten zwischen Workflow-Status und Buchungsstatus schnell zu erkennen.

**Beispiele:**

- Alle Tickets "100% Abgeschlossen" die noch "Offen" sind → Nacharbeiten möglich
- Alle Tickets "In Bearbeitung" die bereits "Abgeschlossen" sind → Achtung: keine Zeitbuchung mehr möglich!