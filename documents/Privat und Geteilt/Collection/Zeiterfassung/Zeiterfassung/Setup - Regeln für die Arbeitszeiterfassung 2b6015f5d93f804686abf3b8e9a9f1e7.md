# Setup - Regeln für die Arbeitszeiterfassung

Article Description: Verlässliche Rahmenbedingungen für die tägliche Zeiterfassung festlegen.
Published: Yes
Suggested: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

Über die zentralen Einstellungen in Poool definieren Sie, **wie Arbeitszeit erfasst, geprüft und gesichert wird**. Mit automatischen Pausen, Sperrfristen und Validierungsregeln stellen Sie sicher, dass gesetzliche Vorgaben eingehalten und Zeitdaten zuverlässig verarbeitet werden.

## 📌 Prozessbeschreibung: Schritt-für-Schritt Anleitung

### **Schritt 1:** Allgemeine Einstellungen

- Navigieren Sie zu `Einstellungen - Zeiterfassung`. Im ersten Reiter können Sie grundlegende Einstellungen für Ihre Zeiterfassung treffen. Diese bestimmen, **wie Mitarbeitende ihre Zeiten erfassen**, welche Informationen angezeigt werden und welche Systemlogiken greifen. Verwenden Sie diese Einstellungen, um:
- das Verhalten der Arbeitszeiterfassung zu steuern
- Standardwerte und Ansichten festzulegen
- Vorgaben für Pausen, Kommentare, Gleitzeit und Reporting zu aktivieren
- die Projektzeiterfassung an Ihre Organisation anzupassen

**Wichtige Regeln im Überblick**

- Festschreiben von Tagen
Mit dieser Einstellung werden die Tage in einem von Ihnen definierten Abstand automatisch festgeschrieben und können nur noch von Administratoren verändert werden.
- Automatische Pause eintragen
Sie definieren ab Wann eine automatische Pause abgezogen werden soll und wie hoch der Abzug ist.

![image.png](Setup%20-%20Regeln%20f%C3%BCr%20die%20Arbeitszeiterfassung/image.png)

### Schritt 2: Gleitzeit ja oder nein

In Poool können Sie unter `Unternehmen → Mitarbeiter` bei jedem Mitarbeitenden das Gleitzeitkonto aktivieren. → [Gleitzeit in Poool](https://academy.poool.cc/zeiterfassung/r3wCMHR3jbgMdC1LkfbPck/gleitzeit--und-stundenkappung/5DVySoeVhPDagrpwNHeedP) 

### **Schritt 3:** Arbeitszeitvalidierungen einrichten

Navigieren Sie zu **`Einstellungen → Zeiterfassung → Arbeitszeitvalidierungsregeln**.`
Aktivieren Sie die gewünschten Regeln. Sie können beliebig viele kombinieren.
Legen Sie eine **Gültigkeitsperiode** fest. Wählen Sie ein Startdatum und definieren Sie bei Bedarf ein Enddatum, wenn Sie neue Regeln einführen möchten.

- **Übersicht aller Regeln**
    
    
    | **Regel** | **Beschreibung** |
    | --- | --- |
    | **Keine offenen Zeiterfassungsslots** | Falls eine Erfassung der Arbeitsstunden **noch offen** ist oder beispielsweise ein Mitarbeiter:in vergessen hat, sich auszustempeln, gilt die Arbeitszeiterfassung als unbeendet. Es erscheint dann eine Fehlermeldung und ein Ende der Arbeitszeit muss nachgetragen werden.  |
    | **Keine Nachtarbeit** | In dieser Einstellung kann die Zeiterfassung für “Nachtarbeit” gesperrt werden. Die **Eingrenzung für potentielle Arbeitszeiten** können Sie selbst bei Aktivierung der Regel festlegen.  |
    | **Höchstarbeitszeit nicht überschritten** | In dieser Regel können Sie eine Höchstarbeitszeit einstellen, von beispielsweise 10h. Wenn eine erfasste Arbeitszeit **mehr** als die 10h ausmacht, wird eine Fehlermeldung angezeigt.  |
    | **Pausenzeiten eingehalten** | Mit dieser Regel wird festgelegt, nach wie viel geleisteten Stunden eine Pausenzeit **notwendig** ist. In Deutschland beispielsweise ist es **verpflichtend**, nach sechs Stunden eine Pausenzeit von mindestens 30 Minuten zu machen.  |
    | **Keine parallelen Zeiterfassungsslots** | Mit dieser Einstellung legt man fest, dass keine parallelen Zeiten erfasst werden. Es kann somit **nicht gleichzeitig** eine Arbeitszeit im Büro und im Home Office erscheinen. Damit wird verhindert, dass man ausversehen doppelt Stunden bucht oder sich Arbeitsslots überschneiden.  |
    | **Projektzeit darf nicht größer als Arbeitszeit sein** | Mit dieser Regel wird verhindert, dass die Arbeits- und Projektzeit **nicht übereinstimmt**, sondern die Projektzeit der Arbeitszeit gleicht.  |
    | **Projektzeit darf nicht weniger als Arbeitszeit sein** | Mit dieser Regel wird sichergestellt, dass die geleistete Arbeitsstunden immer Projekten zugeordnet sind. So wird vermieden, dass Arbeitsstunden **nicht korrekt abgerechnet** werden.  |

💡 *Hinweis:* Wenn sich Regeln oder Arbeitszeitmodelle ändern, setzen Sie einfach ein Enddatum und starten eine neue Gültigkeitsperiode.

## 💼 Anwendungsfälle

- Validierung von Arbeitszeiten auf gesetzlichen Grundlagen (z. B. Höchstarbeitszeit, Pausenregelungen)
- Vermeidung von Nachtarbeits- oder Überschreitungsfehlern

## ✅ Vorteile im Überblick

- **Automatische Kontrolle:** Reduziert manuelle Prüfungen durch klare Regeln.
- **Hohe Datenqualität:** Stellt konsistente und plausible Zeitdaten sicher.
- **Flexibilität:** Anpassbare Perioden ermöglichen Regeländerungen ohne Aufwand.

---

## 🔍 **Zusammenfassung des Prozesses**

Mit den Arbeitszeitvalidierungsregeln legen Sie verbindliche Qualitätsstandards für Ihre Zeiterfassung fest. Aktivieren Sie relevante Regeln, definieren Sie Gültigkeitszeiträume und sorgen Sie so für saubere, rechtssichere Zeitdaten.

---