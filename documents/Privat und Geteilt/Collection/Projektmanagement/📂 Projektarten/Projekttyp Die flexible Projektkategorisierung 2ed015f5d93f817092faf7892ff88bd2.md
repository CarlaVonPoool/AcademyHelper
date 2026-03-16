# Projekttyp: Die flexible Projektkategorisierung

Article Description: Erfahren Sie, wie Sie mit dem Projekttyp Ihre Projekte kategorisieren und nach übergeordneten Kriterien wie Jahresbudgets, Leistungsarten oder Standorten auswerten können.
Last Updated: 19. Januar 2026
Published: No
Suggested: No

## 📊 Artikel-Metadaten

|  |  |
| --- | --- |
| **Artikel-Typ** | Erklärung |
| **Erfahrungslevel** | 🚀 Grundlagen |

---

## 🤔 Die Herausforderung

Unternehmen arbeiten oft mit unterschiedlichen Projektarten, Jahresbudgets oder saisonalen Schwerpunkten. Für saubere Auswertungen und kundengerechtes Reporting braucht es eine Möglichkeit, Projekte nach einem übergeordneten Kriterium zu gruppieren – unabhängig davon, wann das Projekt tatsächlich läuft.

---

## 💡 Was ist der Projekttyp?

Der Projekttyp ist ein **frei definierbares Kategorisierungsfeld**, das Sie jedem Projekt zuweisen können. Er ermöglicht es, Projekte nach einem übergeordneten Kriterium zu gruppieren und auszuwerten.

**Wichtige Eigenschaften:**

- Ein Projekt kann immer nur **einen** Projekttyp haben
- Der Projekttyp wird in den **Einstellungen** zentral angelegt
- Die **Reihenfolge** der Projekttypen bestimmt die Sortierung im Dropdown

> ⚠️ **Wichtig:** Da nur ein Projekttyp pro Projekt möglich ist, sollten Sie sich im Vorfeld überlegen, nach welchem Kriterium Sie Ihre Projekte primär auswerten möchten. Für weitere Kategorisierungen können Sie **Tags** verwenden – diese sind flexibler und ein Projekt kann mehrere Tags haben.
> 

---

## ⭐ Vorteile im Überblick

- **Strukturierte Auswertungen:** Gruppierung und Filterung nach Projekttyp in allen relevanten Reports
- **Flexibles Kunden-Reporting:** Liefern Sie genau die Auswertungen, die Ihr Kunde benötigt (z.B. nach Budget-Jahr)
- **Umsatz-Ziele:** Definieren Sie Teilziele auf Projekttypebene
- **Projektnummern:** Nutzen Sie den Projekttyp als Variable in Ihrer Nummernstruktur (z.B. `WEB-2025-001`)

---

## 💼 Anwendungsfälle

### Szenario 1: Jahresbudgets / Scope-Jahre (Konzerne)

**Situation:** Ein Konzern vergibt Budgets nach Geschäftsjahren. Projekte laufen aber nicht sauber nach Kalenderjahr – es gibt Budget-Verschiebungen und Projekte, die über den Jahreswechsel hinaus laufen. Der Kunde möchte dennoch ein Reporting pro Budget-Jahr.

**Lösung mit Projekttyp:**

| Projekttyp | Budget | Beschreibung |
| --- | --- | --- |
| Scope 2024 | 2,5 Mio € | Alle Projekte aus dem 2024er Budget |
| Scope 2025 | 3,0 Mio € | Alle Projekte aus dem 2025er Budget |
| Scope 2026 | 1,8 Mio € | Alle Projekte aus dem 2026er Budget |

Ein Projekt aus "Scope 2024" kann noch in 2025 oder 2026 laufen – bleibt aber im richtigen Budget-Topf für das Kunden-Reporting.

### Szenario 2: Leistungsarten (Agenturen)

**Situation:** Eine Agentur bietet verschiedene Dienstleistungen an und möchte auswerten, welcher Bereich den meisten Umsatz generiert.

**Projekttypen:** Werbung, Digital, Beratung, Social Media, Messe

### Szenario 3: Saisons (Veranstaltungsbranche)

**Situation:** Ein Veranstaltungsunternehmen plant Events nach Saisons und möchte die Wirtschaftlichkeit pro Saison auswerten.

**Projekttypen:** Sommer, Winter

### Szenario 4: Standorte / Lokationen

**Situation:** Projekte sollen nach Veranstaltungsort oder Niederlassung ausgewertet werden.

**Projekttypen:** Standort 1, Standort 2, Messe, Arena

---

## ⚙️ Wo wird der Projekttyp angelegt?

Projekttypen werden zentral in den Einstellungen verwaltet:

`Einstellungen` → `Projekte` → Reiter `Projekttypen`

**Felder bei der Anlage:**

- **Titel** (Pflichtfeld): Der Name des Projekttyps – sollte aussagekräftig sein
- **Nummer** (optional): Kann in der Projektnummer und im Buchhaltungsexport verwendet werden

**Funktionen in der Übersicht:**

- **Reihenfolge ändern:** Über das Up/Down-Icon können Sie die Sortierung anpassen. Diese Reihenfolge wirkt sich auf das Dropdown-Auswahlfeld in den Projekteinstellungen aus.
- **Löschen:** Über das Mülleimer-Icon – aber nur wenn der Projekttyp noch keinem Projekt zugewiesen wurde. Ist das Icon grau/inaktiv, wird der Projekttyp bereits verwendet.

> 💡 **Tipp:** In den Systemeinstellungen (`Einstellungen` → `Allgemein`) können Sie den Projekttyp als **Pflichtfeld** aktivieren. Dann ist das Erstellen eines Projekts nur mit Angabe eines Projekttyps möglich.
> 

---

## 📍 Wo wird der Projekttyp am Projekt eingestellt?

Der Projekttyp wird direkt am Projekt hinterlegt:

- **Bei der Projektanlage:** Im Feld "Projekttyp" in der Projektanlagemaske
- **Nachträglich:** In den Projekteinstellungen eines bestehenden Projekts

---

## 🔍 Wo ist der Projekttyp in Poool sichtbar?

| Bereich | Wie wird der Projekttyp genutzt? |
| --- | --- |
| `Reporting` → `Projektauswertung` | Als **Gruppierungsebene** – Projekte werden pro Kunde nach Projekttyp gruppiert, mit Zwischensummen |
| `Reporting` → `Zielauswertung` | Für **Teilziele** auf Projekttypebene (Umsatz-Ziele) |
| `Projekte` → `Projektmanagement` | Als Filterkriterium und in der Projektübersicht |

---

## 📤 In welchen Exporten ist der Projekttyp enthalten?

| Export | Projekttyp enthalten? | Besonderheit |
| --- | --- | --- |
| **Projektauswertung-Export** | ✅ Ja | Option "Summe pro Projekttyp" für Zwischensummen |
| **Projektmanagement-Export** | ✅ Ja | Als Feld in jeder Zeile |
| **Stundenmanagement-Export (Intern)** | ✅ Ja | Nur im Intern-Export, nicht im Kunden-Export |
| **Stundenmanagement-Export (Kunde)** | ❌ Nein | – |

---

## 🎯 Umsatz-Ziele auf Projekttypebene

In `Einstellungen` → `Kosten & Ziele` → `Ziele` können Sie Umsatz-Ziele mit **Teilzielen pro Projekttyp** definieren. So sehen Sie in der Zielauswertung auf einen Blick, welcher Bereich wie viel zum Gesamtziel beiträgt.

**Beispiel:** Jahresziel 5 Mio € aufgeteilt auf:

- Scope 2025: 3 Mio €
- Scope 2024 (Restbudget): 2 Mio €

---

## ❓ Häufige Fragen

**Kann ein Projekt mehrere Projekttypen haben?**

Nein, ein Projekt kann immer nur einen Projekttyp haben. Für zusätzliche Kategorisierungen verwenden Sie Tags.

**Kann ich den Projekttyp bei mehreren Projekten gleichzeitig ändern?**

Nein, der Projekttyp kann nicht über eine Massenbearbeitung angepasst werden. Er muss bei jedem Projekt einzeln in den Projekteinstellungen geändert werden.

**Kann ich einen Projekttyp löschen?**

Nur wenn er noch keinem Projekt zugewiesen wurde. Ist das Mülleimer-Icon grau/inaktiv, wird der Projekttyp bereits verwendet.

**Kann ich den Projekttyp nachträglich ändern?**

Ja, der Projekttyp kann in den Projekteinstellungen jederzeit geändert werden – allerdings nur einzeln pro Projekt.

**Wie wirkt sich die Reihenfolge der Projekttypen aus?**

Die Reihenfolge bestimmt die Sortierung im Dropdown-Auswahlfeld bei der Projektanlage und in den Projekteinstellungen.

---

## 📚 Verwandte Artikel

### Praktische Umsetzung

- Neues Projekt anlegen
- Projektstatus & -typen hinzufügen

### Auswertungen & Exporte

- Projektauswertung
- Projektauswertung-Export
- Umsatz-Ziele

### Ähnliche Konzepte

- Nummernkreisläufe (Projekttyp in Projektnummer)
- Tags (für flexible Mehrfach-Kategorisierung)