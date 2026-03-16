# Gleitzeit

Article Description: Flexible Arbeitszeit durch Gleitzeit – Zeitausgleich automatisch verwalten 
Last Updated: 25. November 2025
Published: Yes
Suggested: No

# Gleitzeit in Poool einrichten und verwalten

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

Gleitzeit in Poool ist gesondert geregelt und unterscheidet sich klar von [Überstunden](%C3%9Cberstunden%201a3eb9448f5f43eeb2da0ed8c090532f.md). In Ihrem Gleitzeitkonto können Sie **Plus- und Minusstunden** ansammeln, die automatisch ausgeglichen werden. Sollten Sie an einem Tag mal früher gehen oder länger bleiben, werden Ihre Mehr- und Minusstunden kontinuierlich gegeneinander aufgerechnet.

Ihren aktuellen **Gleitzeitstand** können Sie jederzeit unten rechts in der Zeiterfassung oder in Ihrem **Persönliche Info Widget** einsehen.

---

## ✅ Systemvoraussetzungen

- **Arbeitszeiten** sind beim Mitarbeiter hinterlegt `Unternehmen → Mitarbeiter`

---

## 📌 Schritt-für-Schritt: Gleitzeit einrichten

### Schritt 1: Gleitzeitkonto für Mitarbeiter aktivieren

- Navigieren Sie zu `Unternehmen → Mitarbeiter`
- Wählen Sie den gewünschten Mitarbeiter aus
- Scrollen Sie zu **„**Zeiterfassung**"**
- Aktivieren Sie  `Gleitzeitkonto führen`
- **Speichern** Sie die Änderungen

![image.png](Gleitzeit/image.png)

**Wichtig:** Die Berechnung erfolgt automatisch auf Basis der hinterlegten Sollarbeitszeit des Mitarbeiters.

---

### Schritt 2: Zeiterfassungstyp "Zeitausgleich" konfigurieren

Damit Mitarbeitende ihr Gleitzeitkonto nutzen können, benötigen Sie einen passenden **Zeiterfassungstyp**:

- Gehen Sie zu `Einstellungen → Zeiterfassung → Zeiterfassungstypen`
- Klicken Sie auf `+ Zeiterfassungstyp`
- Konfigurieren Sie wie folgt:
    - **Titel:** Zeitausgleich (oder Gleitzeit)
    - **Basistyp:** ⚠️ **Ruhezeit** (sehr wichtig! Nur so wird vom Gleitzeitkonto abgezogen)
    - **Verfügbarkeit:** untertägig, halbtägig, ganztägig (je nach Bedarf)
    - **Arbeitsort verwendbar:** Nein
    - **Antrag erforderlich:** Optional (abhängig von Ihrer Freigabe-Policy)
    - **Genehmigung erforderlich:** Optional
    - **Icon & Farbe:** Nach Wunsch (z.B. 🏖️ in Blau)
- **Speichern**

💡 Hinweis**:** Der Basistyp "Ruhezeit" sorgt dafür, dass bei einer Zeitausgleichs-Buchung:

- Die Abwesenheit vom Gleitzeitkonto abgezogen wird
- In der Zeiterfassung sichtbar ist, dass der Mitarbeiter nicht anwesend ist
- Kein Urlaubstag verbraucht wird

**Beispiel:** Ein Mitarbeiter hat +8h Gleitzeit aufgebaut. Er bucht am Freitag "Zeitausgleich" für 4 Stunden. Sein Gleitzeitkonto reduziert sich auf +4h.

---

## 👥 Wo sehen Mitarbeitende ihre Gleitzeit?

Mitarbeitende können ihren aktuellen Gleitzeitstand an **drei Stellen** einsehen:

### 1. Widget "Persönliche Infos" (Dashboard)

- Aktueller Gleitzeitstand
- Aktuelle Periode mit Start-/Enddatum
- Veränderung im Zeitraum
- Gekappte Stunden (falls vorhanden)

![image.png](Gleitzeit/image%201.png)

### 2. Zeiterfassung (unten rechts)

- Live-Anzeige des aktuellen Stands
- Direkt beim Buchen von Zeiten sichtbar

![image.png](Gleitzeit/image%202.png)

### 3. Zeiterfassung → Reporting

- Aktuelle Gleitzeitperiode grafisch dargestellt
- Warnung bei drohender Kappung (rote Stunden)

---

## 🔗 Weiterführende Artikel

[Arbeitszeiterfassung](Arbeitszeiterfassung%20672951a3d27b48c99ba3c22a7aaaa277.md)

[Projektzeiterfassung](Projektzeiterfassung%20490f3bfd76e34113a68da4a3a67c6675.md)

[Überstunden](%C3%9Cberstunden%201a3eb9448f5f43eeb2da0ed8c090532f.md)

[Gleitzeit- und Stundenkappung](Gleitzeit-%20und%20Stundenkappung%2025b015f5d93f808f832ee74e8944a413.md)

[Management der Mitarbeiterzeiterfassung](Management%20der%20Mitarbeiterzeiterfassung%200bf1e750ed4c47928743923ff7e90a80.md)

[Stundenmanagement](Stundenmanagement%20f601b175a546410998d5867f3d9cd0d4.md)

[Abwesenheiten beantragen](../Abwesenheiten/Abwesenheitsantr%C3%A4ge%20stellen%20564d480876544e75844444209ea9f3e2.md)

[Abwesenheiten verwalten](../Abwesenheiten/Abwesenheitsantr%C3%A4ge%20verwalten%202a9015f5d93f80989084edd955163942.md)