# Gleitzeit- und Stundenkappung

Article Description: Gleitzeit transparent verwalten mit automatischer Stundenkappung.
Last Updated: 27. Oktober 2025
Published: Yes
Suggested: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## ⚖️ Gleitzeitkappung

Mit dem Feature **Gleitzeitperioden und automatische Stundenkappung** können Sie Gleitzeitregelungen im Unternehmen transparent und regelkonform verwalten. Am Ende jeder definierten Periode wird das Gleitzeitkonto automatisch auf eine festgelegte Höchstgrenze begrenzt. Überzählige Stunden werden protokolliert und nicht übertragen – ganz ohne manuelles Eingreifen.

### ✅ Systemvoraussetzungen & Einstellungen

Die automatische Gleitzeitkappung aktivieren Sie unter `Zeiterfassung **→** Einstellungen` an. 

### 👥 Gleitzeitperioden konfigurieren

Sobald die Funktion freigeschalten wurde, definieren Sie die Gleitzeitperioden der Mitarbeiter und die jeweilige Kappungsgrenze. 

- Öffnen Sie **`Unternehmen → Mitarbeiter`**
- Wählen Sie die gewünschte Person
- Aktivieren Sie das **Gleitzeitkonto und die verpflichtende Zeiterfassung**
- Legen Sie eine Gleitzeitperiode und ein Intervall fest. Folgende Kappungsintervalle stehen hier zur Verfügung:
    - Wöchentlich
    - Monatlich
    - Vierteljährlich
    - Jährlich
    - Benutzerdefiniert
- Definieren Sie die **Kappungsgrenze** (z. B. 3 Stunden)

![image.png](Gleitzeit-%20und%20Stundenkappung/image.png)

tra

<aside>
💡

 Diese Information kann alternativ auch via Personio-Schnittstelle übergeben werden. 

</aside>

### Abbildung im Zeitkonto

Gekappte Stunden werden nach Ablauf der Periode nicht gelöscht, sondern **klar im Zeitkonto dokumentiert**. Am Ende jeder Gleitzeitperiode prüft das System automatisch, ob die definierte Kappungsgrenze überschritten wurde. Ist das der Fall, wird die Differenz als **„gekappte Stunden“** ausgewiesen.

Unter `Unternehmen → Mitarbeiter → Zeitkonto` sehen Sie den **aktuellen Gleitzeitstand** sowie eine **historische Auflistung aller Gleitzeitperioden**. Hier wird transparent dargestellt:

- wie viele Stunden pro Periode aufgebaut wurden,
- wie viele davon gekappt wurden,
- und welcher Saldo ins nächste Zeitkonto übertragen wurde.

Diese Übersicht bietet sowohl für HR als auch für Mitarbeitende eine klare und nachvollziehbare Dokumentation.

![image.png](Gleitzeit-%20und%20Stundenkappung/image%201.png)

### Neuberechnung bei Korrekturen

Bei nachträglichen Änderungen (z. B. manuelle Korrekturen oder Import aus Personio) können Sie die Gleitzeit **neu berechnen**. Das System passt dabei automatisch alle Werte der betroffenen Periode an – inklusive gekappter Stunden.

### Widget: Persönliche Infos

Im Widget “Persönliche Infos” der Mitarbeitenden auf Ihrem Dashboard sehen Mitarbeitende ihren **aktuellen Gleitzeitstand**. Beim Aufklappen werden zusätzlich angezeigt:

- Aktuelle Periode mit Start- und Enddatum
- Veränderung im Zeitraum
- Gekappte Stunden (sofern vorhanden)

![image.png](Gleitzeit-%20und%20Stundenkappung/image%202.png)

### Zeiterfassung / Reporting

Unter **`Zeiterfassung - Reporting`** finden Sie nun zusätzlich folgende Informationen. 

- Die aktuelle **Gleitzeitperiode**, grafisch dargestellt. Rote Stunden werden - wenn nicht konsumiert - am Ende der Periode gekappt.
- Alle historischen Gleitzeitperioden und die dazugehörigen Kappungsinformationen.

![image.png](Gleitzeit-%20und%20Stundenkappung/image%203.png)

### 💼 Anwendungsfälle

### Szenario 1: Gleitzeit im Monatsrhythmus begrenzen

Ein Unternehmen legt fest, dass maximal **20 Überstunden pro Monat** aufgebaut werden dürfen. Am Ende jedes Monats prüft das System automatisch das Gleitzeitkonto und kappt alle Stunden, die über der festgelegten Grenze liegen. Diese **nicht übertragenen Stunden** werden dokumentiert und im Zeitkonto transparent ausgewiesen. Mitarbeitende sehen jederzeit, wie viel Gleitzeit sie angesammelt – und ggf. verloren – haben.

### Szenario 2: Gleitzeitübertrag ins neue Jahr begrenzen

Zum Jahresende soll sichergestellt werden, dass **maximal 40 Stunden ins neue Jahr übernommen werden**. Die Gleitzeitkappung wird auf **jährliche Perioden** konfiguriert und übernimmt automatisch die Begrenzung. Alles, was über 40 Stunden liegt, wird gekappt und ebenfalls im Zeitkonto nachvollziehbar angezeigt. So behalten Sie jederzeit die Kontrolle über langfristige Gleitzeitansammlungen – ohne manuelle Berechnungen oder Korrekturen.