# Stunden-Freigaben und Ablehnungen

Last Updated: 29. Oktober 2025
Published: Yes
Suggested: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

# **Stunden-Freigaben und Ablehnungen**

Die Stundenfreigabe ist ein zentraler Bestandteil zur Qualitätssicherung in der Zeiterfassung – vom ersten Zeiteintrag bis hin zur Abrechnung. Jede Buchung durchläuft dabei einen strukturierten Prüfprozess mit klar definierten Rollen und Statuswerten.

## 📌 Status von Stundenbuchungen

Jede Zeiterfassungsbuchung kann einen von drei Status haben:

- **Offen** – noch nicht geprüft
- **Freigegeben** – geprüft und abrechnungsfähig
- **Abgelehnt** – fehlerhaft oder unklar, muss überarbeitet werden

Diese Status sind direkt im Stundenmanagement sichtbar und steuerbar.

## 🔒 **Stufe 1: Prüfung der gebuchten Stunden**

![image.png](Stunden-Freigaben%20und%20Ablehnungen/image.png)

Unter `Buchhaltung → Stundenmanagement` prüfen in der Regel Projektmanager die gebuchten Stunden. In der Spalte **"Freigabe"** stehen zwei Icons zur Verfügung:

- ✔️ (Häkchen) - **Freigabe** der Stunden
- ❌ (X) - **Ablehnung** der Stunden

Beim Klick auf ein Icon blinkt die entsprechende Zeile kurz farbig – **grün** bei Freigabe, **rot** bei Ablehnung. Das aktivierte Symbol wird hervorgehoben. Ein erneuter Klick auf das aktive Icon setzt den Status wieder auf **offen** zurück.

### ✅ **Freigabe von Stunden**

Die Stunden werden mit Klicken des Hakens bestätigt und freigegeben. Sie gelten nun als abrechnungsfähig und können in die spätere Rechnungsstellung einfließen. Ist eine Stunde freigegeben worden, kann der Zeiterfasser die Stundenbuchung **nicht mehr verändern**. 

### ❌ **Ablehnung von Stunden**

Stunden, die unklar oder fehlerhaft sind, können **abgelehnt** werden. Es können hierfür Ablehnungsgründe definiert werden, um einen möglichst effizienten Prozess der Weiterverarbeitung gebuchter Stunden zu gewährleisten. Unter ****`Einstellungen → Zeiterfassung` muss die Option **“Ablehnungsgründe für Stundenfreigabe”** aktiviert werden. 

![image.png](Stunden-Freigaben%20und%20Ablehnungen/image%201.png)

Im Anschluss können Ablehnungsgründe **** im gleichen Fenster unter dem Reiter ****`Ablehnungsgründe` frei konfiguriert werden.

Beispiele hierfür sind:

- Unklare Beschreibung der Tätigkeit
- Falsches Projekt oder Ticket

![image.png](Stunden-Freigaben%20und%20Ablehnungen/image%202.png)

### 🔄️ Mehrere Stunden gleichzeitig bearbeiten

Um Zeit zu sparen, können Sie auch **mehrere Stunden** gleichzeitig freigeben, ablehnen oder bearbeiten. Aktivieren Sie dazu auf der linken Seite die Checkboxen der gewünschten Einträge. Sobald mehrere Zeilen ausgewählt sind, erscheinen oben zusätzliche **Sammelaktionen**, mit denen Sie alle gewählten Stunden **gemeinsam** bearbeiten können.

![image.png](Stunden-Freigaben%20und%20Ablehnungen/4b9df1f9-cfff-485f-b049-752004125878.png)

## 🔁 **Stufe 2: Korrektur durch den Zeiterfasser (optional)**

Abgelehnte Stunden gehen automatisch zurück an den Zeiterfasser, der die Buchung auf Basis des Ablehnungsgrunds überarbeiten kann. Unter ****`Zeiterfassung` im Reiter `Abgelehnte Stunden` kann er die einzelnen Zeitevents einsehen und direkt bearbeiten.

![image.png](Stunden-Freigaben%20und%20Ablehnungen/image%203.png)

Nach der Korrektur landen diese Stunden wieder im Stundenmanagement und liegen der zuständigen Person zur erneuten Prüfung vor.

Dieser Kreislauf aus **Ablehnung, Bearbeitung und Freigabe** kann beliebig oft durchlaufen werden, bis die Zeiten final korrekt sind.

## 📤 **Stufe 3: Rechnungsstellung freigegebener Stunden**

Sobald die Stunden **freigegeben** sind, stehen sie für die **Verrechnung in Rechnungen** zur Verfügung. Dadurch ist sichergestellt, dass nur geprüfte und bestätigte Leistungen abgerechnet werden.
Im Abrechnungsassistenten werden standardmäßig nur freigegebene Stunden angezeigt. Es besteht jedoch die Möglichkeit, auch nicht freigegebene Stunden einzublenden, indem die Option **„Alle Stunden anzeigen“** aktiviert wird.

![image.png](Stunden-Freigaben%20und%20Ablehnungen/image%204.png)

### 🛠️ **Sonderfall: Änderungen durch z.B. Controlling**

In Ausnahmefällen können auch bereits freigegebene Stunden nachträglich **bearbeitet werden**, z. B. durch das Controlling im Rahmen einer internen Prüfung. Diese Möglichkeit ist an ein Rollenrecht geknüpft, das individuell im Bereich `Benutzer → Benutzerrollen` konfigurierbar ist.

Nur Nutzer mit entsprechender Berechtigung können so direkt auf bereits freigegebene Zeiten zugreifen und Anpassungen vornehmen – z. B. bei Korrekturen im Rahmen einer internen Prüfung oder bei nachträglichen Projektänderungen.

![image.png](Stunden-Freigaben%20und%20Ablehnungen/image%205.png)

## 🔍 **Zusammenfassung des Prozesses**

| Rolle | Aufgabe |
| --- | --- |
| **Zeiterfasser** | Erfasst Stunden und bearbeitet seine eigenen abgelehnten Stundenbuchungen. |
| **Projektmanager** | Prüft Stunden, gibt sie frei oder lehnt sie ab. |
| **Controller** | Kann (mit Berechtigung) freigegebene Stunden anpassen. |