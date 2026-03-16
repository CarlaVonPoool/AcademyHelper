# Rollen und Berechtigungen

Article Description: Wer darf was - Rollen und Berechtigungen für mehr Sicherheit in Poool.
Published: Yes
Suggested: No

# Rollen und Berechtigungen

Die Verwaltung von Rollen und Berechtigungen in Poool ist entscheidend, um sicherzustellen, dass Mitarbeiter die richtigen **Zugriffsrechte** haben und Aufgaben effizient erledigen können. 

## Rollen vordefinieren

Um Rollen mit bestimmten **Berechtigungen** anzulegen, navigieren Sie zu `Benutzer → Rollen`. Wir haben dort bereits ein paar **vordefinierte Rollen** angelegt, die unserer Erfahrung nach, gut zu den einzelnen Bereichen passen. Das kann natürlich in jedem Unternehmen **unterschiedlich** sein und daher können Sie frei entschieden, ob Sie die Rollen so übernehmen wollen, anpassen wollen oder löschen wollen. 

![Untitled](Rollen%20und%20Berechtigungen/Untitled.png)

### Rollen anpassen

Um eine Rolle **anzupassen**, öffnen Sie die gewünschte Rolle mit einem Klick darauf. Im ersten Schritt können Sie den **Titel** der Rolle anpassen. Überschrieben Sie den existierenden Titel mit einem von Ihnen gewünschten Titel.

![Untitled](Rollen%20und%20Berechtigungen/Untitled%201.png)

Der nächste Schritt sind die einzelnen **Rechte**. Die Rechte im System sind in folgende Teilbereiche gegliedert:

- **Basis**: Hier finden Sie die Basisrechte für Poool, wie beispielsweise Zugriffsrechte für Projekte, Angebote, Aufträge aber auch die Zeiterfassung.
- **Stundenmanagement**: Im Stundenmanagement geht es darum, wer gebuchte Zeiten beispielsweise bearbeiten oder löschen darf.
- **Eingangsrechnungen, Kosten**: Hier geht es vorrangig darum, wer Eingangsrechnungen sehen und bearbeiten kann aber auch um den Fremdkostenbereich einer Rechnung.
- **Reporting**: Im Reporting Bereich geht es um die einzelnen Auswertungen in Poool. Hier ist es sinnvoll, sich im Vorhinein Gedanken zu machen, wer Zugriff auch die Unternehmenszahlen haben darf.
- **Planner**: Hier geht es darum, wer Zugriff auf den Planner hat.
- **Freigaben**: Hier geht es um den Freigabeprozess in Poool.
- **Banking API**: Hier geht es um die Anbindung Ihrer Bank. Auch hier ist es wichtig, sich im Vorhinein Gedanken zu machen, wer die Transaktionen des Unternehmens einsehen darf.
- **Spesen**: In diesem Bereich geht es um die Spesenabrechnung in Poool.

Überlegen Sie sich, welche Rolle welche Rechte bekommen soll und setzen Sie die **Kästchen** entsprechend. Sollte es bei einer Position eingeschränkte Rechte geben, wird unter dem **Zugriffsrecht** beschrieben, was die eingeschränkten Rechte bedeuten. 

### **Verfeinerte Zugriffssteuerung auf Projekte und Tickets:**

Bei manchen Berechtigungen gibt es eine weitere präzisere Einschränkung, die beispielsweise den Zugriff auf Projekte und Tickets noch granularer steuert. Neben "Zugriff" und "Kein Zugriff" gibt es die Optionen "Eingeschränkter Zugriff" (nur verantwortliche Personen haben Zugriff) oder "Team-Zugriff" (nur Personen des jeweiligen Teams haben Zugriff).

![image.png](Rollen%20und%20Berechtigungen/image.png)

Wenn Sie die Rechte Ihren **Wünschen** entsprechend vergeben haben, drücken Sie auf `Speichern`, um die Rolle zu speichern. 

### Rollen löschen und neu anlegen

Wenn Sie die Rolle ganz **löschen** wollen und eine neue Rolle anlegen wollen, drücken Sie auf den **Papierkorb** auf der rechten Seite. 
Um eine **neue** Rolle zu erstellen, drücken Sie auf `+ Benutzerrolle`. Sie können dann wieder die **Liste** der Rechte durchgehen und entscheiden, welche Rechte di Rolle erhalten soll. Mit dem Button `Speichern`, wird die Benutzerrolle abgespeichert. 

![Untitled](Rollen%20und%20Berechtigungen/Untitled%202.png)

### Rollen vergeben

Um eine Rolle und somit die zugehörigen **Zugriffsrechte** einem Mitarbeiter zuzuteilen, navigieren Sie zu `Benutzer → Verwaltung`. Wählen Sie den gewünschten **Mitarbeiter** aus, dem Sie eine Rolle geben wollen oder erstellen Sie einen neuen Benutzer. 

<aside>
ℹ️ Wenn Sie nicht wissen, wie ein neuer Benutzer angelegt wird, finden Sie hier unseren Guide zu [Mitarbeiter anlegen](../../Setup%20&%20Konfiguration/Poool%20Setup/Anlage%20der%20Mitarbeiter%20546662b8693f4916890e027a8ad502d9.md) in dem auch beschrieben wird, wie ein Benutzer angelegt wird.

</aside>

Wechseln Sie im **Benutzer** auf den Raster **Zugriffsrechte.** Drücken Sie auf `bearbeiten` und wählen Sie bei `Rolle` die **entsprechende Rolle** aus, die Sie dem Mitarbeiter geben wollen. Mit der **Zuteilung** dieser Rolle bekommt der Mitarbeiter alle Rechte die der Rolle gewährt wurden. 

![Untitled](Rollen%20und%20Berechtigungen/Untitled%203.png)

## Benutzerdefinierte Berechtigungen vergeben

Wollen Sie Ihrem Mitarbeiter nun **zusätzliche Rechte** zu den Rechten der Rolle geben, setzen Sie den **Haken** bei “Benutzerdefinierte Zugriffsrechte auf Basis der Rolle verwenden”.

![Untitled](Rollen%20und%20Berechtigungen/Untitled%204.png)

Sobald Sie den **Haken** setzen, öffnet sich ein neues Fenster, in dem Sie gefragt werden, ob Sie die **benutzerdefinierten** Zugriffsrechte mit den Einstellungen der aktuellen Rolle **überschreiben** wollen oder die Zugriffsrechte **behalten** wollen. 

![Untitled](Rollen%20und%20Berechtigungen/Untitled%205.png)

Wenn Sie `Beibehalten` wählen, werden die **benutzerdefinierten Rechte**, die Sie bis dato für den Benutzer eingestellt haben, beibehalten. Wenn Sie auf `Überschreiben` drücken, werden die benutzerdefinierten Rechte mit den Rechten der Rolle überschrieben und der Benutzer hat **keine** benutzerdefinierten Rechte mehr. 
Sie können die **Rechte** für den einzelnen Benutzer jetzt **individuell** anpassen und sobald Sie mit den **Einstellungen** zufrieden sind, können Sie die Einstellungen mit dem Button `Speichern` abspeichern.