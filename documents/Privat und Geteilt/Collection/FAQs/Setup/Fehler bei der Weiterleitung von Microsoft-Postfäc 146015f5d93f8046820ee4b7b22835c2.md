# Fehler bei der Weiterleitung von Microsoft-Postfächern an die Billbox

Published: Yes
Suggested: No

Standardmäßig blockiert **Microsoft 365** aus Sicherheitsgründen die Weiterleitung von E-Mails an **externe** Empfänger. Wenn eine Weiterleitung scheitert, wird eine **Fehlermeldung** wie die folgende zurückgesendet:

![Fehler.png](Fehler%20bei%20der%20Weiterleitung%20von%20Microsoft-Postf%C3%A4c/Fehler.png)

Diese Fehlermeldung ist nur ein **Beispiel**. Es kann auch eine andere **Fehlermeldung** auftauchen.

Um die **Weiterleitung** von E-Mails an externe Empfänger wie die Billbox zu ermöglichen, müssen Sie in den **Anti-Spam-Richtlinien** von Microsoft 365 eine Ausnahme einrichten. Gehen Sie dazu wie folgt vor:

1. **Öffnen Sie die Anti-Spam-Richtlinien im Security-Bereich von Microsoft 365:**
    
    Navigieren Sie zu [https://security.microsoft.com/antispam](https://security.microsoft.com/antispam).
    
2. **Erstellen Sie eine neue ausgehende Richtlinie:**
    
    Klicken Sie auf `+ Richtlinie erstellen` ****und wählen Sie `ausgehend von` aus.
    
3. **Benennen Sie die Richtlinie:**
    
    Geben Sie der neuen Richtlinie einen eindeutigen Namen und eine Beschreibung, z. B. *„Erlaubnis für externe Weiterleitung“*.
    
4. **Wählen Sie die Benutzerkonten aus:**
    
    Klicken Sie auf `Weiter` und suchen Sie nach dem Benutzerkonto, das E-Mails weiterleiten soll (z. B. *support-test@poool.cc*). Wählen Sie dieses Konto aus.
    
5. **Aktivieren Sie die automatische Weiterleitung:**
    
    Scrollen Sie im Abschnitt `Weiterleitungsregeln` nach unten. Im Dropdown-Menü unter `Automatische Weiterleitungsregeln` wählen Sie `Ein - Weiterleitung ist aktiviert` aus und klicken auf `Weiter`.
    
6. **Speichern und aktivieren Sie die Richtlinie:**
    
    Überprüfen Sie alle Einstellungen und klicken Sie abschließend auf `Richtlinie erstellen`, um die neue Richtlinie zu speichern. Das sollte dann wie folgt aussehen:
    
    ![Test1.PNG](Fehler%20bei%20der%20Weiterleitung%20von%20Microsoft-Postf%C3%A4c/Test1.png)
    

Die **Änderungen** treten in der Regel innerhalb **weniger Stunden** in Kraft, können aber in Einzelfällen bis zu 24 Stunden dauern. Sobald die neue Richtlinie **aktiv** ist, wird die Weiterleitung an externe Empfänger wie die Billbox **zugelassen**. 

<aside>
ℹ️

Beachten Sie, dass diese Funktion sicherheitskritisch ist und nur für **vertrauenswürdige Empfänger** verwendet werden sollte.

</aside>