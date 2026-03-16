# Poool Rest-API & Webhooks

Article Description: Informationen zur Einrichtung und Nutzung der Poool API
Published: Yes
Suggested: No

## Einrichtung

Die Poool API wird je nach Software Paket initial von unserem Support eingerichtet. Wenden Sie sich dazu gerne an support@poool.cc. Bei der Gelegenheit freuen wir uns sehr, wenn Sie uns ihren Anwendungsfall schildern. Vielleicht können wir zur Anbindung einzelner Tools auch Tipps oder Vorlagen bereitstellen.

## Einrichtung API

Unter `System > API & Webhooks` finden Sie den Konfigurationsbereich. Hier können neue API Benutzer erstellt und berechtigt werden.

![image.png](Poool%20Rest-API%20&%20Webhooks/image.png)

Jeder API Benutzer hat einen oder mehere API Token welche, ähnlich wie ein Benutzer in Poool, unterschiedliche Berechtigungen und kann Aktiv oder Inaktiv sein. Jeder Token wird nach Anlage einmalig angezeigt (bitte an einer sicheren Stelle speichern)

Die Berechtigungen sollten sorgfältig gewählt werden und nur die Bereiche umfassen, die auch tatsächlich benötigt werden. Die verfügbaren Berechtigungen richten sich dabei nach den aktuell verfügbaren Endpunkten.

![image.png](Poool%20Rest-API%20&%20Webhooks/image%201.png)

## Einrichtung Webhooks

Webhooks helfen dabei, auf bestimmte Aktionen in Poool zu reagieren und Workflows in anderen Tools auszulösen. Zum Beispiel kann ein Kontakt nach Anlage in ein Newsletter Tool übertragen werden.

Webhooks werden im Reiter Webhooks angelegt und können Aktiv/Inaktiv gestellt werden.

![image.png](Poool%20Rest-API%20&%20Webhooks/image%202.png)

Für einen Webhook definieren wir:

- Einen Titel als allgemeine Beschreibung
- Status Aktiv / Inaktiv
- Eine Checksum zur Prüfung der Validität eines Aufrufs.
- Die URL, welche aufgerufen werden soll (meist aus dem Workflow Tool Ihrer Wahl als Trigger vorgegeben)
- Das Ereignis, bei welchem die URL aufgerufen werden soll

## Technische Dokumentation

<aside>
🔗

**Link zur technischen Dokumentation: [Poool API & Webhooks](https://app.poool.cc/api/docs/)**

</aside>

## Fragen / Anwendungsfälle

Bitte zögern Sie nicht, uns mit ihren Anwendungsfällen unter [support@poool.cc](mailto:support@poool.cc) zu kontaktieren - wir freuen uns auf den Austausch.