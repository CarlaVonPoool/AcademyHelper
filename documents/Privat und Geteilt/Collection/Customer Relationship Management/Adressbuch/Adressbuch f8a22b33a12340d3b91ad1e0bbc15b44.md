# Adressbuch

Article Description: Einfache Kontaktsuche: Wie du schnell und problemlos die gewünschten Informationen in deinem Adressbuch findest
Last Updated: 17. Juli 2023
Published: Yes
Suggested: Yes

# Adressbuch

Im Adressbuch werden alle relevanten Daten zu ihren **Geschäftskontakten** verwaltet. Dazu gehören u.a. Kontaktinformationen und Adressen ihrer Kunden, **Zahlungsinformationen** zu ihren Lieferanten oder auch **allgemeine Informationen** zu interessanten Persönlichkeiten. 

Das Adressbuch bildet die Basis Ihres Poool’s - Mit ihren Kunden starten sie **Projekte** im Projektmanagement und Lieferanten sind ein wesentlicher **Bestandteil** Ihrer Buchhaltung.

[https://vimeo.com/741798790](https://vimeo.com/741798790)

# Das Adressbuch

Unter `CRM → Adressbuch` finden Sie all Ihre Kontaktdaten an einem Ort. Über die linke **Seitenleiste** können Sie ihr Adressbuch nach Firmen, Personen oder Standorten durchsuchen. Darunter sehen sie die **zuletzt angelegten** Firmen und Personen. Im rechten Bereich erscheinen die Informationen zum ausgewählten Kontakt. Neben allgemeinen **Firmeninformationen** gibt es über die Reiter **Detailansichten** zu Personen, Standorten, Partnern, Kunden, Lieferanten, Leads und Zahlungsmitteln (Klick auf die 3 Punkte).

![adressbuch-1.png](Adressbuch/adressbuch-1.png)

# Kontakte anlegen

Mit `+ Kontakt` erstellen Sie einen neuen Kontakt in ihrem **Adressbuch**. Die Entscheidung, ob es sich bei dem Kontakt um einen Kunden, Lieferanten oder allgemeinen Kontakt handelt, treffen Sie später über die entsprechende **Aktivierung**.

![adressbuch-2.png](Adressbuch/adressbuch-2.png)

Es stehen Ihnen hier einige Standardfelder zur Verfügung.

Über den **internen Firmennamen** definieren Sie wie die Firma im Unternehmen weitläufig bezeichnet wird, im Unterschied **zum offiziellen Firmennamen**, wo wir empfehlen, den vom Kunden bevorzugten Namen für Anschriften einzutragen.

Für die Kunden- und Lieferantenverwaltung wichtig ist hier auch das **Kürzel**, über welches die Kontakte eindeutig **wiederzuerkennen** sind. Über das Dropdown “Feld hinzufügen” im rechten oberen Eck können Sie zusätzliche Datenfelder ergänzen (z.B. ein Notizfeld).

Über die individuell konfigurierten **Tags** können Kontakte kategorisiert werden. Das erweitert die Auswertbarkeit (Auswertung) des **Kundenstammes** und ermöglicht bestimmte Lieferanten schneller zu finden.

<aside>
❗ Die Option “*Art: Person*” ist an dieser Stelle zur Anlage von Ein-Personen Unternehmen. *Personen als Ansprechpartner* zu einzelnen Firmen werden direkt unter dem entsprechenden Kontakt erstellt.

</aside>

## Kontaktmöglichkeiten

Zu jedem Kontakt, egal ob Firma oder Person, lassen sich individuelle **Kontaktmöglichkeiten** konfigurieren. Hier lassen sich Telefonnummern, Fax, E-Mail Adressen, Webseiten und viele andere **Informationen** zum Kontakt hinterlegen.

![adressbuch-3.png](Adressbuch/adressbuch-3.png)

Über die Bezeichnung können **Kontaktmöglichkeiten** unterschieden werden (z.B. E-Mail Privat, E-Mail geschäftlich)

<aside>
❗ Kontaktmöglichkeiten für einzelne Ansprechpartner speichern Sie direkt zu den Personen der Firma. Es sollte **keine** Zeile *Telefon - James Gordon - +49 123 4562937* geben.

</aside>

![adressbuch-4.png](Adressbuch/adressbuch-4.png)

<aside>
ℹ️ Unter `System → Kontaktmöglichkeiten` können Sie individuelle Kontaktarten anlegen (z.B. Wordpress Backend, …)

</aside>

## Adressen

Zusätzlich zu den Kontaktmöglichkeiten können Sie eine oder mehrere **Anschriften** hinterlegen. Bei Angebots-/Auftrags- und Rechnungsstellungen kann auf diese Anschriften **zurückgegriffen** werden. Die eingetragene Anschrift wird automatisch in das **Dokument** hinzugefügt.

![adressbuch-5.png](Adressbuch/adressbuch-5.png)

<aside>
ℹ️ Nutzen Sie die Adressbezeichnung, um unterschiedliche Adressen zu kennzeichnen (z.B. Rechnungs- und Lieferanschrift). Zusätze wie *z.H. Person A* werden automatisch von Poool eingetragen.

</aside>

## Personen

Zu einer Firma können auch Personen und die dazugehörigen **Kontaktmöglichkeiten** und **Funktion** angelegt werden. Dies ermöglicht es Ihnen u.a. die Person als **Kontaktperson** für Projekte, Kalkulationen oder Tickets anzugeben.

![adressbuch-6.png](Adressbuch/adressbuch-6.png)

![adressbuch-7.png](Adressbuch/adressbuch-7.png)

Die individuelle **Briefanrede** wird hier hinterlegt, um Dokumente, welche direkt an diese Person versendet werden, automatisch zu personalisieren. Über den Platzhalter `{ANREDE}` können Sie die Anrede in Ihren **Textvorlagen** einbinden. 

Über die **Funktion** können sie die Rolle der Person im Unternehmen speichern. Auch für Personen stehen **Tags** zur Gruppierung zur Verfügung. Basierend auf diesen Tags lassen sich später z.B. **Mailinglisten** (Tag “Weihnachtskarte”) erstellen.

**Zusätzliche Felder** lassen sich auch hier über das Dropdown im rechten oberen Eck hinzufügen.

Die Übersicht bietet Ihnen die **Möglichkeit** direkt alle angelegten Personen einzusehen.

## Standorte

Über Standorte können Sie getrennte **Niederlassungen** eines Kontakts erfassen. Dabei stehen neben einer eigenen Standortbezeichnung auch eigene **Kontaktmöglichkeiten** und **Adressen** zur Verfügung. Standorte bilden oft eigene **Rechtsformen**, weshalb für einen Standort auch eine eigene UID angelegt werden kann.

❗ **Unterschied Standort - Adresse**

- Standorte haben im Vergleich zur zusätzlichen Adresse eine eigene UID.
- Einzelne Personen können einem Standort der Firma zugeordnet werden (einer Adresse nicht).

**Unterschied Standort - Eigener Kontakt**

- Bei einem Projekt kann bei der Adressauswahl einfach zwischen Standorten gewechselt werden.
- Einzelne Standorte können **nicht** separat ausgewertet werden. Dazu müssten eigene Kontakte angelegt werden.