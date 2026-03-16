# DATEV/BMD Setup

Article Description: Vereinfachte Buchführung - Unser DATEV/BMD Setup für nahtlose Integration und Kontrolle.
Last Updated: 9. Oktober 2023
Published: Yes
Suggested: No

## **Kunden- und Lieferantenkonten**

### **Option 1: Kunden-/Lieferantennummer**

Sobald Sie Ihren Kunden / Lieferanten im CRM aktiviert haben, werden automatisiert Kunden- oder Lieferantennummern von Poool vergeben. Ihr Steuerberater arbeitet somit mit der Kunden- oder Lieferantennummer von Poool.

### **Option 2: Manuelle Eingabe DATEV Konto**

Selbstverständlich können wir auch die Kunden- oder Lieferantennummer Ihres Steuerberaters übernehmen. Die Kunden- oder Lieferantennummer können Sie manuell unter `CRM → Adressbuch` im Reiter `Kunde` oder `Lieferant` bearbeiten und unter „DATEV-Konto“ eingegeben.

![Bild1.png](DATEV%20BMD%20Setup/Bild1.png)

<aside>
ℹ️ **Sammelkunde**: DATEV Sammelkunden sind dazu gedacht, bei einem Kunden mit mehreren Adressen und Anschriften aber einem gemeinsamen Konto jeweils die erste Adresszeile der Rechnung als Kundenname zu exportieren.
Bei “normalen” Kunden wird immer der Firmenname offiziell als Kundenname exportiert.

</aside>

## **Erlöskonten bei Ausgangsrechnungen (Steuersatz der Gruppe)**

Die Erlöskonten bei Ausgangsrechnungen sowie die Umsatzsteuersätze können unter `System → Umsatzsteuer`  mit dem Button `+ Umsatzsteuer` hinterlegt werden. Sollte Ihnen ein Steuersatz bzw. ein Erlöskonto fehlen, können Sie dieses jederzeit ergänzen. Bitte hier nur tatsächlich für Ausgangsrechnungen benötigte Konten eintragen (Erlöse aus Förderungen werden eher selten in Poool abgebildet).

![Bild2.png](DATEV%20BMD%20Setup/Bild2.png)

*Voreinstellung* = dieser Steuersatz wird automatisiert als Standard-Steuersatz bei Ausgangs- oder Eingangsrechnungen verwenden

![Bild3.png](DATEV%20BMD%20Setup/Bild3.png)

Die Umsatzsteuersätze können bei Erstellung eines Angebots pro Kunde individuell eingegeben werden. Dafür klicken Sie auf die Gruppenbeschreibung und geben dort den gewünschten Steuersatz ein. Dies ist auch bei Aufträgen und Ausgangsrechnungen möglich.

![Bild4.png](DATEV%20BMD%20Setup/Bild4.png)

Natürlich ist es auch möglich, mehrere Steuersätze zu verwenden. Für jeden Steuersatz muss allerdings immer eine separate Gruppe angelegt werden.

## **Aufwandskonten bei Eingangsrechnungen (Kostenaufteilung)**

Die Aufwandskonten bei Eingangsrechnungen können Sie unter `Einstellungen → Kosten & Ziele` im Reiter `Kostenaufteilung` mit `+ Kostenaufteilung` selbstständig anlegen. Auch hier sollten nur tatsächlich benötigte Aufwandskonten hinterlegt werden (Aufwände für Steuern werden i.d.R. selten in Poool erfasst).

![Bild5.png](DATEV%20BMD%20Setup/Bild5.png)

![Bild6.png](DATEV%20BMD%20Setup/Bild6.png)

Bei Eingangsrechnungen können Sie die Aufwandskosten einzelnen Positionen zuteilen, indem Sie bei Aufteilung das gewünschte Konto hinterlegen.

![Bild7.png](DATEV%20BMD%20Setup/Bild7.png)

Nun sind alle wichtigen Informationen für den Buchhaltungsexport hinterlegt und Sie können Ihren Buchhaltungsexport starten und Ihrem Steuerberater zukommen lassen.

Wie Sie den Buchhaltungsexport anstoßen können, finden Sie im folgenden Guide:

 [Buchhaltungsexport](Buchhaltungsexport%205ed6ba26e2b54723bdadaba0397def5b.md) 

Sollten Sie bereits unseren neuen Buchhaltungsexport haben oder sich für diesen interessieren, finden Sie im folgenden Guide mehr:

[Einrichtung Buchhaltungsexport](Einrichtung%20Buchhaltungsexport%20616dd9aac3df44b4aacd82f872a87537.md) 

<aside>
ℹ️

### Wichtige Hinweise zur Eingabe von Rechnungsdaten für den Export

Beim Export der Eingangsrechnungen sind folgende Formatvorgaben zu beachten:

- **Maximale Zeichenlänge**: Rechnungs- und Positionstitel sowie Kunden-/Lieferantennamen dürfen maximal **60 Zeichen** lang sein.
- **Zeichenbeschränkungen**: Keine Umlaute (ä, ö, ü), kein „ß“, keine Satz- oder Sonderzeichen verwenden.
- **Rechnungsnummern**: Keine Leerzeichen in Rechnungsnummern einfügen.
- **Monatliche Abrechnungen**: Zur besseren Unterscheidung sollte bei wiederkehrenden Abrechnungen (z. B. Strom, Miete) der **Monat im Titel** angegeben werden.
</aside>