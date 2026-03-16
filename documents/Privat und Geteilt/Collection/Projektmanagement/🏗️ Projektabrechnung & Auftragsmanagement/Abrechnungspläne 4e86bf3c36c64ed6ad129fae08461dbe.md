# Abrechnungspläne

Article Description: Strukturierte Finanzübersicht - Gestalten Sie Ihre Abrechnungspläne mit Präzision.
Last Updated: 10. Juli 2023
Published: Yes
Suggested: No

## Arbeiten mit Abrechnungsplänen

Abrechnungspläne können Sie in Angeboten und Aufträgen nutzen um Zahlungsmodalitäten zu hinterlegen (z.B. *20% Anzahlung bei Beauftragung, 80% Restzahlung nach Fertigstellung*).

Bei **Angeboten** können Auftragsplanungen dazu genutzt werden, den Kunden über die entsprechenden Zahlungsmodalitäten zu informieren . 

Bei **Aufträgen** helfen Ihnen Abrechnungspläne, ohne großen Aufwand An- und Teilzahlungen zu ihrem Auftrag zu hinterlegen.

## Abrechnungspläne anlegen

Unter `Einstellungen → Abrechnung → Reiter: Abrechnungspläne` finden sie alle eingerichteten Abrechnungspläne.

![projektmanagement-pauschalprojekte-3.png](Abrechnungspl%C3%A4ne/projektmanagement-pauschalprojekte-3.png)

### **Vorhandene Abrechnungspläne**

Poool schlägt 2 Abrechnungspläne als Standard vor:

- *nach Gruppen*: Die Abrechnung erfolgt nach den in der Kalkulation definierten Leistungsgruppen.
- *nach Positionen*: Die Abrechnung erfolgt nach den in der Kalkulation definierten Leistungspositionen.
- 

### Neue Abrechnungspläne

Über `+ Abrechnungsplan` können sie neue Abrechnungspläne hinterlegen.

- **Titel**: Legt fest, mit welchem Titel der Auftragsplan in der Auswahlliste erscheint. Der Titel kann mittels Platzhalter `{ABRECHNUNGSPLAN_TITEL}` in die Textvorlagen Ihrer Angebote übernommen werden.
- **Beschreibung**: Dieser Text kann mittels Platzhalter `{ABRECHNUNGSPLAN_BESCHREIBUNG}` in die Textvorlagen Ihrer Angebote übernommen werden.
- **Modus**: Über den Modus legen Sie fest, ob die Abrechnungsschritte
    - *Manuell*: Sie legen die Prozentuale Verteilung der Abrechnungsschritte selbst fest.
    - *nach Gruppen*: Die Abrechnung erfolgt nach den in der Kalkulation definierten Leistungsgruppen.
    - *nach Positionen*: Die Abrechnung erfolgt nach den in der Kalkulation definierten Leistungspositionen.
    
    erstellt werden
    
- **Schritte** (bei Modus *Manuell*):
Hier legen Sie die einzelnen Zahlungsschritte, deren zeitliche Abfolge und den Anteil der Abrechnungssumme fest. Über das Beschreibungsfeld können Sie darüber hinaus dynamische Texte erstellen, die dann bei der Abrechnung entsprechend auf der Rechnung ausgewiesen werden können ([Abrechnungsassistent](../../Buchhaltung/Abrechnung/Abrechnungsassistent%2026a7806fc295433b9785e5072b977d86.md))

### Beispiel für einen Abrechnungsplan

Der gezeigte Screenshot zeigt einen Abrechnungsplan für eine 50% / 50% Abrechnung.

50% der Auftragssumme werden nach 0 Tagen (→ Bei Beauftragung) abgerechnet.
Die übrigen 50% werden nach 60 Tagen (kann nach Anwendung angepasst werden) abgerechnet.

![projektmanagement-pauschalprojekte-4.png](Abrechnungspl%C3%A4ne/projektmanagement-pauschalprojekte-4.png)

**Beschreibungsfelder**

> {AUFTRAGSPLANUNG_PROZENT}% Anzahlung lt. Auftrag {AUFTRAG_NUMMER}:
→ 50% Anzahlung lt. Auftrag AU-2023-001
> 

## Abrechnungspläne in Angeboten

Beim Erstellen von Angeboten können Sie bereits den geplanten Abrechnungsplan hinterlegen. 

[https://player.vimeo.com/video/872601147?badge=0&amp;autopause=0&amp;quality_selector=1&amp;progress_bar=1&amp;player_id=0&amp;app_id=58479](https://player.vimeo.com/video/872601147?badge=0&amp;autopause=0&amp;quality_selector=1&amp;progress_bar=1&amp;player_id=0&amp;app_id=58479)

Auf dem Angebotsdokument wird bei entsprechender Einrichtung in den Textvorlagen der Beschreibungstext des Zahlungsplans angezeigt.

![Untitled](../../Buchhaltung/Abrechnung/Abrechnungspl%C3%A4ne/Untitled%202.png)

![Untitled](../../Buchhaltung/Abrechnung/Abrechnungspl%C3%A4ne/Untitled%203.png)

## Abrechnungspläne in Aufträgen

Wird ein Angebot mit Abrechnungsplan beauftragt, wird der ausgewählte Abrechnungsplan direkt in der **Auftragsplanung** abgebildet. Für den Fall, dass sich etwas ändert, oder Sie keinen Abrechnungsplan ausgewählt haben, können Sie beim Auftrag in der Auftragsplanung die Angaben noch einmal ändern bzw. überschreiben.

[https://player.vimeo.com/video/872600643?badge=0&amp;autopause=0&amp;quality_selector=1&amp;progress_bar=1&amp;player_id=0&amp;app_id=58479](https://player.vimeo.com/video/872600643?badge=0&amp;autopause=0&amp;quality_selector=1&amp;progress_bar=1&amp;player_id=0&amp;app_id=58479)

In der Auftragsplanung scheinen auch die im Abrechnungsplan hinterlegten Beschreibungen auf. Die Abrechnung der einzelnen Positionen erfolgt anschließend über den Abrechnungsassistenten oder über die Auftragsplanung direkt.