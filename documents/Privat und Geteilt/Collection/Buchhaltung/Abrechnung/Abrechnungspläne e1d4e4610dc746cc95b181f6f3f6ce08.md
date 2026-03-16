# Abrechnungspläne

Article Description: Flexibilität und Genauigkeit: Anleitung zur präzisen Erstellung von Abrechnungsplänen
Last Updated: 9. Oktober 2023
Published: Yes
Suggested: No

## Arbeiten mit Abrechnungsplänen

Abrechnungspläne können Sie in Angeboten und Aufträgen nutzen um Zahlungsmodalitäten zu hinterlegen (z.B. *20% Anzahlung bei Beauftragung, 80% Restzahlung nach Fertigstellung*).

Bei **Angeboten** können Auftragsplanungen dazu genutzt werden, den Kunden über die entsprechenden Zahlungsmodalitäten zu informieren . 

Bei **Aufträgen** helfen Ihnen Abrechnungspläne, ohne großen Aufwand An- und Teilzahlungen zu ihrem Auftrag zu hinterlegen.

## Abrechnungspläne anlegen

Unter `Einstellungen → Abrechnung → Reiter: Abrechnungspläne` finden sie alle eingerichteten Abrechnungspläne.

![Untitled](Abrechnungspl%C3%A4ne/Untitled.png)

### **Vorhandene Abrechnungspläne**

Poool schlägt 2 Abrechnungspläne als Standard vor:

- *nach Gruppen*: Die Abrechnung erfolgt nach den in der Kalkulation definierten Leistungsgruppen.
- *nach Positionen*: Die Abrechnung erfolgt nach den in der Kalkulation definierten Leistungspositionen.

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
Hier legen Sie die einzelnen Zahlungsschritte, deren zeitliche Abfolge und den Anteil der Abrechnungssumme fest. Über das Beschreibungsfeld können Sie darüber hinaus dynamische Texte erstellen, die dann bei der Abrechnung entsprechend auf der Rechnung ausgewiesen werden können (‣).

### Beispiel für einen Abrechnungsplan

Der gezeigte Screenshot zeigt einen Abrechnungsplan für eine 50% / 50% Abrechnung.

50% der Auftragssumme werden nach 0 Tagen (→ Bei Beauftragung) abgerechnet.
Die übrigen 50% werden nach 60 Tagen (kann nach Anwendung angepasst werden) abgerechnet.

![Untitled](Abrechnungspl%C3%A4ne/Untitled%201.png)

**Beschreibungsfelder**

> {AUFTRAGSPLANUNG_PROZENT}% Anzahlung lt. Auftrag {AUFTRAG_NUMMER}:
→ 50% Anzahlung lt. Auftrag AU-2023-001
> 

> 50% Anzahlung mit Rechnung {RECHNUNG_1_NUMMER}
→ 50% Anzahlung mit Rechnung RE-2023-001
> 

## Abrechnungspläne in Angeboten

Beim Erstellen von Angeboten können Sie bereits den geplanten Abrechnungsplan hinterlegen. 

[Untitled](Abrechnungspl%C3%A4ne/Untitled.mp4)

Auf dem Angebotsdokument wird bei entsprechender Einrichtung in den Textvorlagen der Beschreibungstext des Zahlungsplans angezeigt.

![Untitled](Abrechnungspl%C3%A4ne/Untitled%202.png)

![Untitled](Abrechnungspl%C3%A4ne/Untitled%203.png)

## Abrechnungspläne in Aufträgen

Wird ein Angebot mit Abrechnungsplan beauftragt, wird der ausgewählte Abrechnungsplan direkt in der **Auftragsplanung** abgebildet. Für den Fall, dass sich etwas ändert, oder Sie keinen Abrechnungsplan ausgewählt haben, können Sie beim Auftrag in der Auftragsplanung die Angaben noch einmal ändern bzw. überschreiben.

[Untitled](Abrechnungspl%C3%A4ne/Untitled%201.mp4)

In der Auftragsplanung scheinen auch die im Abrechnungsplan hinterlegten Beschreibungen auf. Die Abrechnung der einzelnen Positionen erfolgt anschließend über den Abrechnungsassistenten oder über die Auftragsplanung direkt.