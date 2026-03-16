# Digitale Rechnungen mit ZUGFeRD & XRechnung

Article Description: Digitale Rechnunge ganz einfach in Poool nach XRechnung oder ZUGFeRD Standard erstellen
Published: Yes
Suggested: No

Bei Poool können Sie bei der Erstellung von Rechnungsdokumenten **drei** Formate auswählen: 

- PDF
- ZUGFeRD-Datei
- XRechnungs-Datei

ZUGFeRD und XRechnung sind international wichtige Standards. Diese Formate ermöglichen es Ihnen, effizient und rechtlich konforme elektronische Rechnungen auszutauschen. Im Austausch mit öffentlichen Auftraggebern ist die XRechnung in Deutschland der gesetzliche Standard.

### Schritt 1:

**Das richtige Setup**

Um die Formate richtig nutzen zu können, müssen Sie vorab ein paar wichtige Einstellungen treffen. Unter “System” unter “Einheiten” und “Mehrwertsteuer”  müssen einmalig die richtigen ZUGFeRD Codes übernommen werden. 

![zugferd-und-xrechnung-einheiten-1.png](Digitale%20Rechnungen%20mit%20ZUGFeRD%20&%20XRechnung/zugferd-und-xrechnung-einheiten-1.png)

![zugferd-und-xrechnung-einheiten-2.png](Digitale%20Rechnungen%20mit%20ZUGFeRD%20&%20XRechnung/zugferd-und-xrechnung-einheiten-2.png)

![image.png](Digitale%20Rechnungen%20mit%20ZUGFeRD%20&%20XRechnung/image.png)

### Schritt 2:

**Die Bankanbindung**

Ein geeignetes Bankkonto zu hinterlegen, ist ebenfalls ein essenzieller Schritt, um die Formate nutzen zu können. Für diesen Schritt gehen Sie unter “CRM” und “Adressbuch” in Ihren eigenen Datensatz. Über die drei Punkte oben rechts können Sie auf die Zahlungsmittel übergehen und dort das richtige Bankkonto verbinden. Hier wählen Sie bei “Standard” das “Hauptkonto (ZUGFeRD und XRechnung) aus. Damit sind die nötigen Schritte im Hintergrund abgeschlossen.

![zugferd-und-xrechnung-bankverbindung-1.png](Digitale%20Rechnungen%20mit%20ZUGFeRD%20&%20XRechnung/zugferd-und-xrechnung-bankverbindung-1.png)

![zugferd-und-xrechnung-bankverbindung-2.png](Digitale%20Rechnungen%20mit%20ZUGFeRD%20&%20XRechnung/zugferd-und-xrechnung-bankverbindung-2.png)

### Schritt 3:

**Das richtige Format wählen**

Bei der Rechnungserstellung innerhalb von Poool hat sich für Sie als Nutzer nichts verändert. Um eine Rechnung zu erstellen, gehen Sie wie gewohnt vor und wählen bei dem Rechnungsformat das gewünschte Format aus. Nach dem abspeichern zeigt Ihnen ein Pop-up Fenster das generierte Dokument an, welches Sie daraufhin herunterladen oder versenden können.

![zugferd-und-xrechnung-rechnungserstellung.png](Digitale%20Rechnungen%20mit%20ZUGFeRD%20&%20XRechnung/zugferd-und-xrechnung-rechnungserstellung.png)

## **Dateien in XRechnungen einbetten**

Beim Format XRechnung können außerdem weitere Dokumente wie Stundenreports, Belegte, etc. eingebettet werden. 

**Was sind die Vorteile?**

- Volldigitale Übergabe einschließlich Nachweise und Reports
- Einbettung direkt in die XRechnung, also kein separates Handling von Dateien
- Rechtlich konform und automatisiert im Export enthalten

**Schritt-für-Schritt-Anleitung**

1. Erstellen Sie eine Ausgangsrechnung im Format der XRechnung.
2. Unter **Dokument** erscheint nun der Bereich **„Zusätzliche Dateien“** (*nur verfügbar bei XRechnung).*
3. Fügen Sie hinzu:
    - **+ Dokument** – für interne Reports (z. B. Stundenreport) Wählen Sie hier  die passende Dokumentenvorlage aus (z. B. „Stundenreport“).
    - **+ Datei** – für manuelle Uploads (z. B. extern erzeugte PDF)
4. Klicken Sie auf **„Dokument aktualisieren“**, um die Anhänge einzubetten.
5. Im XRechnung-Export (XML) erscheinen die Dateien nun base64-kodiert im Anhang.

![image.png](Digitale%20Rechnungen%20mit%20ZUGFeRD%20&%20XRechnung/image%201.png)

## ✅ **Checkliste: XRechnung**

### 🔧 **Vorbereitung & Systemkonfiguration**

- [ ]  Eigene Bankverbindung im CRM korrekt gepflegt *(mit Markierung als Hauptkonto für XRechnung/ZUGFeRD)*
- [ ]  Empfänger enthält gültige Leitweg-ID

### 📄 **Rechnung erstellen**

- [ ]  Dokumentformat „XRechnung“ ausgewählt
- [ ]  Pflichtfelder befüllt:
    - [ ]  Leitweg-ID (Empfängeradresse)
    - [ ]  Projektreferenz (falls gefordert)
    - [ ]  Leistungszeitraum im Textfeld oder Positionsfeld
- [ ]  Korrekte Steuersätze (inkl. Steuerlogik für Inland, EU, Drittland)

---

## ✅ **Checkliste: ZUGFeRD**

### 🔧 **System-Setup**

- [ ]  ZUGFeRD-Einheiten gepflegt (inkl. Code je Einheit)
- [ ]  Bankverbindung (CRM) als ZUGFeRD-kompatibel markiert
- [ ]  Dokumentenvorlage mit ZUGFeRD-konformer Konfiguration aktiv

### 📄 **Rechnung erstellen**

- [ ]  Dokumenttyp: Standard-Rechnung oder ZUGFeRD-kompatible Vorlage
- [ ]  Alle Pflichtdaten korrekt gepflegt:
    - [ ]  Empfängeradresse
    - [ ]  Steuersätze korrekt
    - [ ]  Rechnungsdatum, Nummer, Betrag, Leistungszeitraum