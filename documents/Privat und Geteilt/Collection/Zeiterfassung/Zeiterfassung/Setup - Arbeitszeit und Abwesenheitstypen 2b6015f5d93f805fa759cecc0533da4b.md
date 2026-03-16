# Setup - Arbeitszeit und Abwesenheitstypen

Article Description: Arbeitszeiten und Abwesenheiten definieren verwalten.
Published: Yes
Suggested: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

Mit den Zeiterfassungstypen in Poool können Sie Arbeitszeit und Abwesenheiten individuell gestalten und an Ihre internen Prozesse anpassen. Ob Urlaub, Krankenstand oder Gleitzeit – jeder Typ lässt sich flexibel einrichten und mit Berechtigungen oder Genehmigungen kombinieren.

## ✅ Systemvoraussetzungen & Einstellungen

- Zugriff auf **`Einstellungen → Zeiterfassung`**
- Optional: Berechtigungen für Freigabeprozesse `System → Freigabeprozesse`

## 📌 Prozessbeschreibung: Schritt-für-Schritt Anleitung

- **Schritt 1:** Öffnen Sie **`Einstellungen → Zeiterfassung → Zeiterfassungstypen`**
    
    Hier finden Sie alle bestehenden Typen und können neue hinzufügen oder bestehende bearbeiten.
    
    ![image.png](Setup%20-%20Arbeitszeit%20und%20Abwesenheitstypen/image.png)
    

- **Schritt 2:** Klicken Sie auf **`+ Zeiterfassungstyp**`
    - Vergeben Sie Titel, Abkürzung, Farbe und Symbol.
    - Wählen Sie einen passenden **Basistyp** (z. B. Normalarbeitszeit, Urlaub, Krankenstand).
    - Definieren Sie, ob der Typ als **Standard** für die Zeiterfassung gelten soll (nur einer möglich, also meistens Arbeitszeit).
    - Legen Sie fest, ob dieser Typ für ganztägige, halbtägige und untertägige Abwesenheiten verwendet werden darf.
        
        ![image.png](Setup%20-%20Arbeitszeit%20und%20Abwesenheitstypen/image%201.png)
        

- **Schritt 3:** Legen Sie fest, ob dieser Arbeits- oder Abwesenheitstyp auch als **Arbeitsort** (Büro, Home-Office, Dienstreise) verwendet werden darf.
- **Schritt 4:**  Legen Sie fest, **ob ein Antrag erforderlich** ist – inklusive Optionen für Genehmigung, Anhang oder E-Mail-Benachrichtigung.
- Schritt 5: Legen Sie fest, wer Abwesenheitsänträge freigeben darf. → [Abwesenheitsanträge verwalten](../Abwesenheiten/Abwesenheitsantr%C3%A4ge%20verwalten%202a9015f5d93f80989084edd955163942.md)

## 💼 Standardeinstellungen gängiger Zeiterfassungstypen

Hier ein Auszug gängiger Zeiterfassungstypen als Orientierung und Starthilfe. 

| **Zeiterfassungstyp** | **Basistyp** | **Verfügbarkeit** | **Arbeitsort verwendbar** | **Antrag / Genehmigung erforderlich** | **Beschreibung / Hinweise** |
| --- | --- | --- | --- | --- | --- |
| **Arbeitszeit / Büro** | Normalarbeitszeit | untertägig, halbtägig, ganztägig | ✅ Ja | ❌ Nein | Standardtyp für reguläre Arbeitszeiten im Büro. Keine Genehmigung notwendig. |
| **Gleitzeit** | Ruhezeit | untertägig, halbtägig, ganztägig | ❌ Nein | 🔁 Optional (abhängig von Unternehmenspolitik) | Für Zeitausgleich oder flexible Arbeitszeiten. Kann Antragspflicht haben. |
| **Urlaub** | Urlaub | halbtägig, ganztägig | ✅ Ja | ❌ Nein | Abwesenheitstyp für Urlaubstage. Wird vom Urlaubskontingent abgezogen. |
| **Fortbildung** | Normalarbeitszeit | halbtägig, ganztägig | ✅ Ja | ❌ Nein | Für Schulungen, Weiterbildungen oder Seminare während der Arbeitszeit. |