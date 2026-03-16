# Preislisten - Export

Article Description: Export der Preislisten aus den Einstellungen und am Projekt.
Last Updated: 14. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## 🎯 Was ist der Preislisten-Export?

Der Preislisten-Export ermöglicht Ihnen, Ihre Preislisten als Excel-Datei (.xlsx) herunterzuladen. Jede Funktion wird dabei als einzelne Zeile exportiert – mit allen relevanten Informationen zu Preisen, Einheiten und Steuersätzen.

**Besonderheit:** Sie haben die Wahl zwischen zwei Export-Varianten:

- **Preisliste intern:** Für internes Controlling – mit allen drei Stundensätzen (Intern, Extern, Kunde) und Aktiv-Status
- **Preisliste extern:** Für Kunden – nur der Kundenpreis, ohne interne Kalkulationsdaten

## ✨ Vorteile

- **Zwei Perspektiven:** Interne Version für Kalkulation, externe Version für Kunden
- **Vollständige Stammdaten:** Alle Funktionen mit Preisen, Einheiten und Beschreibungen
- **Sicherheit:** Externe Version enthält keine sensiblen Kalkulationsdaten
- **Flexibel:** Export aus Einstellungen (alle Preislisten) oder direkt am Projekt (aktuelle Preisliste)

## 🔧 Grundlegende Bedienung

### Export aus den Einstellungen

1. Navigieren Sie zu **Einstellungen → Preise & Funktionen → Preislisten**
2. Suchen Sie die gewünschte Preisliste in der Übersicht
3. Klicken Sie auf das **Download-Icon** für intern oder extern
4. Der Download startet automatisch

![image.png](Preislisten%20-%20Export/image.png)

### Export am Projekt

1. Öffnen Sie das gewünschte **Projekt**
2. Wechseln Sie zum Tab **Hauptprojekt**
3. Scrollen Sie zum Bereich **Stundenkosten** (unten)
4. Klicken Sie auf **→ Aktuelle Preisliste Kunde (xlsx)** oder **→ Aktuelle Preisliste intern (xlsx)**
5. Der Download startet automatisch

![image.png](Preislisten%20-%20Export/image%201.png)

<aside>
⚠️

**Wichtig:** Im Projekt wird immer die **aktuell gültige Preisliste** verwendet. Hat sich während der Projektlaufzeit die Preisliste geändert, werden die neuen Preise herangezogen – nicht die historischen zum Projektzeitpunkt.

</aside>

<aside>
💡

**Tipp:** Für historische Preise exportieren Sie die entsprechende Preisliste aus den Einstellungen (sofern dort noch vorhanden).

</aside>

## ⚙️ Export-Varianten im Vergleich

| Variante | Enthält | Nicht enthalten | Typischer Einsatz |
| --- | --- | --- | --- |
| **Preisliste intern** | Alle Funktionen (auch inaktive), alle 3 Stundensätze, Aktiv-Status, Interner Titel | – | Kalkulation, Controlling, Margenanalyse |
| **Preisliste extern** | Nur aktive Funktionen, nur Kundenpreis | Inaktive Funktionen, interne/externe Stundensätze, Aktiv-Status, Interner Titel | Kunden-Preisübersicht, Angebotsgrundlage |

## 💼 Anwendungsfälle

### Szenario 1: Margen einer Funktionsgruppe analysieren

Sie möchten prüfen, wie profitabel Ihre Leistungen kalkuliert sind.

**Vorgehen:** **Preisliste intern** exportieren → Spalten "Intern" und "Kunde" vergleichen → Differenz zeigt die Marge pro Leistung

### Szenario 2: Preisübersicht für Kunden erstellen

Ein Kunde fragt nach Ihren aktuellen Preisen.

**Vorgehen:** **Preisliste extern** exportieren → Enthält nur Kundenpreise ohne sensible Daten → Bei Bedarf formatieren und versenden

### Szenario 3: Stammkunden-Konditionen vergleichen

Sie möchten prüfen, welche Sonderkonditionen ein Stammkunde gegenüber den Standardpreisen hat.

**Vorgehen:** Standard-Preisliste und Stammkunden-Preisliste jeweils **intern** exportieren → Kundenpreise in Excel vergleichen

### Szenario 4: Inaktive Funktionen identifizieren

Sie möchten prüfen, welche Funktionen aktuell deaktiviert sind.

**Vorgehen:** **Preisliste intern** exportieren → Nach Spalte "Aktiv" filtern (Wert = "Nein")

## 📋 Exportierte Felder – Preisliste extern (8 Felder)

| Feld | Beschreibung |
| --- | --- |
| Funktionsgruppe | Gruppierung der Funktion (z.B. "Online", "Klassik") |
| Funktion | Technischer Name der Leistung |
| Standardtitel | Titel für Angebote und Rechnungen |
| Standardbeschreibung | Optionaler Beschreibungstext |
| Standardeinheit | Stunden, Tage, Pauschal, Stück, monatlich |
| Standardsteuer | Steuerschlüssel (z.B. "DE") |
| Standardsteuersatz | Steuersatz in Prozent |
| Preis netto | Kundenpreis (Verkaufspreis) |

## 📋 Exportierte Felder – Preisliste intern (12 Felder)

Der interne Export enthält alle Felder des externen Exports plus folgende Zusatzinformationen:

### Zusätzliche Felder

| Feld | Beschreibung |
| --- | --- |
| Aktiv | Ja/Nein – ist die Funktion aktuell nutzbar? |
| Interner Titel | Bezeichnung für interne Verwendung |
| Intern | Interner Stundensatz (echte Kosten) |
| Extern | Externer Stundensatz (Weitergabe an Partner) |
| Kunde | Kundenpreis (Verkaufspreis) – entspricht "Preis netto" im externen Export |

<aside>
💡

**Tipp für Controlling:** Die Differenz zwischen "Kunde" (was der Kunde zahlt) und "Intern" (echte Kosten) zeigt Ihnen die Marge pro Funktion.

</aside>

<aside>
⚠️

## Wichtige Hinweise

- **Berechtigung erforderlich:** Für den Export aus den Einstellungen benötigen Sie Admin-Rechte.
- **Nur aktive Funktionen extern:** Die externe Preisliste filtert inaktive Funktionen automatisch heraus.
- **Aktualität am Projekt:** Der Export am Projekt zeigt immer die aktuell gültige Preisliste – nicht die zum Projektstart.
- **Datenschutz:** Die interne Preisliste enthält sensible Kalkulationsdaten – nur für berechtigte Personen freigeben.
</aside>

## 📚 Verwandte Artikel

### Grundlagen

[Preislisten](../../Setup%20&%20Konfiguration/Poool%20Setup/Preislisten%2032d214225e994d0998a815ac98ed3d95.md)

### Weitere Exporte

[Exporte im Überblick](Exporte%20im%20%C3%9Cberblick%202b7015f5d93f801092a6ddc9d5488ab1.md)