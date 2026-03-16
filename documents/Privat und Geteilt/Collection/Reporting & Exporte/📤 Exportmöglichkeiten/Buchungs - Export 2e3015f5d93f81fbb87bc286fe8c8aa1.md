# Buchungs - Export

Article Description: Export aller Buchungspositionen aus Eingangs- und Ausgangsrechnungen.
Last Updated: 14. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: Yes

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## 🎯 Was ist der Buchungs-Export?

Der Buchungs-Export liefert eine detaillierte Übersicht aller Buchungspositionen aus Eingangs- und Ausgangsrechnungen in einem gewählten Zeitraum. Im Gegensatz zu den Rechnungsexporten zeigt er nicht die Rechnungen selbst, sondern die einzelnen Buchungszeilen mit ihren Zuordnungen zu Konten, Kostenstellen und Budgets – ideal für die Buchhaltung und das Controlling.

## ✨ Vorteile

- **Positionsebene statt Rechnungsebene:** Jede einzelne Buchungsposition wird separat ausgewiesen – perfekt für detaillierte Auswertungen
- **Eingangs- und Ausgangsrechnungen kombiniert:** Ein Export für alle Buchungen, unabhängig von der Rechnungsart
- **Vollständige Kontenzuordnung:** Buchhaltungskonten für USt., Kunden, Lieferanten und Aufteilungen auf einen Blick
- **Flexibles Buchungsdatum:** Filterung nach dem tatsächlichen Buchungsdatum, nicht nach Rechnungsdatum

## 🔧 Export-Varianten

| Variante | Format | Inhalt | Typischer Anwendungsfall |
| --- | --- | --- | --- |
| Buchungen | XLSX | Strukturierte Datentabelle aller Buchungspositionen | Buchhaltung, Steuerberater, Controlling |
| Dokumente | ZIP | Belegbilder der zugehörigen Rechnungen | Archivierung, Belegnachweis |

## 🚀 So erstellst du den Export

1. Navigiere zu **Buchhaltung → Export**
2. Wähle den gewünschten **Zeitraum** (Filterung nach Buchungsdatum)
3. Klicke auf **Buchungen** für die XLSX-Datei oder auf **Dokumente** in der Buchungs-Zeile für das Beleg-Archiv

![image.png](Buchungs%20-%20Export/image.png)

<aside>
⚠️

 **Wichtiger Hinweis zum Datumsfilter**

- Der Zeitraumfilter bezieht sich auf das **Buchungsdatum** der einzelnen Positionen – nicht auf das Rechnungsdatum! Eine Rechnung vom 15.01.2026 kann Buchungen mit unterschiedlichen Buchungsdaten enthalten.
</aside>

<aside>
💡

**Unterschied zu Rechnungsexporten**

Der Buchungs-Export zeigt die einzelnen Positionen einer Rechnung. Hat eine Rechnung drei Buchungszeilen, erscheinen im Export drei Zeilen – jeweils mit den Rechnungsdaten als Kontext.

</aside>

## 💼 Anwendungsfälle

### Szenario 1: Übergabe an den Steuerberater

Du exportierst alle Buchungen eines Monats und übergibst die Datei an deinen Steuerberater. Die Spalten mit den Buchhaltungskonten (USt., Kunden, Lieferanten) ermöglichen eine direkte Verbuchung.

### Szenario 2: Kostenstellen-Auswertung

Du exportierst alle Buchungen und erstellst in Excel eine Pivot-Tabelle nach Kostenstellen. So siehst du auf einen Blick, welche Kosten und Erlöse pro Kostenstelle angefallen sind.

### Szenario 3: Budget-Controlling

Du filterst den Export in Excel nach der Spalte "Budget" und vergleichst die tatsächlichen Buchungen mit deinen Planwerten – aufgeschlüsselt nach Einnahmen und Ausgaben.

### Szenario 4: USt.-Abstimmung

Für die Umsatzsteuer-Voranmeldung benötigst du eine Aufstellung nach Steuersätzen. Die Spalten "Buchung Ust. Satz" und "Buchung Ust." liefern dir die nötigen Daten für die Abstimmung.

## 📋 Enthaltene Felder (XLSX)

### Allgemeine Informationen

| Feld | Beschreibung |
| --- | --- |
| Art | Art der Buchung (Eingangsrechnung/Ausgangsrechnung) |
| Kunde / Lieferant Nummer | Eindeutige Nummer des Kunden oder Lieferanten |
| Kunde / Lieferant | Name des Kunden oder Lieferanten |
| Projektnummer(n) | Zugeordnete Projektnummer(n) |

### Rechnungsbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Rechnung Absender | Name des Rechnungsabsenders |
| Rechnung Absender UID | UID-Nummer des Absenders |
| Rechnung Empfänger | Name des Rechnungsempfängers |
| Rechnung Empfänger UID | UID-Nummer des Empfängers |
| Rechnung Titel | Bezeichnung/Titel der Rechnung |
| Rechnungsnummer | Rechnungsnummer |
| Rechnung Poool Belegnummer | Interne Poool-Belegnummer |
| Rechnungsdatum | Datum der Rechnung |
| Rechnung Fälligkeit | Fälligkeitsdatum der Rechnung |
| Rechnung Status | Aktueller Status der Rechnung |
| Rechnung netto | Nettobetrag der Gesamtrechnung |
| Rechnung brutto | Bruttobetrag der Gesamtrechnung |

### Buchungsbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Buchungstitel | Bezeichnung der einzelnen Buchungsposition |
| Buchungsdatum | Datum der Buchung (Filterkriterium!) |
| Buchungsart | Art der Buchung (z.B. Erlös, Aufwand) |
| Buchungskonto | Zugeordnetes Buchungskonto |
| Buchung netto | Nettobetrag der Buchungsposition |
| Buchung Ust. Satz | Umsatzsteuersatz in Prozent |
| Buchung Ust. | Umsatzsteuerbetrag der Position |
| Buchung brutto | Bruttobetrag der Buchungsposition |

### Zuordnungen

| Feld | Beschreibung |
| --- | --- |
| Budget | Zugewiesenes Budget |
| Kostenstelle | Zugewiesene Kostenstelle |
| Aufteilung | Art der Kostenaufteilung (z.B. Fixkosten, Personal) |

### Buchhaltungskonten

| Feld | Beschreibung |
| --- | --- |
| Buchhaltungskonto Ust. | Konto für die Umsatzsteuer |
| Buchhaltungskonto Kunde | Debitorenkonto des Kunden |
| Buchhaltungskonto Lieferant | Kreditorenkonto des Lieferanten |
| Buchhaltungskonto Aufteilung | Konto für die Kostenaufteilung |