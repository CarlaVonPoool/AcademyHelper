# Eingangsrechnungs - Export

Article Description: Export aller Eingangsrechnungen als Excel-Tabelle oder ZIP mit Belegdokumenten.
Last Updated: 14. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: Yes

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## 🎯 Was ist der Eingangsrechnungs-Export?

Der Eingangsrechnungs-Export ermöglicht den Download aller erfassten Eingangsrechnungen eines bestimmten Zeitraums – entweder als strukturierte Excel-Tabelle (XLSX) für Auswertungen oder als ZIP-Archiv mit den Belegdokumenten für die Ablage. Er liefert alle relevanten Lieferantendaten, Beträge, Zahlungsstatus und die USt.-Aufteilung.

## ✨ Vorteile

- **Vollständige Lieferantenübersicht:** Alle relevanten Stammdaten inkl. UID-Nummer und Kreditorenkonto auf einen Blick
- **Transparente Kostenstruktur:** Zuordnung zu Projekten, Budgets und Kostenstellen direkt ersichtlich
- **Detaillierte USt.-Aufteilung:** Automatische Aufschlüsselung nach Steuersätzen für die Buchhaltung
- **Zahlungsstatus im Überblick:** Offene, bezahlte und fällige Beträge sofort erkennbar

## 🔧 Export-Varianten

| Variante | Format | Inhalt | Typischer Anwendungsfall |
| --- | --- | --- | --- |
| Eingangsrechnungen | XLSX | Strukturierte Datentabelle mit allen Rechnungsfeldern | Auswertungen, Buchhaltung, Reporting |
| Dokumente | ZIP | Belegbilder der Eingangsrechnungen | Archivierung, Weitergabe an Steuerberater |

## 🚀 So erstellst du den Export

1. Navigiere zu **Buchhaltung → Export**
2. Wähle den gewünschten **Zeitraum** (Filterung nach Rechnungsdatum)
3. Optional: Grenze über **Eingangsrechnungsnummer von/bis** auf bestimmte Rechnungsnummern ein
4. Klicke auf **Eingangsrechnungen** für die XLSX-Datei oder auf **Dokumente** in der Eingangsrechnungs-Zeile für das Beleg-Archiv

<aside>
💡

Die Einschränkung auf einen Zeitraum wird durch die Angabe eines Rechnungsnummernbereichs deaktiviert.

</aside>

![image.png](Eingangsrechnungs%20-%20Export/image.png)

<aside>
⚠️

Wichtiger Hinweis zum Datumsfilter

Der Zeitraumfilter bezieht sich ausschließlich auf das **Rechnungsdatum** – nicht auf das Fälligkeitsdatum oder Buchungsdatum!

</aside>

<aside>
💡

## Weitere Hinweise

- Die Umsatzsteuer-Spalten werden dynamisch basierend auf den im System konfigurierten Steuersätzen generiert.
- Der Export enthält alle Eingangsrechnungen unabhängig vom Status (Offen, Bezahlt, etc.).
</aside>

## 💼 Anwendungsfälle

### Szenario 1: Monatliche Buchhaltung

Du exportierst alle Eingangsrechnungen eines Monats und übergibst die Datei an deine Buchhaltung oder deinen Steuerberater. Die USt.-Aufteilung ermöglicht eine direkte Zuordnung zu den entsprechenden Steuerkonten.

### Szenario 2: Projektkostenkontrolle

Du filterst den Export nach einem bestimmten Zeitraum und analysierst anschließend in Excel, welche Kosten welchen Projekten zugeordnet wurden. Die Spalte "Projektnummer(n)" ermöglicht eine einfache Pivot-Auswertung.

### Szenario 3: Offene-Posten-Analyse

Du exportierst alle Eingangsrechnungen und filterst in Excel nach Status "Offen" und dem Fälligkeitsdatum, um überfällige Zahlungen zu identifizieren.

### Szenario 4: Vorsteuer-Voranmeldung vorbereiten

Für die UStVA benötigst du alle Eingangsrechnungen eines Monats. Exportiere den relevanten Zeitraum und nutze die aufgeschlüsselten USt-Spalten für deine Meldung.

## 📋 Enthaltene Felder (XLSX)

### Lieferantenbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Lieferantennummer | Eindeutige Kennung des Lieferanten in Poool |
| Lieferant Konto | Kreditorenkonto aus der Buchhaltung |
| Lieferant | Name des Lieferanten |
| Lieferant UID | Umsatzsteuer-Identifikationsnummer des Lieferanten |

### Rechnungsbezogene Felder

| Feld | Beschreibung |
| --- | --- |
| Rechnungsnummer (Lieferant) | Rechnungsnummer vom Lieferanten |
| Rechnungsnummer (intern) | Interne Rechnungsnummer in Poool (z.B. ER-XXX-0001) |
| Eingangsrechnung | Bezeichnung/Titel der Eingangsrechnung |
| Rechnungsdatum | Datum der Rechnung (Filterkriterium!) |
| Fälligkeit | Fälligkeitsdatum der Zahlung |
| Status | Aktueller Status (Offen, Bezahlt, etc.) |
| Zahlungsmittel | Verwendetes Zahlungsmittel |
| Beleg | Angabe ob Belegbild vorhanden (ja/nein) |
| Tags | Zugewiesene Tags zur Kategorisierung |

### Zuordnungen

| Feld | Beschreibung |
| --- | --- |
| Kundennummer | Zugeordnete Kundennummer (falls vorhanden) |
| Projektnummer(n) | Zugeordnete Projektnummer(n) |
| Budget | Zugewiesenes Budget |
| Kostenstelle | Zugewiesene Kostenstelle |
| Aufteilung | Art der Kostenaufteilung (z.B. Fixkosten, Personal) |

### Beträge

| Feld | Beschreibung |
| --- | --- |
| Summe netto | Nettobetrag der Rechnung |
| Summe brutto | Bruttobetrag der Rechnung |
| Summe bezahlt | Bereits bezahlter Betrag |
| Summe Abzüge | Abgezogene Beträge (z.B. Skonto) |
| Summe offen | Noch offener Restbetrag |
| Ust. [Steuersatz] | Aufschlüsselung nach konfigurierten Steuersätzen (dynamisch) |