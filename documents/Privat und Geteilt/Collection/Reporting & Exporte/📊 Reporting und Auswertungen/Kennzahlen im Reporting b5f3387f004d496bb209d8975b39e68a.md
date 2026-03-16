# Kennzahlen im Reporting

Article Description: Erklärung und Definition der Kennzahlen im Reporting.
Last Updated: 19. Januar 2026
Published: Yes
Suggested: No
Neues UI 2026: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀🚀 Praxis |

# Erklärung der Begrifflichkeiten

## **Abgerechnet**

Das Feld `Abgerechnet` zeigt Ihnen, wie viel Geld Sie dem Kunden **bereits in Rechnung gestellt bzw. fakturiert** haben.

💼 **Praxisbeispiel:**

Sie arbeiten an einem Webprojekt für einen Kunden. Im Laufe des Projekts haben Sie zwei Teilrechnungen geschrieben: eine über 2.000 € und eine über 3.500 €.

→ Der Wert im Feld `Abgerechnet` beträgt **5.500 €**.

⭐️ **Warum ist das wichtig?**

Diese Kennzahl zeigt Ihnen, welcher Teil Ihrer Arbeit bereits als Umsatz verbucht wurde – im Gegensatz zu noch offenen Aufträgen oder geplanten Leistungen.

💡 **Formel:** 

Summe aller erstellten Ausgangsrechnungen mit einer Rechnungsnummer (Entwürfe oder noch ausstehende wiederkehrende Rechnungen sind nicht enthalten)

## **Kosten**

Das Feld `Kosten` zeigt Ihnen alle **Fremdleistungskosten**, die für dieses Projekt angefallen sind – also alles, was Sie an externe Dienstleister oder Lieferanten bezahlen mussten.

💼 **Praxisbeispiel:**

Für ein Kundenprojekt haben Sie einen externen Fotografen für 800 € beauftragt und Stockfotos für 150 € gekauft. Beide Rechnungen wurden als Eingangsrechnungen erfasst.

→ Der Wert im Feld `Kosten` beträgt **950 €**.

⭐️ **Warum ist das wichtig?**

Nur wenn Sie Ihre Kosten kennen, wissen Sie, wie viel vom Umsatz tatsächlich bei Ihnen hängen bleibt. Je niedriger die Kosten bei gleichem Umsatz, desto profitabler ist Ihr Projekt.

<aside>
💡

 **Wichtig:** Es werden nur Eingangsrechnungen berücksichtigt (egal ob vorerfasst oder gebucht). Kosten aus dem Modul `Bestellungen` sind hier nicht enthalten.

</aside>

💡 **Formel:** 

Summe aller Eingangsrechnungen, die diesem Projekt zugeordnet sind

## **Rohertrag (DB I)**

Das Feld `Rohertrag` zeigt Ihnen, wie viel Geld nach Abzug aller Fremdkosten von Ihrem Umsatz **übrig bleibt**. In der Betriebswirtschaft wird dieser Wert auch als **Deckungsbeitrag I (DB I)** bezeichnet.

💼 **Praxisbeispiel:**

Sie haben für Ihr Webprojekt insgesamt 5.500 € abgerechnet. Dafür haben Sie einen externen Fotografen und Stockfotos im Wert von 950 € eingekauft.

→ Der `Rohertrag` beträgt **4.550 €** (5.500 € - 950 €).

→ In Prozent: **82,7 %** (4.550 € ÷ 5.500 €)

<aside>
⚙

**Hinweis:** Bei einigen Auswertungen in Poool wird der Rohertrag auch als Prozentwert angezeigt oder kann als solcher eingeblendet werden.

</aside>

⭐️ **Warum ist das wichtig?**

Der Rohertrag zeigt Ihnen, was Ihr Projekt wirklich "erwirtschaftet" hat – also der Betrag, der zur Deckung Ihrer eigenen Löhne, Miete und anderer Fixkosten zur Verfügung steht. Je höher der Rohertrag, desto profitabler ist das Projekt für Ihr Unternehmen.

💡 **Formel:** 

Abgerechnet (Umsatz) − Kosten (Fremdleistungen)

## **Soll Honorar**

Das Feld `Soll Honorar` zeigt Ihnen, **welcher Betrag in Rechnung gestellt werden müsste** – basierend auf allen Stunden, die auf das Projekt gebucht wurden, und den hinterlegten Stundensätzen (Kundenpreise).

<aside>
⚙

**Hinweis:** Das Soll Honorar ist immer die **Sicht des Kunden** – also was die geleistete Arbeit den Kunden kosten würde.

</aside>

💼 **Praxisbeispiel:**

Für Ihr Webprojekt haben zwei Mitarbeiter Zeiten gebucht:

- Designer: 20 Stunden à 95 €/Stunde = 1.900 €
- Entwickler: 35 Stunden à 120 €/Stunde = 4.200 €

→ Das `Soll Honorar` beträgt **6.100 €**.

⭐️ **Warum ist das wichtig?**

Das Soll Honorar zeigt Ihnen den "wahren Wert" Ihrer geleisteten Arbeit. Vergleichen Sie diesen Wert mit dem, was Sie tatsächlich abgerechnet haben – so erkennen Sie, ob Sie Ihre Zeit angemessen in Rechnung gestellt haben oder ob Leistungen "verschenkt" wurden.

💡 **Formel:** 

Summe aller gebuchten Stunden × jeweiliger Stundensatz (Kundenpreis)

## **Diff Soll-Ist (DB II)**

Das Feld `Diff Soll-Ist` zeigt Ihnen, **ob Sie mit dem Projekt Geld verdient oder draufgezahlt haben** – also was nach Abzug aller Kosten (Fremdkosten und interne Arbeitszeit) wirklich übrig bleibt. In der Betriebswirtschaft wird dieser Wert auch als **Deckungsbeitrag II (DB II)** bezeichnet.

💼 **Praxisbeispiel:**

Ihr Webprojekt hat folgende Werte:

- Rohertrag: 4.550 € (was nach Abzug der Fremdkosten übrig bleibt)
- Soll Honorar: 6.100 € (was Ihre interne Arbeitszeit den Kunden kosten würde)

→ Die `Diff Soll-Ist` beträgt **-1.550 €** (4.550 € - 6.100 €).

Das bedeutet: Sie haben mehr Arbeitszeit investiert, als Sie dem Kunden in Rechnung gestellt haben.

<aside>
⚠️

**Wichtig:** Bei der Auswertung ist zu unterscheiden, ob die Ausgangszahlen aus der **Kundensicht** (Kundenpreise) oder aus der **internen Kostensicht** (interne Stundensätze) stammen. Je nach Sichtweise muss die Kennzahl unterschiedlich bewertet oder nachberechnet werden – Ihr Unternehmen kann für beide Perspektiven eigene Schwellenwerte definieren.

</aside>

⭐️ **Warum ist das wichtig?**

Diese Kennzahl zeigt Ihnen die tatsächliche Wirtschaftlichkeit eines Projekts. Ein negativer Wert bedeutet, dass Sie Leistungen "verschenkt" haben. Ein positiver Wert zeigt, dass das Projekt effizienter abgewickelt wurde als kalkuliert.

<aside>
⚙

**Hinweis:** Bei einigen Auswertungen in Poool wird die Diff Soll-Ist auch als Prozentwert angezeigt oder kann als solcher eingeblendet werden.

</aside>

💡 **Formel:** 

Rohertrag − Soll Honorar

## **Stunden**

Das Feld `Stunden` zeigt Ihnen, **wie viel Arbeitszeit insgesamt auf das Projekt gebucht wurde**.

💼 **Praxisbeispiel:**

Für Ihr Webprojekt haben drei Mitarbeiter Zeiten erfasst:

- Designer: 20 Stunden
- Entwickler: 35 Stunden
- Projektleitung: 5 Stunden

→ Das Feld `Stunden` zeigt **60 Stunden**.

⭐️ **Warum ist das wichtig?**

Die gebuchten Stunden sind die Grundlage für viele weitere Kennzahlen wie das Soll Honorar. Außerdem können Sie so den tatsächlichen Aufwand mit Ihrer ursprünglichen Planung vergleichen – und für zukünftige Projekte besser kalkulieren.

💡 **Formel:** 

Summe aller gebuchten Stunden auf diesem Projekt

## **Auftrag**

Das Feld `Auftrag` zeigt Ihnen, **wie viel Geld noch offen ist** – also der Betrag aus Aufträgen, der noch nicht abgerechnet bzw. fakturiert wurde.

💼 **Praxisbeispiel:**

Sie haben mit Ihrem Kunden einen Auftrag über 10.000 € vereinbart. Bisher haben Sie zwei Teilrechnungen über insgesamt 5.500 € geschrieben.

→ Das Feld `Auftrag` zeigt **4.500 €** (der noch offene Betrag).

⭐️ **Warum ist das wichtig?**

Diese Kennzahl zeigt Ihnen, wie viel Umsatz aus bestehenden Aufträgen noch "in der Pipeline" ist – also Geld, das Sie noch in Rechnung stellen können.

💡 **Formel:** 

Summe aller offenen Aufträge (noch nicht fakturiert)

## **Auftrag + abgerechnet**

Das Feld `Auftrag + abgerechnet` zeigt Ihnen den **Gesamtwert des Projekts** – also alles, was Sie bereits in Rechnung gestellt haben, plus alles, was noch offen ist.

💼 **Praxisbeispiel:**

Ihr Webprojekt hat folgende Werte:

- Abgerechnet: 5.500 € (bereits fakturiert)
- Auftrag: 4.500 € (noch offen)

→ Das Feld `Auftrag + abgerechnet` zeigt **10.000 €**.

⭐️ **Warum ist das wichtig?**

Diese Kennzahl gibt Ihnen den vollständigen Überblick über das Projektvolumen – unabhängig davon, wie weit das Projekt fortgeschritten ist. So sehen Sie auf einen Blick, welchen Gesamtumsatz Sie mit diesem Projekt erwarten können.

💡 **Formel:** 

Abgerechnet + Auftrag (offene Aufträge)

## **Interne Stundenkosten**

Das Feld `Interne Stundenkosten` zeigt Ihnen, **was die geleistete Arbeitszeit Ihr Unternehmen tatsächlich kostet** – basierend auf den internen Stundensätzen Ihrer Mitarbeiter.

<aside>
⚙

**Hinweis:** Die internen Stundenkosten sind die **Sicht Ihres Unternehmens** – also was Sie die Arbeitszeit intern kostet. Im Gegensatz dazu zeigt das Soll Honorar die Kundensicht (was die Arbeit den Kunden kosten würde).

</aside>

💼 **Praxisbeispiel:**

Für Ihr Webprojekt haben zwei Mitarbeiter Zeiten gebucht:

- Designer: 20 Stunden × 45 € (interner Stundensatz) = 900 €
- Entwickler: 35 Stunden × 60 € (interner Stundensatz) = 2.100 €

→ Die `Internen Stundenkosten` betragen **3.000 €**.

⭐️ **Warum ist das wichtig?**

Mit dieser Kennzahl sehen Sie, was ein Projekt Ihr Unternehmen wirklich kostet. Vergleichen Sie die internen Stundenkosten mit dem Rohertrag – so erkennen Sie, ob ein Projekt aus interner Sicht profitabel ist.

💡 **Formel:** 

Summe aller gebuchten Stunden × jeweiliger interner Stundensatz

### ⚠️ Vor der Auswertung: Datenbasis prüfen

Bevor Sie Kennzahlen bewerten oder vergleichen, sollten Sie immer zuerst klären, **welche Datenbasis** den Zahlen zugrunde liegt. Je nach Einstellung können die gleichen Kennzahlen unterschiedliche Aussagen treffen.

## 🔍 **Stellen Sie sich folgende Fragen:**

**1. Sind die Daten vollständig und bereinigt?**

- Sind alle Stunden gebucht?
- Sind alle Eingangsrechnungen erfasst und zugeordnet?
- Sind offene Aufträge aktuell?
- Wurden fehlerhafte Buchungen korrigiert?

**2. Welche Sichtweise?**

- **Kundensicht:** Berechnung mit Kundenpreisen (Stundensätze, die Sie dem Kunden in Rechnung stellen)
- **Interne Sicht:** Berechnung mit internen Stundensätzen (was die Arbeitszeit Ihr Unternehmen kostet)

**3. Welcher Umsatz?**

- **Nur Abgerechnet:** Nur bereits fakturierte Beträge
- **Auftrag + Abgerechnet:** Gesamtes Projektvolumen inkl. offener Aufträge

**4. Welche Filter sind gesetzt?**

- Zeitraum (z.B. aktueller Monat, Quartal, Jahr)
- Projektstatus (laufend, abgeschlossen, alle)
- Kunde, Abteilung, Mitarbeiter, etc.

<aside>
💡

**Tipp:** Führen Sie vor jeder Auswertung eine kurze Datenprüfung durch und definieren Sie im Unternehmen einheitliche Standards, welche Datenbasis und Filter für welche Auswertung verwendet werden – so sind Ihre Kennzahlen vergleichbar und aussagekräftig.

</aside>