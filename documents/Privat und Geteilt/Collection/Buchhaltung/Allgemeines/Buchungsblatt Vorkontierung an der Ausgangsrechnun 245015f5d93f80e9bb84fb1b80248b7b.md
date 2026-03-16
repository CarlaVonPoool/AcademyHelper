# Buchungsblatt: Vorkontierung an der Ausgangsrechnung

Article Description: Das Buchungsblatt ermöglicht die präzise Vorkontierung von Ausgangsrechnungen – inklusive Erfolgskonten, Beträgen und Umsatzsteuer.
Published: Yes
Suggested: No

Mit dem **Buchungsblatt** können Sie an jeder Ausgangsrechnung (AR) eine oder mehrere **Buchungszeilen mit Erfolgskonten, Beträgen und Umsatzsteuern** erfassen. Damit wird eine präzise **Vorkontierung für die Buchhaltung** ermöglicht – insbesondere, wenn die Rechnung nicht direkt die Struktur der Buchung abbildet, z. B. bei Pauschalen oder bei Rechnungen, die auf mehrere Konten aufgeteilt werden müssen.

### ✅ Vorteile

- **Saubere Vorkontierung:** Weisen Sie einer Rechnung mehrere Erfolgskonten zu, unabhängig von der Struktur der Rechnungspositionen.
- **Trennung von Inhalt & Buchhaltung:** Projektmanager:innen erstellen Rechnungen kundenfreundlich – die Buchhaltung kontiert separat.
- **DATEV-kompatibel:** Die hinterlegten Buchungszeilen werden automatisch im Export berücksichtigt.

## 💼 Anwendungsfälle

### **Szenario 1: Pauschalrechnung mit interner Aufteilung**

Ein Projektmanager erstellt eine Pauschalrechnung über 10.000 €, die aus Leistungen verschiedener Teams besteht. Die Rechnung enthält nur eine Position, aber die Buchhaltung legt über das Buchungsblatt eine Vorkontierung auf drei verschiedene Erfolgskonten an.

### Szenario 2: Buchhaltung übernimmt die Kontierung

Die Projektleitung schreibt Rechnungen wie gewohnt – das Buchhaltungsteam ergänzt später das Buchungsblatt zur Vorkontierung, z. B. je nach Umsatzsteuerlogik oder Erlöskontenstruktur.

### 🚀 So funktioniert’s

Die Funktion Buchungsblatt muss initial aktiviert werden. Wenden Sie sich dazu gerne an den [Poool-Support](mailto:support@poool.cc). 

### Buchungsblatt öffnen

Öffnen Sie eine **Ausgangsrechnung** (Entwurf oder abgeschlossen).

Links neben den **Bill Sources und dem Abrechnungsassistent** sehen Sie das Symbol: 
**📄 Buchungsblatt**

- Klicken Sie auf das Symbol, um das Buchungsblatt zu öffnen.

![image.png](Buchungsblatt%20Vorkontierung%20an%20der%20Ausgangsrechnun/image.png)

- Es öffnet sich ein Modal mit einer Übersicht aller bisherigen Buchungszeilen.

![image.png](Buchungsblatt%20Vorkontierung%20an%20der%20Ausgangsrechnun/image%201.png)

### Neue Buchungszeilen hinzufügen

Im Modal können Sie beliebig viele **Buchungszeilen** erfassen. Jede Zeile besteht aus:

- **Erfolgskonto** (Pflichtfeld – freiwählbar, z. B. `4400`, `8400`, etc.)
- **Betrag** (Netto oder Brutto – je nach Steuerlogik)
- **Umsatzsteuer** (Standardsteuer oder angepasst je nach Vorgabe)

![image.png](Buchungsblatt%20Vorkontierung%20an%20der%20Ausgangsrechnun/image%202.png)

> ⚠️ Die im Buchungsblatt erfassten **Steuersummen** müssen mit der Rechnung übereinstimmen. Das System prüft automatisch, ob die Vorkontierung korrekt ist und meldet Abweichungen. Ohne vollständige oder gültige Kontierung kann die Rechnung nicht gespeichert werden.
> 

### Export

Wenn ein Buchungsblatt an einer Rechnung existiert, wird dieses beim **Buchhaltungsexport automatisch verwendet**:

- Die Daten werden in den **DATEV XML-Export** übernommen
- Jede Buchungszeile erscheint mit Konto, Betrag und Steuer im Exportfile.