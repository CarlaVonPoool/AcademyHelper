# Spesenabrechung Mitarbeiterlieferanten-Funktion

Article Description: So rechnen Sie Spesen korrekt über individuelle Mitarbeiterkonten ab – automatisiert und DATEV-kompatibel.
Published: Yes
Suggested: No

Mit dieser Funktion können Sie Spesen korrekt und automatisiert über das **individuelle Kreditorenkonto eines Mitarbeiters oder einer Mitarbeiterin** abrechnen. Das Konto wird bei der **Eingangsrechnung**, im **Buchungsexport** und in der **Mitarbeiterverwaltung** automatisch berücksichtigt. Damit vermeiden Sie manuelle Umbuchungen in DATEV oder anderen Systemen.

## ✨ Vorteile

- **Korrekte Buchung auf Mitarbeiterebene:** Das Mitarbeiterkonto wird automatisch in die Buchung übernommen, Mitarbeiter:innen müssen nicht mehr als Lieferanten angelegt werden.
- **Nahtlose Integration:** Die Daten fließen von der Eingangsrechnung über das Mitarbeiterprofil bis zum Buchungsexport.

## So geht’s

### 1. Kreditorenkonto beim Mitarbeiter hinterlegen

Gehen Sie zu `Unternehmen **→** Mitarbeiter`

- Öffnen Sie den jeweiligen Mitarbeitereintrag
- Tragen Sie das **Kreditorenkonto** im Feld „Mitarbeiterkonto“ ein (z. B. `KREDITOR-4845`)
- Speichern Sie die Änderungen

> 💡 Dieses Konto wird künftig automatisch bei Spesenabrechnungen verwendet, wenn der Mitarbeitende als Kontakt gesetzt wird.
> 

![image.png](Spesenabrechung%20Mitarbeiterlieferanten-Funktion/image.png)

### 2. Spesen-Eingangsrechnung erstellen

Nachdem eine Spesenabrechnung erstellt und genehmigt wurde, erscheint diese unter `Buchhaltung → Eingangsrechnungen` und kann hier erfasst und bearbeitet werden. 

- Als Lieferant wird Ihr Unternehmen ausgegeben, als Kontakt können Sie nun den jeweiligen Mitarbeiter auswählen.
- Füllen Sie die üblichen Felder der Eingangsrechnung aus (Betrag, Konto, Buchungstext etc.).

![Screenshot 2025-07-21 144651.png](Spesenabrechung%20Mitarbeiterlieferanten-Funktion/Screenshot_2025-07-21_144651.png)

### 3. Buchungsexport konfigurieren

Administratoren können unter `System-Buchhaltungsexporte` festlegen, dass für Mitarbeitende das **Mitarbeiterkonto** verwendet wird – anstelle des Standard-Kontos.

- Öffnen Sie die Konfiguration Ihres Buchungsexports (z. B. DATEV)
- Aktivieren Sie die Option:
Lieferant: Mitarbeiterkonto oder Buchhaltungskonto

![Screenshot 2025-07-21 163854.png](Spesenabrechung%20Mitarbeiterlieferanten-Funktion/Screenshot_2025-07-21_163854.png)

### Verwendung im DATEV-Export

Wenn das Mitarbeiterkonto gepflegt ist und die Konfiguration aktiv ist, wird dieses Konto im **DATEV XML-Export** verwendet.

### Beispiel: Spesenabrechnung von „Alfred“

```xml
xml
KopierenBearbeiten
<accountsPayableLedger>
  <date>2025-07-21</date>
  <amount>1.20</amount>
  <accountNo>4050</accountNo>
  <currencyCode>EUR</currencyCode>
  <invoiceId>-</invoiceId>
  <bookingText>Spese von Alfred</bookingText>
  <exchangeRate>1.000000</exchangeRate>
  <bpAccountNo>KREDITOR-4845</bpAccountNo>
  <supplierName>Unser Alfred</supplierName>
</accountsPayableLedger>

```

> 💡 Wichtig: `bpAccountNo` enthält das **Mitarbeiterkonto**, nicht das Lieferantenkonto
>