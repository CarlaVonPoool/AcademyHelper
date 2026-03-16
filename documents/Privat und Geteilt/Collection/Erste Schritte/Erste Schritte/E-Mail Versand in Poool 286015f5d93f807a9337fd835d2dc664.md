# E-Mail Versand in Poool

Published: Yes
Suggested: No

## 📊 Artikel-Metadaten

| **Artikel-Typ** | Best Practice |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

# 📨Überblick der drei E-Mail Versandmethoden

In Poool können automatisch E-Mails versendet werden – etwa für **Rechnungen oder Angebote.**

Dafür stehen drei Versandarten zur Verfügung:

- **Microsoft 365** (Outlook / Exchange-Integration)
- **SMTP** (eigener Mailserver oder externer Anbieter)
- **Mailjet**

## **💼 Microsoft 365 (empfohlen)**

Diese Option wird genutzt, wenn E-Mails direkt über das eigene Microsoft-365-Postfach versendet werden sollen (z. B. über *rechnungen@firma.at*). Der Versand erfolgt dann über die Microsoft-Cloud, unter Verwendung eures Accounts.

**💪 Vorteile**

- Versand über das eigene Firmenkonto
- Mails erscheinen im eigenen Postfach in den gesendeten Elementen
- Erfüllt Compliance- und Datenschutzvorgaben
- Authentifiziert über Microsoft 365 – keine Drittanbieter
- Hohe Zustellrate durch Microsoft-Infrastruktur

🛠️ **Voraussetzungen**

- Aktiver Microsoft-365-Account
- App-Kennwort oder OAuth2-Verbindung (je nach Tenant-Richtlinie)
- Einrichtung durch euer IT-Team empfohlen

### **💻 Einrichtung in Poool**

- Unter  **`System > E-Mail`**  stellt ihr unter E-Mail Einstellungen auf die Versandmethode Microsoft 365 um

![image.png](E-Mail%20Versand%20in%20Poool/image.png)

- Oben rechts neben deinem Profilbild siehst du ein Zahnrad. Dort kannst du unter **`Kontoeinstellungen`** dein Microsoft Konto verknüpfen.

![image.png](E-Mail%20Versand%20in%20Poool/image%201.png)

**Einsatzempfehlung:**
👉 Wenn ihr Microsoft 365 im Unternehmen verwendet

👉 Wenn alle Systemmails aus eurem Unternehmenspostfach kommen sollen

👉 Wenn Compliance oder Nachvollziehbarkeit wichtig ist

## **⚙️ SMTP (eigener Server oder Drittanbieter)**

Mit der SMTP-Methode könnt ihr Poool an **jeden beliebigen Mailserver** anbinden – etwa Exchange, IONOS, Gmail oder einen dedizierten Mailrelay-Server. Damit bleibt die komplette Kontrolle bei euch.

**💪 Vorteile**

- Volle technische Kontrolle über den Versand
- Unabhängig von Poool oder Drittanbietern
- Ideal für Unternehmen mit eigener IT-Infrastruktur
- Funktioniert mit jedem SMTP-fähigen Mailserver

🛠️ **Voraussetzungen**

- SMTP-Zugangsdaten (Host, Port, Benutzername, Passwort)
- TLS/SSL-Verschlüsselung aktiv
- SPF-, DKIM- und DMARC-Einträge korrekt gesetzt (für hohe Zustellrate)

### **💻 Einrichtung in Poool**

- Unter  **`System > E-Mail`**  stellt ihr unter E-Mail Einstellungen auf die Versandmethode SMTP um und befüllst die anderen Zeilen mit deinen Serverdaten
- Über den Testmailversand kann geprüft werden, ob die Einrichtung geklappt hat

![image.png](E-Mail%20Versand%20in%20Poool/image%202.png)

**Einsatzempfehlung:**

👉 Wenn ihr euren Mailversand intern steuern wollt

👉 Wenn IT-seitig eigene Mailserver vorhanden sind

👉 Wenn ihr maximale Kontrolle über Versand und Konfiguration haben wollt
****

❌ **Achtung:** Poool prüft nur, ob die Verbindung zum Mailserver funktioniert. Die Zustellbarkeit der E-Mails (Spam-Filter, Reputation, Limitierungen) liegt in eurer Verantwortung und hängt von eurer Serverkonfiguration ab.

## **☁️ Mailjet**

Mailjet kann verwendet werden, wenn ihr keine eigene Mailinfrastruktur anbinden möchtet und auch keine Microsoft Anwendungen genutzt werden. Poool hat hierfür eine technische Standardintegration vorbereitet. Es können damit Mails versenden werden, allerdings übernimmt Poool keinerlei Verantwortung für den Versand, die Zustellbarkeit oder Authentifizierung.

**Das steckt dahinter:**

- Mailjet ist **eine externe Infrastruktur**, die Poool lediglich anbietet
- Wir **richten sie technisch ein**, aber
    - keine Garantie für Zustellbarkeit
    - keine Anpassung von SPF/DKIM/DMARC
    - keine Nachverfolgung oder Logprüfung
- Absender-Domains können abweichen oder generisch erscheinen

### **💻 Einrichtung in Poool**

Für die Einrichtung mit Mailjet, bitte kontaktieren Sie den Poool-Support (support@poool.cc) und wir kümmern uns um die Einrichtung. 

**Einsatzempfehlung:**
👉 Wenn temporär keine eigene Versandlösung verfügbar ist

## **💡 Welche Methode passt zu euch?**

**Ihr nutzt Microsoft 365 im Unternehmen?**

→ Wählt 💼 **Microsoft 365**. So laufen alle E-Mails über euer bestehendes Cloud-Postfach und ihr bleibt vollständig in eurer gewohnten Umgebung.

**Ihr habt einen eigenen Mailserver?**

→ Wählt ⚙️ **SMTP**. Damit habt ihr volle technische Kontrolle und könnt den Versand genau so konfigurieren, wie ihr ihn intern benötigt.

**Ihr habt weder Microsoft 365 noch eigenen Mailserver?**

→ Für den Start könnt ihr ☁️ **Mailjet** nutzen. Langfristig empfehlen wir aber eine der beiden anderen Optionen für professionellen Einsatz.

## 💡 Tipps & Hinweise

- Nach jeder Änderung kann mit **„Testmail versenden“** geprüft werden, ob der Versand funktioniert.
- Prüft regelmäßig, ob Mails beim Empfänger ankommen – insbesondere nach DNS-Änderungen oder Passwortänderung.
- Achtet darauf, dass SPF, DKIM und DMARC korrekt konfiguriert sind (gilt für alle Methoden).
- Wenn Poool das Mailjet-Konto verwaltet, werden technische Änderungen (DKIM, SPF, Bounce-Handling) automatisch aktualisiert.
- Bei Microsoft 365 oder SMTP trägt euer IT-Team die Verantwortung für Zustellbarkeit und Limitierungen.

---

## 🔍 **Zusammenfassung auf einen Blick**

| Methode | Kontrolle | Einrichtung | Zustellbarkeit | Empfehlung |
| --- | --- | --- | --- | --- |
| **Microsoft 365** | Hoch | Mittel | Hoch | ✅ Empfohlen für M365-Nutzer |
| **SMTP** | Sehr hoch | Erfordert IT | Hoch | ✅ Empfohlen für eigene Server |
| **Mailjet** | Gering | Sehr einfach | Variabel | Wenn keine eigene Versandlösung verfügbar ist. |