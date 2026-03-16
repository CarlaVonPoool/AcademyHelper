# Guide - Erste Schritte zur Zeiterfassung

Article Description: In diesem Guide lernen Sie, wie Sie die Zeiterfassung in Poool richtig aufsetzen.
Last Updated: 26. November 2025
Published: Yes
Suggested: No

In diesem Guide lernen Sie, wie Sie die Zeiterfassung in Poool richtig aufsetzen – Schritt für Schritt und mit Hintergrundinformationen zu den wichtigsten Entscheidungen.

**In diesem Video finden Sie einmal eine Übersicht zur allgemeinen Struktur des Guides:**

[https://vimeo.com/1167387717/4b82160cf3?share=copy&fl=sv&fe=ci](https://vimeo.com/1167387717/4b82160cf3?share=copy&fl=sv&fe=ci)

## 🏁 Zielsetzung

Nach diesem Guide haben Sie : 

- [ ]  Alle Mitarbeitenden in Poool angelegt und eingeladen
- [ ]  Arbeitszeitmodelle und Regeln definiert
- [ ]  Funktionen und Preislisten für Projektabrechnung hinterlegt
- [ ]  Ihr Team geschult und startklar gemacht

⌛ **Zeitaufwand**: ca. 2–3 Stunden (je nach Teamgröße)
****

---

## ☁️ Vor dem Start: Wichtige Entscheidungen

Bevor Sie mit dem Setup beginnen, sollten Sie folgende Fragen für sich klären. Das spart später Zeit und Umwege.

**Mitarbeiterverwaltung:**

- [ ]  Wie legen wir Mitarbeitende an? (SSO / Personio / Manuell)
- [ ]  Wie melden sich Mitarbeitende an? (Microsoft / Google / E-Mail+Passwort)
- [ ]  Wer ist für Stammdatenpflege verantwortlich?

**Arbeitszeit:**

- [ ]  Arbeiten wir mit Gleitzeit ?
- [ ]  Welche Regeln gelten? (Höchstarbeitszeit, Pausenregelung, Nachtarbeit)
- [ ]  Welche Abwesenheitstypen gibt es? (Urlaub, Krankheit, Homeoffice, etc.)
- [ ]  Müssen Abwesenheiten genehmigt werden?

**Projektzeit:**

- [ ]  Wie detailliert erfassen wir Projektzeit? (Projekt / Phasen / Tickets)
- [ ]  Wer darf auf welche Projekte buchen?
- [ ]  Arbeiten wir mit unterschiedlichen Stundensätzen pro Tätigkeit?

---

## 1️⃣ Mitarbeiter anlegen und einladen

Damit die Zeiterfassung funktioniert, müssen zunächst alle Mitarbeitenden in Poool vorhanden sein. Es gibt drei Wege, Mitarbeitende anzulegen. Wählen Sie die Methode, die am besten zu Ihrer IT-Infrastruktur passt.

[https://vimeo.com/1168036436?share=copy&fl=sv&fe=ci](https://vimeo.com/1168036436?share=copy&fl=sv&fe=ci)

**📌 Alle 3 Varianten im Überblick**

- V1: **Registrierung via Single Sign-On (SSO) →** [Microsoft Single-Sign on einrichten](https://academy.poool.cc/unternehmen--mitarbeiter/byhcLND247SzyAdzNxFpku/microsoft-single-sign-on-einrichten/ssk1neACkq8kVEDu62HMi4)
    
    Wenn Ihre Organisation SSO (z. B. über Microsoft Entra ID oder Google Workspace) nutzt, können Mitarbeitende sich **selbst registrieren**, sobald sie sich das erste Mal über den Unternehmens-Login anmelden.
    
    **Vorteile:**
    
    - kein manueller Aufwand für Anlage
    - Benutzerverwaltung bleibt zentral
    - sicheres Anmeldeverfahren
- V2: **Anlage über Personio-Schnittstelle →** Kontaktiere den [Poool-Support](mailto:support@poool.cc)
    
    Poool importiert alle relevanten Personaldaten (Name, E-Mail, Arbeitszeitmodell, Abteilung etc.) automatisch aus Personio.
    
    **Vorteile:**
    
    - keine doppelte Datenpflege
    - Änderungen (Eintritt, Austritt, Namensänderung etc.) werden automatisch übernommen
    
    **Voraussetzung:**
    
    - Personio-Integration muss aktiv und richtig konfiguriert sein
- V3: **Manuelle Anlage in Poool →** [Mitarbeiter anlegen](https://academy.poool.cc/unternehmen--mitarbeiter/byhcLND247SzyAdzNxFpku/mitarbeiter-anlegen/nX8FPVJadVVu2YQpo9rfe8)
    
    Sie können Mitarbeitende jederzeit direkt in Poool anlegen und über einen Einladungslink einladen.
    
    **Vorteile:**
    
    - volle Flexibilität
    - kein Setup von Schnittstellen notwendig
    
    **Nachteile:**
    
    - manueller Pflegeaufwand

**✅ Erledigt, wenn:**

- [ ]  Alle aktiven Mitarbeitenden sind in Poool angelegt
- [ ]  Login-Methode ist festgelegt und kommuniziert
- [ ]  Mindestens ein Testlogin wurde erfolgreich durchgeführt

**➡️ Nächster Schritt:** Jetzt definieren wir, wie Arbeitszeit erfasst werden soll.

---

## 2️⃣ Zeitmodelle und Regeln

Arbeitszeitmodelle legen fest, wie viele Stunden ein Mitarbeitender arbeiten soll (Sollarbeitszeit) und ob Gleitzeit möglich ist.

[https://vimeo.com/1169496693?share=copy&fl=sv&fe=ci](https://vimeo.com/1169496693?share=copy&fl=sv&fe=ci)

**📌 Los geht’s**

- **Schritt 1:** Zeitmodelle bei Mitarbeitenden hinterlegen anlegen → [Mitarbeiter anlegen](https://academy.poool.cc/unternehmen--mitarbeiter/byhcLND247SzyAdzNxFpku/mitarbeiter-anlegen/nX8FPVJadVVu2YQpo9rfe8#schritt-3-mitarbeiter-anlegen)
Hinterlegen Sie bei allen Mitarbeitenden Ihre Sollarbeitszeiten pro Woche und legen Sie fest, ob [Gleitzeitkonten](Gleitzeit%202b5015f5d93f80a3bf2cf54df5b43810.md) geführt werden.
- **Schritt 2**: Zeiterfassungstypen und Abwesenheiten anlegen → [Setup Arbeitszeiterfassung](https://academy.poool.cc/zeiterfassung/r3wCMHR3jbgMdC1LkfbPck/setup-neue-zeiterfassung/eRpCqiw9NSTYeAw4X8BDoS)
Definieren Sie welche Arten von Zeiterfassungstypen (Büro, Homeoffice) und Abwesenheiten (Urlaub, Krankenstand, Zeitausgleich, …) zur Verfügung stehen sollen, und welche davon genehmigt werden müssen.
- **Schritt 3**: Regeln für die Zeiterfassung festlegen → [Setup Regeln Zeiterfassung](Setup%20-%20Regeln%20f%C3%BCr%20die%20Arbeitszeiterfassung%202b6015f5d93f804686abf3b8e9a9f1e7.md) 
Legen Sie fest, welche Validierungsregeln für die Zeiterfassung gelten (Höchstarbeitzseit, Pausenregelungen, Nacharbeitbeschränkung).
- **Schritt 4**: Prozess für Genehmigungen einrichten → [Abwesenheitsanträge erstellen](https://academy.poool.cc/zeiterfassung/r3wCMHR3jbgMdC1LkfbPck/abwesenheitsantr%C3%A4ge/bE6TM7RZYojvNqztxFYh93) 
Definieren Sie, wer welche Abwesenheiten genehmigt und wie der Freigabeprozess abläuft.

**✅ Erledigt, wenn:**

- [ ]  Die Sollarbeitszeit aller Mitarbeitenden hinterlegt ist
- [ ]  Zeiterfassungsregeln sind aktiviert (Höchstarbeitszeit, Pausen etc.)
- [ ]  Alle benötigten Abwesenheitstypen sind angelegt
- [ ]  Genehmigungsprozesse sind definiert und getestet

**➡️ Nächster Schritt:** Jetzt richten wir Funktionen und Preislisten für Projektabrechnungen ein.

---

## 3️⃣  Funktionen & Preise

Funktionen und Preislisten legen fest, mit welchem **Stundensatz** eine gebuchte Zeit bewertet wird. Sie sind die Grundlage für Projektauswertungen, Kalkulation und Rechnungsstellung.

[https://vimeo.com/1169585999?share=copy&fl=sv&fe=ci](https://vimeo.com/1169585999?share=copy&fl=sv&fe=ci)

**📌 Los geht’s**

- **Schritt 1:** Anlegen von Funktionen und Preislisten → [Funktionen & Preisliste](https://academy.poool.cc/setup--konfiguration/2opHDG95rj4iYqTzeQcCpe/preislisten/7gYWu5fpVurHqvWWaeaLa4) 
Richten Sie Funktionen (Design, Administration, Konzept) ein und definieren Sie die internen und externen Preise im Standard oder kundenspezifisch.
- **Schritt 2**: Hinterlegen der Funktion bei den Mitarbeitenden → [Mitarbeiter anlegen](https://academy.poool.cc/unternehmen--mitarbeiter/byhcLND247SzyAdzNxFpku/mitarbeiter-anlegen/nX8FPVJadVVu2YQpo9rfe8)

**✅ Erledigt, wenn:**

- [ ]  Alle relevanten Funktionen sind angelegt
- [ ]  Standard-Stundensätze sind hinterlegt
- [ ]  Alle Mitarbeitenden haben mindestens eine Standardfunktion zugewiesen
- [ ]  (Optional) Kundenspezifische Preislisten sind erstellt

**➡️ Nächster Schritt:** Jetzt bereiten wir die Projekte für die Zeitbuchung vor.

---

## 4️⃣ Projekte einrichten

Jetzt definieren Sie, wie detailliert Projektzeit erfasst werden soll und wer auf welche Projekte buchen darf. Alle Grundlagen zur Anlage eines Projektes finden Sie hier → [Ein neues Projekt anlegen](https://academy.poool.cc/projektmanagement/jnxo2NXfdDf2VUGGajFjNu/neues-projekt/rbJmw5JDux5u1PYgLr8NTS)  

[https://vimeo.com/1170283701?share=copy&fl=sv&fe=ci](https://vimeo.com/1170283701?share=copy&fl=sv&fe=ci)

**📌 Los geht’s**

- **Schritt 1:** Anlegen von Projekten → [Ein neues Projekt anlegen](https://academy.poool.cc/projektmanagement/jnxo2NXfdDf2VUGGajFjNu/neues-projekt/rbJmw5JDux5u1PYgLr8NTS)  
Beim Anlegen eines Projektes können Sie entscheiden, ob Mitarbeitenden Projektzeiten auf Projektebene, in Phasen oder auf Tickets buchen können.

| **Variante** | **Typische Anwendungsfälle** | **Einstellung in Poool** |
| --- | --- | --- |
| **Projektzeit auf Projektebene** | - Kleine, einfache Projekte
- Interne Tätigkeiten („Vertrieb“, „Marketing“, „Support“)
- Allgemeine Verwaltung | Standardeinstellung |
| **Projektzeit in Phasen** | - Größere Projekte mit klaren Arbeitsschritten
- Beratungs- / Implementierungsprojekte
- Projekte mit Meilensteinen | Unter `Projekt → Phasen` anlegen |
| **Projektzeit auf Tickets** | - Sehr detaillierte Aufgaben
- IT-Projekte, Softwareentwicklung- Support / Service Tickets
- Projekte mit vielen Arbeitspaketen | Tickets im Projekt anlegen und (optional) Ticketfunktionen zuweisen. 
 |

- **Schritt 2:** Berechtigungen vergeben
Beim Erstellen des Projektes treffen Sie unter “Berechtigt für Zeiterfassung” die Entscheidung, wer Zeiten auf dieses Projekt buchen darf:
    - Alle Personen (offenes Projekt)
    - Teammitglieder
    Mitarbeitende aus bestimmten Teams dürfen Zeiten buchen.
    - Zugewiesene Personen
    Hier werden Mitarbeitenden, die auf dem Projekt zeit erfassen einzeln ausgewählt.

![imac.png](Guide%20-%20Erste%20Schritte%20zur%20Zeiterfassung/imac.png)

**✅ Erledigt, wenn:**

- [ ]  Mindestens ein Test-Projekt ist angelegt
- [ ]  Die Zeiterfassungs-Ebene für das Projekt ist definiert (Projekt / Phasen / Tickets)
- [ ]  Berechtigungen sind vergeben
- [ ]  Eine Test-Zeitbuchung wurde erfolgreich durchgeführt

**➡️ Nächster Schritt:** Jetzt schulen wir Ihr Team, damit alle wissen, wie die Zeiterfassung funktioniert.

---

## 5️⃣ Mitarbeiter schulen

Fast geschafft! Die technische Grundlage steht. Jetzt müssen Ihre Mitarbeitenden nur noch wissen, wie sie Poool im Alltag nutzen.

[https://vimeo.com/1170328964?share=copy&fl=sv&fe=ci](https://vimeo.com/1170328964?share=copy&fl=sv&fe=ci)

**📌 Los geht’s**

- **Schritt 1:** Wie erfasse ich Arbeitszeit → [Arbeitszeit erfassen](https://academy.poool.cc/zeiterfassung/r3wCMHR3jbgMdC1LkfbPck/arbeitszeiterfassung/dJRmDcQk7hmMzQqb9t2YaV)
Zeigen Sie Ihren Mitarbeitenden, wie sie ihre Arbeitszeit erfassen und welche Regeln dabei gelten.
- **Schritt 2**: Wie erfasse ich Projektzeit → [Projektzeit erfassen](https://academy.poool.cc/erste-schritte/wuxxfwixqv5R3fYVvZ9cr2/projektzeiten-erfassen/oWwKtRqRydjddKEwEiXoDD)
Zeigen Sie verschiedene Möglichkeiten zur Projektzeiterfassung: Stoppuhr, Mein Tag, Wochenübersicht.
Tipp → [Favoriten in der Projektzeiterfassung anlegen](https://academy.poool.cc/zeiterfassung/r3wCMHR3jbgMdC1LkfbPck/favoriten-in-der-projektzeiterfassung/tQwC4REN88RmftPYNRaVs2)
- **Schritt 3**: Wie beantrage ich Abwesenheiten
Zeigen Sie, welche Abwesenheitstypen beantragt werden können und welcher Genehmigungsprozess dahinter steht.

**✅ Erledigt, wenn:**

- [ ]  Alle Mitarbeitenden wurden eingeschult
- [ ]  Häufige Fragen wurden geklärt
- [ ]  Alle Mitarbeitenden haben mindestens einmal Zeit erfasst

---

## ✅ Checkliste: Setup abgeschlossen

Herzlichen Glückwunsch! Sie haben alle wichtigen Schritte abgeschlossen.

- [ ]  Alle Mitarbeitenden sind angelegt und können sich einloggen
- [ ]  Arbeitszeitmodelle sind zugewiesen
- [ ]  Zeiterfassungsregeln sind aktiviert
- [ ]  Abwesenheitstypen und Genehmigungsprozesse sind eingerichtet
- [ ]  Funktionen und Preislisten sind hinterlegt
- [ ]  Projekte sind angelegt und Berechtigungen vergeben
- [ ]  Team wurde geschult und hat erste Zeiten erfasst

---

## 🎉 Geschafft!

Ihre Zeiterfassung ist jetzt einsatzbereit. Viel Erfolg mit Poool!

---

## Weiterführende Links

[Arbeitszeiterfassung](Arbeitszeiterfassung%20672951a3d27b48c99ba3c22a7aaaa277.md)

[Projektzeiterfassung](Projektzeiterfassung%20490f3bfd76e34113a68da4a3a67c6675.md)

[Management der Mitarbeiterzeiterfassung](Management%20der%20Mitarbeiterzeiterfassung%200bf1e750ed4c47928743923ff7e90a80.md)

[Stundenmanagement](Stundenmanagement%20f601b175a546410998d5867f3d9cd0d4.md)

[Überstunden](%C3%9Cberstunden%201a3eb9448f5f43eeb2da0ed8c090532f.md)

[Gleitzeit- und Stundenkappung](Gleitzeit-%20und%20Stundenkappung%2025b015f5d93f808f832ee74e8944a413.md)

[Abwesenheiten beantragen](../Abwesenheiten/Abwesenheitsantr%C3%A4ge%20stellen%20564d480876544e75844444209ea9f3e2.md)

[Abwesenheiten verwalten](../Abwesenheiten/Abwesenheitsantr%C3%A4ge%20verwalten%202a9015f5d93f80989084edd955163942.md)