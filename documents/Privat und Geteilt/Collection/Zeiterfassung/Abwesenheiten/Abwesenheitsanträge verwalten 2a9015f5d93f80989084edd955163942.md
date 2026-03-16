# Abwesenheitsanträge verwalten

Published: Yes
Suggested: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## Abwesenheiten einrichten

In diesem Artikel erfahren Sie, wie Sie Abwesenheitsarten und Genehmigungsprozesse in Poool einrichten, verwalten und steuern.

Damit legen Sie fest, welche Abwesenheiten beantragt werden müssen, wer sie freigeben darf und wie Mitarbeitende informiert werden.

---

## ✅ Systemvoraussetzungen & Einstellungen

- Zugriff auf `Einstellungen → Zeiterfassung → Zeiterfassungstypen`
- Zugriff auf `System → Freigabeprozesse`
- Administrator- oder HR-Berechtigung
- Abwesenheitstypen sind korrekt angelegt
    - ➡️ Anleitung: [Abwesenheitsarten anlegen](https://academy.poool.cc/zeiterfassung/r3wCMHR3jbgMdC1LkfbPck/setup-neue-zeiterfassung/eRpCqiw9NSTYeAw4X8BDoS)
        - Navigieren Sie zu `Einstellungen → Zeiterfassung → Zeiterfassungstypen.`
            
            Hier legen Sie fest, welche Abwesenheitsarten im Unternehmen genutzt werden (z. B. Urlaub, Krankenstand, Sonderurlaub).
            
        - Unter dem Abschnitt **Antrag** stehen Ihnen folgende Optionen zur Verfügung:
            - **Antrag erforderlich** 
            Mitarbeitende müssen für diesen Typ eine Abwesenheit beantragen (z. B. Urlaub, Sonderurlaub).
            - **Genehmigung erforderlich**
            Der Antrag wird erst nach Freigabe durch eine berechtigte Person wirksam.
            - **Tag nicht festschreiben**
            Tage bleiben nach Eintrag editierbar.
            - **Anhänge**
            Aktivieren, wenn z. B. bei Krankenstand ein Dokument hochgeladen werden soll
            - **E-Mail-Benachrichtigung**
            Definiert, ob bei Antrag oder Genehmigung automatisch eine E-Mail den Verantwortlichen gesendet wird.
            
            💡*Tipp:* Bei Urlaub empfiehlt sich die Kombination **Antrag erforderlich + Genehmigung erforderlich + E-Mail-Benachrichtigung**. Bei Krankenstand kann zusätzlich **Anhänge erforderlich** aktiviert werden.
            
    

---

## 📌 3 Varianten -  Schritt-für-Schritt Anleitung

### **Variante 1: Zentrale Verantwortlichkeit für Abwesenheiten festlegen**

Diese Einstellung ist ideal, wenn Ihr Unternehmen zentral arbeitet – z. B. wenn alle Abwesenheiten über eine HR-Stelle oder eine Teamleitung laufen sollen.

Unter `Einstellungen → Zeiterfassung` können Sie im Bereich Abwesenheiten bestimmen, wer als verantwortliche Person E-Mails für Abwesenheiten erhält.

![Screenshot 2025-11-11 111903.png](Abwesenheitsantr%C3%A4ge%20verwalten/Screenshot_2025-11-11_111903.png)

### **Variante 2: Verantwortliche für Abwesenheiten pro Mitarbeiter definieren**

Sie können auch für jede Person im Unternehmen individuell festlegen, wer deren Abwesenheitsanträge prüfen oder genehmigen darf. Im `Unternehmen → Mitarbeiter → Mitarbeiterprofil` können Sie unter “Verantwortlich für Abwesenheiten” die jeweilige Person auswählen. 

### **Variante 3: Mehrstufige Freigabeprozesse einrichten (optional)**

Mit mehrstufigen Freigabeprozessen lassen sich **mehrere Genehmigungsschritte in fester Reihenfolge** abbilden.

- Navigieren Sie zu `System → Freigabeprozesse` und klicken Sie auf `+ Freigabeprozess`.
- Wählen Sie die **Art „Abwesenheitsantrag“** und bestimmen Sie, **für welche Abwesenheitsarten** dieser Prozess gilt.
- In der Konfiguration können Sie festlegen:
    
    
    | **Feld** | **Beschreibung** |
    | --- | --- |
    | **Titel / Beschreibung** | Frei wählbar, z. B. „Urlaubsfreigabe Marketing“ oder „Krankmeldungen HR“. |
    | **Verantwortliche Personen** | Einzelne Benutzer, die Anträge prüfen dürfen. |
    | **Verantwortliche Rollen** | Rollenbasiert, z. B. „Teamlead“, „HR-Manager“, „Projektleiter“. |
    | **Freigabe-Ersteller** | Optional: Person, die den Antrag ursprünglich gestellt hat. |
    | **Verantwortlich für Abwesenheiten** | Automatische Zuweisung der Abwesenheitsverantwortlichen aus dem Mitarbeiterprofil. |
    | **Fallback** | Ersatzperson, falls keine zuständige Person oder Rolle hinterlegt ist |

---

## 🔄 Abwesenheiten freigeben

Sobald Mitarbeitende einen Abwesenheitsantrag eingereicht haben, können berechtigte Personen diesen direkt über das Freigabe-Widget im Dashboard oder über `Unternehmen → Abwesenheiten` prüfen.

In der Übersicht werden standardmäßig alle offenen Anträge angezeigt. Über die Filterfunktionen können Sie gezielt nach Mitarbeiter, Zeitraum oder verantwortlicher Person suchen.

![image.png](Abwesenheitsantr%C3%A4ge%20verwalten/image.png)

Durch Klick auf einen Antrag öffnet sich die Detailansicht. Hier kann die Abwesenheit – je nach Berechtigung – genehmigt, abgelehnt oder bei Bedarf kommentiert werden.

⚠️**Hinweis:** Zur besseren Nachvollziehbarkeit können Abwesenheitsanträge nur storniert, nicht gelöscht werden. So bleibt die vollständige Dokumentation aller Anträge und Genehmigungen im System erhalten.

---

## 💼 Entscheidungshilfe: Welcher Freigabeprozess ist der richtige?

Nicht jede Organisation braucht komplexe Genehmigungen. Die folgende Übersicht hilft bei der Auswahl:

| **Anwendungsfall** | **Empfohlene Einstellung / Prozess** | **Einstellung in Poool** |
| --- | --- | --- |
| **Kleines Team mit flacher Struktur** | Einfache Freigabe über eine zentrale Person | Hinterlegen einer verantwortlichen Person für Abwesenheiten unter `Einstellungen → Zeiterfassung → Zeiterfassungstypen` |
| **Kleines Team mit Teamverantwortlichkeiten** | Verantwortliche Person pro Mitarbeiter definieren | Für jeden Mitarbeitenden eine zuständige Person im **Mitarbeiterprofil** unter `Unternehmen → Mitarbeiter` hinterlegen |
| **Mittelgroße Unternehmen** | Mehrstufiger Freigabeprozess: 1. Schritt Teamlead, 2. Schritt HR | Einrichten eines **Freigabeprozesses** unter `System → Freigabeprozesse`, Art „Abwesenheitsantrag“ |

💡 *Praxis-Tipp:* Beginnen Sie mit einfachen Prozessen und erweitern Sie diese bei Bedarf. Eine zu komplexe Genehmigungskette kann den Workflow unnötig verlangsamen.

---

## 🔍 **Zusammenfassung des Prozesses**

In Poool können Sie Abwesenheitsprozesse individuell gestalten – von der einfachen Freigabe bis zu mehrstufigen HR-Workflows. Über die **Zeiterfassungstypen** definieren Sie, welche Anträge erforderlich sind. Mit **Freigabeprozessen** bestimmen Sie, wer prüft und genehmigt. So behalten Sie jederzeit den Überblick über Urlaube, Krankenstände und Sonderfreistellungen.