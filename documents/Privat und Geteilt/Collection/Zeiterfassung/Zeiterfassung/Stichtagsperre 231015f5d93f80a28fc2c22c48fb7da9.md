# Stichtagsperre

Article Description: Zeiterfassung sichern mit einer definierbaren Sperre.
Last Updated: 27. Oktober 2025
Published: Yes
Suggested: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

Die **Stichtagsperre** ist ein zentrales Werkzeug in der Poool Zeiterfassung, um erfasste Stundenbuchungen zu schützen. Mit einer Stichtagsperre wird ein Datum festgelegt, ab dem **keine Zeiterfassungsbuchungen mehr verändert oder hinzugefügt werden können.** Das gilt für alle Rollen und Funktionen – unabhängig vom Freigabestatus.

*Beispiel: Wird der 30.06. als Stichtag gesetzt, können ab dem 01.07. keine Buchungen vor dem 30.06. mehr geändert oder ergänzt werden.*

## 🧑‍💼 **Wer darf eine Stichtagsperre setzen?**

Das Recht, eine Stichtagsperre zu setzen, ist an ein **individuelles Rollenrecht** gebunden. Standardmäßig haben dies **nur ausgewählte administrative Rollen** wie z. B.:

- Controller
- Finanzverantwortliche
- Systemadministratoren

![image.png](Stichtagsperre/image.png)

<aside>
⚙

**Achtung**: Das Recht sollte nur gezielt vergeben werden, da die Sperre nicht rückgängig gemacht werden kann.

</aside>

## ⚠️ **Wichtige Hinweise vor dem Setzen einer Sperre**

Da die Stichtagsperre **endgültig** ist, empfehlen wir dringend, folgende Schritte im Vorfeld zu beachten:

1. **Interne Kommunikation:** Informieren Sie alle Zeiterfasser und Projektmanager frühzeitig über das geplante Stichtagsdatum.
2. **Pufferzeit einplanen:** Planen Sie genügend Zeit ein, damit das Team offene Buchungen prüfen und korrigieren kann.
3. **Letzte Kontrolle:** Stellen Sie sicher, dass alle relevanten Korrekturen, Freigaben und Bearbeitungen abgeschlossen sind.

## 🔧 **So funktioniert das Setzen der Stichtagsperre**

1. Öffnen Sie den Bereich `Unternehmen → Stichtagsperren`.
2. Fügen Sie eine `+ Stichtagsperre` hinzu und wählen Sie das gewünschte Datum aus, bis zu dem die Zeiterfassung gesperrt werden soll.
3. Bestätigen Sie die Auswahl. Ab diesem Zeitraum ist die Zeiterfassung **endgültig geschützt**.

![image.png](Stichtagsperre/image%201.png)

<aside>
⚠️

**Wichtig**: Eine gesetzte Stichtagsperre kann nicht aufgehoben oder verschoben werden. Überlegen Sie also gut, wann sie gesetzt wird.

</aside>