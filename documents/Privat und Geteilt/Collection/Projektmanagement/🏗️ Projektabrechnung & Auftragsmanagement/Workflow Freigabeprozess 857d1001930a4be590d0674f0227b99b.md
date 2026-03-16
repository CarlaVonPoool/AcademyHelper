# Workflow Freigabeprozess

Article Description: Über Freigabeprozesse stellen Sie sicher, dass alle Beteiligten informiert und Prozesse wie vorgesehen ablaufen können.
Last Updated: 6. Oktober 2023
Published: Yes
Suggested: No

In Poool gibt es die Möglichkeit Eingangsrechnungen, Ausgangsrechnungen, Aufträge und Angebote von gewissen Personen freigeben zu lassen.

## Freigabeprozesse einrichten

Damit die unterschiedlichen Freigabeprozesse überhaupt angewendet werden können, müssen die Freigabeprozesse zuerst definiert werden. Dazu navigieren Sie zu `System → Freigabeprozesse`. Mit dem Button `+ Freigabeprozess` können Sie einen neuen Freigabeprozess hinzufügen.

<aside>
ℹ️ Sollten Sie diesen Menüpunkt noch nicht in Ihrem Poool haben, kontaktieren Sie bitte den Support.

</aside>

Im ersten Schritt müssen Sie dem Freigabeprozess einen Namen geben und auswählen, ob der **Prozess aktiv ist oder nicht**. Sie können für den Freigabeprozess Werte festlegen. Dies bedeutet, dass er automatisch für jede Kalkulation **zwischen** diesen beiden **Werten** angewendet wird. 

Sie können hier auch entscheiden, für welchen Bereich der Freigabeprozess angewendet werden soll: Eingangsrechnungen, Ausgangsrechnungen, Aufträge, Angebote, Bestellungen und wiederkehrende Ausgangsrechnungen.  

![image.png](Workflow%20Freigabeprozess/image.png)

Im nächsten Schritt können Sie die unterschiedlichen Stufen des Freigabeprozesses definieren. Mit dem Button `+ Schritt` können Sie einen weiteren Schritt im Freigabeprozess hinzufügen. Sie können entscheiden, ob die Rechnung nur von **einer** oder **mehreren Personen** freigegeben werden muss. Für jeden dieser Schritte können Sie einen Titel und eine Beschreibung vergeben. 

Sie können hier auch entscheiden, ob die **Änderung der Daten** nach diesem Schritt **noch möglich** ist oder nicht. 

In diesem Schritt wird auch die **Verantwortliche Person oder Rolle** für die Freigabe des Dokuments sowie einen **Fallback** ausgewählt. Es können beliebig viele Schritte definiert werden.

![Untitled](Workflow%20Freigabeprozess/Untitled.png)

<aside>
💡 Werden mehrere *Verantwortliche Personen* bei einem Schritt hinterlegt, kann die Freigabe durch eine der Personen erfolgen. Es handelt sich um eine *ODER* Liste.

</aside>

## Freigabeprozess anwenden

Sobald die einzelnen Freigabeprozesse im Hintergrund **angelegt und aktiv** sind, werden sie in den entsprechenden Kalkulationen **direkt angewandt**. Im linken oberen Eck erscheint ein Zeichen für den Freigabeprozess mit der entsprechenden Farbe des Freigabestatus. Ist das Icon **gelb** hinterlegt, bedeutet das, dass der Freigabeprozess **noch offen** ist. Sobald die Farbe **Grün** wird, ist der Freigabeprozess **abgeschlossen**.

![Untitled](Workflow%20Freigabeprozess/Untitled%201.png)

In diesem Schritt kann auch ein weiterer Freigabeprozess **manuell hinzugefügt** werden. Dafür klicken Sie auf das Icon für den Freigabeprozess und wählen im Drop-Down-Menü den Freigabeprozess, den Sie noch hinzufügen möchten.

![Untitled](Workflow%20Freigabeprozess/Untitled%202.png)

## Dokument freigeben

Um ein Dokument freizugeben, navigieren Sie zu `Projekte → Freigaben`. Es öffnet sich eine Liste mit allen Dokumenten, die auf eine Freigabe warten. Filtern Sie nach dem **Verantwortlichen** um eine Liste der Rechnungen zu bekommen, für die **Sie verantwortlich** sind. 

![Untitled](Workflow%20Freigabeprozess/Untitled%203.png)

Sie können das Dokument entweder mit der **Schnellfreigabe** über den **Haken** direkt freigeben oder den Freigabeprozess auf der rechten Seite mit dem Icon öffnen.

Sie sehen, welche Freigabeprozesse bei der Rechnung **angewandt wurden** bzw. können noch weitere Prozesse hinzufügen. 

Um den offenen Prozess freigeben zu können, öffnen Sie den Freigabeprozess. Sie sehen dann die Informationen zu dem Prozess und können diesen kommentieren und mit dem Button `Freigeben` den Prozess abschließen.

![Untitled](Workflow%20Freigabeprozess/Untitled%204.png)

Sie können sich vor der Freigabe das Dokument auch **nochmal genau ansehen**. Dazu klicken Sie in der Liste auf das **Dokument**, das Sie öffnen wollen.

Sie gelangen in die **Kalkulation** des Dokuments, wo sie alle Positionen **bearbeiten** und **anpassen** können. Um Ihren Schritt dann freigeben zu können, klicken Sie erneut auf das **Freigabe-Icon** und öffnen Sie die Freigabemaske, indem Sie den gewünschten Freigabeprozess auswählen. 

Im nächsten Schritt können Sie dann wie oben beschrieben wieder ein Kommentar schreiben und mit `+ Kommentar`  hinzufügen. Mit dem Button `Freigabe` geben Sie den Schritt frei. 

![Untitled](Workflow%20Freigabeprozess/Untitled%205.png)

## Kontrolle Freigabe

In den einzelnen **Übersichtslisten** der Dokumente (z.B. Ausgangsrechnungen, Angebote,…) sehen Sie den **aktuellen Stand** einer Rechnung und können auch den **bisherigen Prozess** verfolgen. Wenn Sie auf das **Freigabe-Icon** klicken, können Sie sehen in welchem **Freigabeschritt** sich das Dokument befindet oder **wann** und **von wem** das Dokument freigegeben wurde. Der **Freigabeprozess** kann mit dem **runden Pfeil** auch einen Schritt **zurückgesetzt** werden oder **gelöscht** werden.

![Untitled](Workflow%20Freigabeprozess/Untitled%206.png)

## Widget *Meine Freigaben*

Allen Benutzern und Benutzerinnen steht in Poool ein Widget “Meine Freigaben” zur Verfügung. Darüber können Mitarbeiter und Mitarbeiterinnen Ihre ausstehenden Freigaben einsehen und direkt bearbeiten. Das Widget kann durch Doppelklick auf dem Dashboard hinzugefügt werden.

![Untitled](Workflow%20Freigabeprozess/Untitled%207.png)

**Wichtig**: Sollten Freigaben für Bereiche angefordert werden, auf die kein Zugriff besteht, werden sensible Informationen wie z.B. Summen ausgeblendet und der Benutzer informiert. In diesen Fällen sollte der Workflow Administrator einschreiten und die Freigabe bzw. dem Benutzer die entsprechenden Berechtigungen erteilen.

## Berechtigungen

Für mehr Kontrolle über Ihre Freigaben stehen spezielle Berechtigungen zur Verfügung. Grundsätzlich können alle Mitarbeiter Freigaben erteilen. Ausgewählte Mitarbeiter können über spezielle Berechtigungen alle ausstehenden Freigaben einsehen (*Management*) und diese ggf. auch direkt bearbeiten (*Administration*). Die Berechtigungen erteilen Sie unter  `Benutzer → Rollen` / `Benutzer → Verwaltung`. Freigabeadministratoren können alle Freigaben erteilen bzw. Freigabeprozesse zurücksetzen. Wir empfehlen im Unternehmen min. eine Person als Freigabeadministrator zu bestimmen, um steckende Prozesse anstoßen zu können.

![Untitled](Workflow%20Freigabeprozess/Untitled%208.png)